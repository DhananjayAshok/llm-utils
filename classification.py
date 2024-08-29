#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.
from common_utils import check_data_args, common_setup, activate_peft, check_token_lengths, handle_data_sizes, train, predict
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import List, Optional
import warnings

import datasets
import evaluate
import numpy as np
import pandas as pd
from datasets import Value, load_dataset

import torch
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, IA3Config

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version



logger = logging.getLogger(__name__)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        #loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([100, 0.01]).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def get_metric_report_str(trainer, metrics):
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    input_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file. "
            )
        },
    )
    output_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file. "
            )
        },
    )
    max_seq_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=True, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_internal_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of internal evaluation examples to this "
                "value if set."
            )
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    data_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    train_val_split: Optional[float] = field(
        default=0.1,
        metadata={"help": "The ratio of the val subset of the dataset to the training subset"},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the test data."}
    )
    prediction_file: Optional[str] = field(
        default=None, metadata={"help": "CSV file to write the predictions to."}
    )
    prediction_column_name: Optional[str] = field(
        default="output", metadata={"help": "The name of the column to write the predictions to. Will throw errors if column already exists."}
    )
    print_examples: bool = field(
        default=False, metadata={"help": "Print some examples to logger to check data."}
    )
    log_file: Optional[str] = field(
        default="clf_ft.log", metadata={"help": "The file to write special logs to."}
    )
    clear_log: bool = field(
        default=False, metadata={"help": "If True then deletes previous special log_file"}
    )
    metric: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation. If None, will be inferred from the dataset.", 
                                "choices": ["accuracy", "precision", "recall", "f1", "mse"]}
    )
    check_tok_count: bool = field(
        default=False, metadata={"help": "Check token count stats of the dataset."}
    )
    


    def __post_init__(self):
        assert self.max_seq_length is not None
        check_data_args(self)



@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    peft: str = field(
        default=None,
        metadata={"help": "Kind of PEFT to use. Options are None, lora, ia3."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha value to use for LoRA."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout value to use for LoRA."},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "The r value to use for LoRA."},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "The bias value to use for LoRA."},
    )    


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a multi-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    special_logging, last_checkpoint, raw_datasets = common_setup(model_args, data_args, training_args)

    # Set data processing
    #region
    if not data_args.predict_only:
        is_regression = (
            raw_datasets["train"].features["label"].dtype in ["float32", "float64"]
            if data_args.do_regression is None
            else data_args.do_regression
        )

        is_multi_label = False
        if is_regression:
            label_list = None
            num_labels = 1
            # regession requires float as label type, let's cast it if needed
            for split in raw_datasets.keys():
                if raw_datasets[split].features["label"].dtype not in ["float32", "float64"]:
                    logger.warning(
                        f"Label type for {split} set to float32, was {raw_datasets[split].features['label'].dtype}"
                    )
                    features = raw_datasets[split].features
                    features.update({"label": Value("float32")})
                    try:
                        raw_datasets[split] = raw_datasets[split].cast(features)
                    except TypeError as error:
                        logger.error(
                            f"Unable to cast {split} set to float32, please check the labels are correct, or maybe try with --do_regression=False"
                        )
                        raise error

        else:  # classification
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            # Trying to find the number of labels in a multi-label classification task
            # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
            # So we build the label list from the union of labels in train/val/test.
            label_list = get_label_list(raw_datasets, split="train")
            for split in ["validation", "test"]:
                if split in raw_datasets:
                    val_or_test_labels = get_label_list(raw_datasets, split=split)
                    diff = set(val_or_test_labels).difference(set(label_list))
                    if len(diff) > 0:
                        # add the labels that appear in val/test but not in train, throw a warning
                        logger.warning(
                            f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                        )
                        label_list += list(diff)
            # if label is -1, we throw a warning and remove it from the label list
            for label in label_list:
                if label == -1:
                    logger.warning("Label -1 found in label list, removing it.")
                    label_list.remove(label)

            label_list.sort()
            num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")
        #endregion

        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        #region
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="text-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        if is_regression:
            config.problem_type = "regression"
            logger.info("setting problem type to regression")
        elif is_multi_label:
            config.problem_type = "multi_label_classification"
            logger.info("setting problem type to multi label classification")
        else:
            config.problem_type = "single_label_classification"
            #logger.info("setting problem type to single label classification")
    else:
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path) # Can set to None
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        device_map="auto"
    )

    model = activate_peft(model_args, model, tokenizer, TaskType.SEQ_CLS)

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if not data_args.predict_only:
        # for training ,we will update the config with label infos,
        # if do_train is not set, we will use the label infos in the config
        if not is_regression:  # classification, training
            label_to_id = {v: i for i, v in enumerate(label_list)}
            # update config with label infos
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in label_to_id.items()}
        elif not is_regression:  # classification, but not training
            #logger.info("using label infos in the model config")
            #logger.info("label2id: {}".format(model.config.label2id))
            label_to_id = model.config.label2id
        else:  # regression
            label_to_id = None

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the "
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def multi_labels_to_ids(labels: List[str]) -> List[float]:
        ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
        for label in labels:
            ids[label_to_id[label]] = 1.0
        return ids

    check_token_lengths(data_args, raw_datasets, training_args, special_logging)

    def preprocess_function(examples):
        # Tokenize the texts
        result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=True)
        if label_to_id is not None and "label" in examples:
            if is_multi_label:
                result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            else:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
        return result

    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        raw_datasets = raw_datasets.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )

    train_dataset, eval_dataset, internal_eval_dataset, test_dataset = handle_data_sizes(data_args, training_args, raw_datasets)


    # Log a few random samples from the training set:
    if data_args.print_examples:
        for index in random.sample(range(len(train_dataset)), 3):
            special_logging.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if data_args.metric is not None:
        metric = evaluate.load(data_args.metric, cache_dir=model_args.cache_dir)
    else:
        if is_regression:
            metric = evaluate.load("mse", cache_dir=model_args.cache_dir)
        else:
            if is_multi_label:
                metric = evaluate.load("f1", config_name="multilabel", cache_dir=model_args.cache_dir)
            else:
                metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
            result = metric.compute(predictions=preds, references=p.label_ids)
        elif is_multi_label:
            preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
            # Micro F1 is commonly used in multi-label classification
            result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
        else:
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
    # we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=internal_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    if not data_args.predict_only:
        # Training
        train(training_args, trainer, last_checkpoint, train_dataset, eval_dataset, special_logging)
    predict(data_args, trainer, test_dataset, special_logging)

if __name__ == "__main__":
    main()
