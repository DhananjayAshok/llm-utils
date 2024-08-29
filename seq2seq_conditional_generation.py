#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
from common_utils import check_data_args, common_setup, activate_peft, check_token_lengths, handle_data_sizes, train, predict
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
#nltk.download('punkt_tab')
import numpy as np
import pandas as pd
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from peft import LoraConfig, TaskType, get_peft_model, IA3Config

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def get_metric_report_str(trainer, metrics):
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s


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
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_input_length` exceeds "
                "the model's position embeddings."
            )
        },
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
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
    input_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the input."},
    )
    output_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the output."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_input_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_output_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_output_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_output_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
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
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    metric: Optional[str] = field(
        default="exact_match", metadata={"help": "The metric to use for evaluation.", 
                                "choices": ["bleu", "rogue", "exact_match"]}
    )

    def __post_init__(self):
        if self.val_max_output_length is None:
            self.val_max_output_length = self.max_output_length
        check_data_args(self)



def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    special_logging, last_checkpoint, raw_datasets = common_setup(model_args, data_args, training_args)

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained( #TODO: Get various models in here. 
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto"
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_input_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_input_length}."
            )
            model.resize_position_embeddings(data_args.max_input_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_input_length)
        else:
            raise ValueError(
                f"`--max_input_length` is set to {data_args.max_input_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_input_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

    model = activate_peft(model_args, model, tokenizer, TaskType.SEQ_2_SEQ_LM)


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.

    # Get the column names for input/target.
    text_column = data_args.input_column_name
    summary_column = data_args.output_column_name
    if not data_args.predict_only:
        assert summary_column is not None, "You need to have an output_column for training Seq2Seq"

    # Temporarily set max_output_length for training.
    max_output_length = data_args.max_output_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # if summary_column is in the frame then remove pairs where at least one record is None, otherwise dont have labels
        inputs = examples[text_column]
        targets = None
        if summary_column in examples:
            targets = examples[summary_column]

        model_inputs = tokenizer(inputs, max_length=data_args.max_input_length, padding=padding, truncation=True)

        if targets is not None:
            # Tokenize targets with the `text_target` keyword argument
            labels = tokenizer(text_target=targets, max_length=max_output_length, padding=padding, truncation=True)

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            if padding == "max_length" and data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]

            model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = None
    eval_dataset = None
    internal_eval_dataset = None
    test_dataset = None
    if "train" in raw_datasets:
        column_names = raw_datasets["train"].column_names
        train_dataset = raw_datasets["train"].shuffle(seed=training_args.seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )
    if "validation" in raw_datasets:
        max_output_length = data_args.val_max_output_length
        column_names = raw_datasets["validation"].column_names
        eval_dataset = raw_datasets["validation"].shuffle(seed=training_args.seed)
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
        internal_eval_dataset = None
        if data_args.max_internal_eval_samples is not None:
            max_internal_eval_samples = min(len(eval_dataset), data_args.max_internal_eval_samples)
            internal_eval_dataset = eval_dataset.select(range(max_internal_eval_samples))
        if internal_eval_dataset is None:
            internal_eval_dataset = eval_dataset

    test_dataset = None
    if "test" in raw_datasets:
        test_dataset = raw_datasets["test"]
        max_output_length = data_args.val_max_output_length
        if data_args.max_test_samples is not None:
            max_test_samples = min(len(test_dataset), data_args.max_test_samples)
            test_dataset = test_dataset.select(range(max_test_samples))
        with training_args.main_process_first(desc="test dataset map pre-processing"):
            test_dataset = test_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on test dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric TODO: Customize
    metric = evaluate.load(data_args.metric, cache_dir=model_args.cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        if data_args.metric == "rogue":
            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        kwargs = {}
        if data_args.metric == "rouge":
            kwargs = {"use_stemmer": True}
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, **kwargs)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_output_length
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=internal_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    train(training_args, trainer, last_checkpoint, train_dataset, eval_dataset, special_logging)
    predict(data_args, trainer, test_dataset, special_logging)
    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()