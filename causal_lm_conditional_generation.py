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
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.
from common_utils import check_data_args, common_setup, activate_peft, check_token_lengths, handle_data_sizes, train, predict
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
import pandas as pd
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry



logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def get_metric_report_str(trainer, metrics):
    loss_str = "eval_loss" if "eval_loss" in metrics else "train_loss"
    try:
        perplexity = math.exp(metrics[loss_str])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
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
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
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

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
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
    input_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the input."},
    )
    output_column_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the output."},
    )
    prediction_column_name: Optional[str] = field(
        default="output", metadata={"help": "The name of the column to write the predictions to. Will throw errors if column already exists."}
    )
    log_file: Optional[str] = field(
        default="clm_ft.log", metadata={"help": "The file to write special logs to."}
    )
    check_tok_count: bool = field(
        default=False, metadata={"help": "Check the token count of the dataset."}
    )
    clear_log: bool = field(
        default=False, metadata={"help": "If True then deletes previous special log_file"}
    )
    metric: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation. If None, will be inferred from the dataset.", 
                                "choices": [""]} #TODO: Add choices
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_seq_length: Optional[int] = field(
        default=280, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        check_data_args(self)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    special_logging, last_checkpoint, raw_datasets = common_setup(model_args, data_args, training_args)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code, device_map="auto")
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    model = activate_peft(model_args, model, tokenizer, TaskType.CAUSAL_LM)
    if "train" in raw_datasets:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["test"].features)

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    text_column_name = data_args.input_column_name
    output_column_name = data_args.output_column_name
    if not data_args.predict_only:
        assert output_column_name is not None, "You need to have an output_column for training conditional LM generation"

    check_token_lengths(data_args, raw_datasets, training_args, special_logging)

    
    additional_tokens_on_tokenize = len(tokenizer("hello")['input_ids']) - 1 
    if additional_tokens_on_tokenize != 1:
        special_logging.warning(f"Tokenizer adds {additional_tokens_on_tokenize} tokens to the input. This is unexpected.")
        logger.warning(f"Tokenizer adds {additional_tokens_on_tokenize} tokens to the input. This is unexpected.")

    def preprocess_function(examples):
        # remove pairs where at least one record is None
        inputs = examples[text_column_name]
        if output_column_name in examples:
            targets = examples[output_column_name] if output_column_name is not None else ""
            to_tok = [inputs[i] + targets[i] for i in range(len(inputs))]
        else:
            to_tok = inputs
        model_inputs = tokenizer(to_tok, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        if output_column_name in examples:
            labels = model_inputs["input_ids"].copy()
            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            labels = [
                    [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
            # if targets is not None then replace all input_only_ids with -100 to ignore them in the loss
            input_only_ids = tokenizer(inputs, max_length=data_args.max_seq_length, padding='do_not_pad', truncation=True)['input_ids']
            inp_lengths = [len(i) for i in input_only_ids]
            for i in range(len(labels)):
                length = inp_lengths[i]
                for j in range(length):
                    labels[i][j] = -100
            model_inputs["labels"] = labels
        return model_inputs

    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    # if test is in raw_datasets, then check if the output_column_name is in the dataset and if it is remove it
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=column_names,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenizing dataset",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                remove_columns=column_names,
                batched=True,
            )

    train_dataset, eval_dataset, internal_eval_dataset, test_dataset = handle_data_sizes(data_args, training_args, lm_datasets)

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir) # TODO: Look at appropriate metric

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=internal_eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )
    if not data_args.predict_only:
        train(training_args, trainer, last_checkpoint, train_dataset, eval_dataset, special_logging)
    
    predict(data_args, trainer, test_dataset, special_logging)    


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()