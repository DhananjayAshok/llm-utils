#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path

import datasets
import torch
import pandas as pd
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
import evaluate
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    DataCollatorForSeq2Seq,
    default_data_collator,
    get_scheduler,
)
from peft import LoraConfig, TaskType, get_peft_model, IA3Config

logger = get_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--task", # string of either classification, regression or clm
        type=str,
        required=True,
        help="The task to use the model for.",
        choices=["classification", "regression", "clm", "lm"],
    )
    parser.add_argument(
        "--metric", 
        type=str,
        default=None,
        help="The metric to use for evaluation. Only applicable for classification and regression tasks.",
        choices=["accuracy", "f1", "precision", "recall", "mse", "rmse", "mae", "pearson", "spearmanr", "kendall"],
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=None,
        help="The name of dataset to use. The dataset should be in the Datasets Hub.",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv file containing the validation data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=0.05,
        help="The percentage of the data_file used as validation set in case there's no validation split. Specify as float decimal",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--lm_type", 
        type=str,
        default=None,
        help="The kind of Language Modelling used for model. Can be either CausalLM or Seq2SeqLM.",
        choices=["causal", "seq2seq"]
    )
    # add arguments, max_train_samples, max_eval_samples, max_internal_eval_samples
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, limit the number of training examples.",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="For debugging purposes or quicker training, limit the number of evaluation examples.",
    )
    parser.add_argument(
        "--max_internal_eval_samples",
        type=int,
        default=None,
        help="For quicker training, limit the number of internal evaluation examples during epochs.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument( #TODO: WOULD LIKE TO MAKE THIS AUTOINFERRED
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str, #TODO: Check this to make it work for classification too
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_output_length",
        type=int,
        default=1024,
        help="The maximum total sequence length for target text after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens). Used only for lm task"
        ),
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps", # TODO: Look into this add steps
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    # add evaluation every
    parser.add_argument(
        "--epochs_per_eval",
        type=float,
        default=0.3,
        help="Will evaluate every epochs_per_eval * n_steps_per_epoch steps. Must be between 0 and 1",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--low_cpu_mem_usage",
        action="store_true",
        help=(
            "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
            "If passed, LLM loading time and RAM consumption will be benefited."
        ),
    )
    parser.add_argument(
        "--peft",
        type=str,
        default=None,
        help="Kind of PEFT to use. Options are None, lora, ia3.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha value to use for LoRA.",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="The dropout value to use for LoRA.",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=64,
        help="The r value to use for LoRA.",
    )   
    parser.add_argument(
        "--lora_bias",
        type=str,
        default="none",
        help="The bias value to use for LoRA.",
    )

    args = parser.parse_args()

    # Sanity checks
    if args.data_file is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a datapath or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv"]:
                raise ValueError("`train_file` should be a csv")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv"]:
                raise ValueError("`validation_file` should be a csv")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")
        
    if args.task == "lm":
        if args.lm_type is None:
            # attempt to auto infer the language model type:
            seq2seqs = ["t5", "bart", "pegasus", "marianmt"]
            causal_lms = ["gpt", "ctrl", "xlnet", "llama", "vicuna", "falcon", "mpt", "mistral"]
            for model_name in seq2seqs:
                if model_name in args.model_name_or_path.lower():
                    args.lm_type = "seq2seq"
                    break
            if args.lm_type is None:
                for model_name in causal_lms:
                    if model_name in args.model_name_or_path.lower():
                        args.lm_type = "causal"
                        break
            if args.lm_type is None:
                raise ValueError(f"Could not infer LM type from model name: {args.model_name_or_path}. Specify `--lm_type`.")
    if args.task == "classification":
        if args.metric is None:
            args.metric = "accuracy"
        else:
            assert args.metric in ["accuracy", "f1", "precision", "recall"], f"Unsupported metric for classification task: {metric}"
    if args.task == "regression":
        if args.metric is None:
            args.metric = "mse"
        else:
            assert args.metric in ["mse", "rmse", "mae", "pearson", "spearmanr", "kendall"], f"Unsupported metric for regression task: {metric}"
    assert args.epochs_per_eval > 0 and args.epochs_per_eval <= 1, "epochs_per_eval must be between 0 and 1"
    return args


def eval_loop(args, model, eval_dataloader, metric, accelerator, train_loss, epoch, completed_steps, force_full=False):
    model.eval()
    losses = []
    n_evals_done = 0
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        if not force_full and args.max_internal_eval_samples is not None and n_evals_done >= args.max_internal_eval_samples:
            break
        with torch.no_grad():
            outputs = model(**batch)
        if args.task == "classification":
            predictions = outputs.logits.argmax(dim=-1)
        elif args.task == "regression":
            predictions = outputs.logits.squeeze()
        if args.task in ["classification", "regression"]:
            predictions, references = accelerator.gather((predictions, batch["labels"]))
                        # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(eval_dataloader) - 1:
                    predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                    references = references[: len(eval_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))
        n_evals_done += len(batch["input_ids"])

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
    except OverflowError:
        eval_loss = float("inf")
    eval_metric = None
    if args.task == "lm":
        # if eval_loss is not inf compute it
        if eval_loss != float("inf"):
            eval_metric = {"perplexity": math.exp(eval_loss)}
        else:
            eval_metric = {"perplexity": float("inf")}
    elif metric is not None:
        eval_metric = metric.compute()
    if eval_loss != float("inf"):
        eval_loss = eval_loss.item()
    if args.with_tracking:
        log_dict = {
                    "train_loss": train_loss,
                    "eval_loss": eval_loss.item(),
                    "epoch": epoch}
        if eval_metric is not None:
            for key in eval_metric:
                log_dict[key] = eval_metric[key]
        accelerator.log(log_dict, step=completed_steps)
    eval_metric_str = str(eval_metric) if eval_metric is not None else ""
    logger.info(f"epoch (step) {epoch} ({completed_steps}): train_loss: {train_loss} eval_loss: {eval_loss} {eval_metric_str}")

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you must provide either a data csv file or train, validation files. 
    # If you provide data csv file it will be split into _train and _validation as per the validation_split_percentage
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    extension = "csv"
    dataset_args = {}
    if args.train_file is None and args.validation_file is None:
        assert args.data_file is not None, "Need a data file to split into train and validation"
        df = pd.read_csv(args.data_file)
        val_size = int(len(df) * args.validation_split_percentage)
        train_size = len(df) - val_size
        df = df.sample(frac=1).reset_index(drop=True)
        train_df = df[:train_size]
        val_df = df[train_size:].reset_index(drop=True)
        train_file = args.data_file.replace(".csv", "_train.csv")
        val_file = args.data_file.replace(".csv", "_validation.csv")
        train_df.to_csv(train_file, index=False)
        val_df.to_csv(val_file, index=False)
        args.train_file = train_file
        args.validation_file = val_file
    data_files["train"] = args.train_file
    data_files["validation"] = args.validation_file
    raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    column_names = raw_datasets["train"].column_names
    text_column_name = None
    priority_order = ["text", "sentence", "input_text", "input_sentence", "input"]
    for col in priority_order:
        if col in column_names:
            text_column_name = col
            break
    target_column_name = None
    priority_order = ["label", "target", "output"]
    for col in priority_order:
        if col in column_names:
            target_column_name = col
            break
    if text_column_name is None:
        raise ValueError("Could not find a text column in the dataset.")
    if target_column_name is None and args.task in ["classification", "regression"]:
        raise ValueError("Could not find a target column in the dataset for classification or regression task.")
    if args.task == "classification":
        Model_Class = AutoModelForSequenceClassification
        num_labels = len(raw_datasets["train"].unique(target_column_name))
    elif args.task == "regression":
        Model_Class = AutoModelForSequenceClassification
        num_labels = 1
    else:
        if args.lm_type == "causal":
            Model_Class = AutoModelForCausalLM
        elif args.lm_type == "seq2seq":
            Model_Class = AutoModelForSeq2SeqLM
        else:
            raise ValueError("Language Model type not supported.")
    if args.config_name:
        if args.task in ["classification", "regression"]:
            config = AutoConfig.from_pretrained(
                args.config_name,
                num_labels=num_labels,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                args.config_name,
                trust_remote_code=True,
            )
    elif args.model_name_or_path:
        if args.task in ["classification", "regression"]:
            config = AutoConfig.from_pretrained(
                args.model_name_or_path,
                num_labels=num_labels,
                trust_remote_code=True,
            )
        else:
            config = AutoConfig.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
            )
    else:
        config = CONFIG_MAPPING[args.model_type]() #TODO: Check this works with classification
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=True
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=True
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    
    if args.model_name_or_path:
        model = Model_Class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            trust_remote_code=True,
            #device_map="auto" #TODO: Check device_map
        )
    else:
        logger.info("Training new model from scratch") #TODO: Check this works with classification
        model = Model_Class.from_config(config, trust_remote_code=True)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if args.peft is not None:
        if args.task == "classification":
            task_type = TaskType.CLS
        elif args.task == "regression":
            task_type = TaskType.CLS
        else:
            if args.lm_type == "causal":
                task_type = TaskType.CAUSAL_LM
            elif args.lm_type == "seq2seq":
                task_type = TaskType.SEQ_2_SEQ
            else:
                raise ValueError("Language Model type not supported.")
        target_modules = ["k_proj", "v_proj", "down_proj"] # TODO: Add other model modules if desired
        feedforward_modules = ["down_proj"] # TODO: Add other model modules if desired
        if args.peft == "lora":
            peft_config = LoraConfig(lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, r=args.lora_r, bias=args.lora_bias, task_type=task_type, target_modules=target_modules)
        elif args.peft == "ia3":
            peft_config = IA3Config(task_type=TaskType.SEQ_CLS, target_modules=target_modules, feedforward_modules=feedforward_modules)
            #model.add_adapter(peft_config)
        if args.peft is not None:
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.task != "lm":
        padding = "max_length" #TODO: Doing dynamic padding now might want to switch to max
        def preprocess_function(examples):
            inputs = examples[text_column_name]
            model_inputs = tokenizer(inputs, max_length=args.max_input_length, padding=padding, truncation=True)

            if args.task == "clm":
                # Tokenize targets with the `text_target` keyword argument
                targets = examples[target_column_name]
                labels = tokenizer(text_target=targets, max_length=args.max_output_length, padding=padding, truncation=True)

                # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
                # padding in the loss.
                if padding == "max_length" and args.ignore_pad_token_for_loss:
                    labels["input_ids"] = [
                        [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                    ]

                model_inputs["labels"] = labels["input_ids"]
            else:
                model_inputs["labels"] = examples[target_column_name]
            return model_inputs
        with accelerator.main_process_first():
            processed_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                desc="Running tokenizer on dataset",
            )
    else:
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name])

        with accelerator.main_process_first():
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

        if args.block_size is None:
            block_size = tokenizer.model_max_length
            if block_size > config.max_position_embeddings:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    f"Using block_size={min(1024, config.max_position_embeddings)} instead. You can change that default value by passing --block_size xxx."
                )
                block_size = min(1024, config.max_position_embeddings)
        else:
            if args.block_size > tokenizer.model_max_length:
                logger.warning(
                    f"The block_size passed ({args.block_size}) is larger than the maximum length for the model "
                    f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
                )
            block_size = min(args.block_size, tokenizer.model_max_length)

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result

        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
        # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
        # to preprocess.
        #
        # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
        # https://huggingface.co/docs/datasets/process#map

        with accelerator.main_process_first():
            processed_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    train_dataset = train_dataset.shuffle(seed=args.seed)
    eval_dataset = eval_dataset.shuffle(seed=args.seed)
    if args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    if args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))

    # DataLoaders creation:
    collator = default_data_collator
    if args.task=="clm" and args.lm_type == "seq2seq":
        collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collator, batch_size=args.per_device_train_batch_size # TODO: Dynamic batch size here
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collator, batch_size=args.per_device_eval_batch_size
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    metric = None
    if args.task == "classification" or args.task == "regression":
        metric = evaluate.load(args.metric)
    elif args.task == "lm":
        args.metric = "perplexity"

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers(f"{args.task}_{args.train_file.replace('.csv', '').split('/')[-1]}", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            checkpoint_path = args.resume_from_checkpoint
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            checkpoint_path = path
            path = os.path.basename(checkpoint_path)

        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.load_state(checkpoint_path)
        # Extract `epoch_{i}` or `step_{i}`
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            resume_step = None
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            # need to multiply `gradient_accumulation_steps` to reflect real steps
            resume_step = int(training_difference.replace("step_", "")) * args.gradient_accumulation_steps
            starting_epoch = resume_step // len(train_dataloader)
            completed_steps = resume_step // args.gradient_accumulation_steps
            resume_step -= starting_epoch * len(train_dataloader)

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    eval_steps = int(args.epochs_per_eval * num_update_steps_per_epoch)
    for epoch in range(starting_epoch, args.epochs):
        model.train()
        latest_loss = 0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss # TODO: Insert label weighted loss here
                # We keep track of the loss at each epoch
                latest_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            
            if args.epoch_per_eval < 1 and completed_steps % eval_steps == 0:
                train_loss = latest_loss.item() / eval_steps
                eval_loop(args, epoch, model, eval_dataloader, metric, accelerator, train_loss, epoch, completed_steps, train_loss=train_loss)
                model.train()

        if args.push_to_hub and epoch < args.epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
                api.upload_folder(
                    commit_message=f"Training in progress epoch {epoch}",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )

        if args.checkpointing_steps == "epoch":
            output_dir = f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
    eval_loop(args, epoch, model, eval_dataloader, metric, accelerator, train_loss, epoch, completed_steps, train_loss=train_loss, force_full=True)
    if args.with_tracking:
        accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )


if __name__ == "__main__":
    main()