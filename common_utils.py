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

def get_metric_report_str(trainer, metrics):
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s

def common_setup(model_args, data_args, training_args):
    training_args.do_train = True
    training_args.do_eval = True
    training_args.predict_with_generate = True
    training_args.eval_strategy = "epoch"
    training_args.auto_find_batch_size  = True
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 2
    if training_args.seed is None:
        training_args.seed = 42

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not data_args.grid_log:
        if os.path.exists(data_args.log_file):
            os.remove(data_args.log_file)
    logdir = os.path.dirname(data_args.log_file)
    if logdir != "" and not os.path.exists(logdir):
        os.makedirs(logdir)
    special_logging = logging.getLogger("special")
    special_logging.setLevel(logging.DEBUG)
    handler = logging.FileHandler(data_args.log_file)
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    special_logging.addHandler(handler)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(transformers.logging.ERROR)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Training hyperparameters
    special_logging.info(f"*** Hyperparameters ***")
    special_logging.info(f"learning_rate: {training_args.learning_rate}")
    if training_args.learning_rate is None or training_args.learning_rate == 5e-05:
        special_logging.warning("\tUsing default learning rate (5e-5). Should likely mess around with it")
    special_logging.info(f"weight_decay: {training_args.weight_decay}")
    if training_args.weight_decay is None or training_args.weight_decay == 0.0:
        special_logging.warning("\tUsing default weight decay (0.0). Should likely mess around with it")
    special_logging.info(f"adam_beta1: {training_args.adam_beta1}")
    if training_args.adam_beta1 is None or training_args.adam_beta1 == 0.9:
        special_logging.warning("\tUsing default adam_beta1 (0.9). Should likely mess around with it")
    special_logging.info(f"adam_beta2: {training_args.adam_beta2}")
    if training_args.adam_beta2 is None or training_args.adam_beta2 == 0.999:
        special_logging.warning("\tUsing default adam_beta2 (0.999). Should likely mess around with it")
    special_logging.info(f"lr_scheduler_type: {training_args.lr_scheduler_type}")
    if training_args.lr_scheduler_type is None or training_args.lr_scheduler_type == "linear":
        special_logging.warning("\tUsing default lr_scheduler_type (linear). Should likely mess around with it")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    #logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined together
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Loading a dataset from your local files.
    #region
    if data_args.data_file is not None:
        df = pd.read_csv(data_args.data_file)
        df = df.sample(frac=1).reset_index(drop=True)
        n_train = int(len(df) * (1 - data_args.train_val_split))
        train_df = df[:n_train]
        val_df = df[n_train:].reset_index(drop=True)
        save_name_path = data_args.data_file.replace(".csv", "")
        train_df.to_csv(f"{save_name_path}_train.csv", index=False)
        val_df.to_csv(f"{save_name_path}_validation.csv", index=False)
        data_files = {"train": f"{save_name_path}_train.csv", "validation": f"{save_name_path}_validation.csv"}
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}
    
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file


    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )
    if "train" in raw_datasets:
        column_names = list(raw_datasets["train"].features)
    elif "test" in raw_datasets:
        column_names = list(raw_datasets["test"].features)
    else:
        raise ValueError(f"Dataset does not have train or test should not be able to reach here")
    if data_args.input_column_name is not None:
        assert data_args.input_column_name in column_names, f"Input column {data_args.input_column_name} not found in dataset with columns {column_names}"
    else:
        for name in ["text", "sentence", "input"]:
            if name in column_names:
                if data_args.input_column_name is None:
                    data_args.input_column_name = name
                else:
                    raise ValueError(f"Multiple possible input columns found: {data_args.input_column_name}, {name}. Please specify one.")
                
    if data_args.output_column_name is not None:
        assert data_args.output_column_name in column_names, f"Output column {data_args.output_column_name} not found in dataset with columns {column_names}"
    else:
        for name in ["label", "target", "output"]:
            if name in column_names:
                if data_args.output_column_name is None:
                    data_args.output_column_name = name
                else:
                    raise ValueError(f"Multiple possible output columns found: {data_args.output_column_name}, {name}. Please specify one.")
    if data_args.input_column_name is None:
        raise ValueError(f"Could not find an input text column in the dataset with columns {column_names}. Please make sure the dataset has a text column.")
    if data_args.output_column_name is None:
        pass # could be language modelling check it TODO
    if "test" in raw_datasets and data_args.output_column_name is not None and data_args.output_column_name in raw_datasets["test"].features:
        raw_datasets["test"] = raw_datasets["test"].remove_columns([data_args.output_column_name])

    if data_args.input_column_name != "text":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.input_column_name, "text")

    if data_args.output_column_name is not None and data_args.output_column_name != "label":
        for key in raw_datasets.keys():
            if key == "test":
                continue
            else:
                raw_datasets[key] = raw_datasets[key].rename_column(data_args.output_column_name, "label")
    if "train" not in raw_datasets:
        data_args.predict_only = True
    return special_logging, last_checkpoint, raw_datasets

def activate_peft(model_args, model, tokenizer, task_type):
    if model_args.peft == "lora":
        target_modules = ["k_proj", "v_proj", "down_proj"] # Will want to add new stuff here to allow in other models
        peft_config = LoraConfig(lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, r=model_args.lora_r, bias=model_args.lora_bias, task_type=task_type, target_modules=target_modules)
    elif model_args.peft == "ia3":
        peft_config = IA3Config(task_type=task_type, target_modules=target_modules, feedforward_modules=["down_proj"])
    #model.add_adapter(peft_config)
    if model_args.peft is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id
    return model


def check_token_lengths(data_args, raw_datasets, training_args, special_logging):
    if data_args.check_tok_count:
        def estimate_length(examples):
            inputs = examples[data_args.text_column_name]
            toked = [len(sent.split()) for sent in inputs]
            return {"tok_count": toked}
        
        with training_args.main_process_first(desc="estimating dataset length"):
            dataset_stats = raw_datasets.map(
                estimate_length,
                batched=True,
            )

        if data_args.predict_only:
            special_logging.info(f"*** Dataset Stats ***")
            tokens = dataset_stats["test"]["tok_count"]
            tokens.sort()
            special_logging.info(f"Test: {len(tokens)}")
            special_logging.info(f"min: {tokens[0]}")
            special_logging.info(f"max: {tokens[-1]}")
            special_logging.info(f"mean: {sum(tokens)/len(tokens)}")
            special_logging.info(f"median: {tokens[len(tokens)//2]}")
            special_logging.info(f"95th percentile: {tokens[int(len(tokens)*0.95)]}")
        else:
            special_logging.info(f"*** Dataset Stats ***")
            tokens = dataset_stats["train"]["tok_count"]
            tokens.sort()
            special_logging.info(f"Train:")
            special_logging.info(f"min: {tokens[0]}")
            special_logging.info(f"max: {tokens[-1]}")
            special_logging.info(f"mean: {sum(tokens)/len(tokens)}")
            special_logging.info(f"median: {tokens[len(tokens)//2]}")
            special_logging.info(f"95th percentile: {tokens[int(len(tokens)*0.95)]}")
            tokens = dataset_stats["validation"]["tok_count"]
            tokens.sort()
            special_logging.info(f"Validation:")
            special_logging.info(f"min: {tokens[0]}")
            special_logging.info(f"max: {tokens[-1]}")
            special_logging.info(f"mean: {sum(tokens)/len(tokens)}")
            special_logging.info(f"median: {tokens[len(tokens)//2]}")
            special_logging.info(f"95th percentile: {tokens[int(len(tokens)*0.95)]}")


def handle_data_sizes(data_args, raw_datasets):
    train_dataset = None
    eval_dataset = None
    internal_eval_dataset = None
    test_dataset = None
    if not data_args.predict_only:
        train_dataset = raw_datasets["train"]
        if data_args.shuffle_train_dataset:
            #logger.info("Shuffling the training dataset")
            train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        eval_dataset = raw_datasets["validation"]
        internal_eval_dataset = None
        if data_args.max_internal_eval_samples is not None:
            max_internal_eval_samples = min(len(eval_dataset), data_args.max_internal_eval_samples)
            internal_eval_dataset = eval_dataset.select(range(max_internal_eval_samples))
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        if internal_eval_dataset is None:
            internal_eval_dataset = eval_dataset

    if "test" in raw_datasets:
        test_dataset = raw_datasets["test"]
        if data_args.max_test_samples is not None:
            max_test_samples = min(len(test_dataset), data_args.max_test_samples)
            test_dataset = test_dataset.select(range(max_test_samples))
    return train_dataset, eval_dataset, internal_eval_dataset, test_dataset


def train(training_args, trainer, last_checkpoint, train_dataset, eval_dataset, special_logging):
    special_logging.info("*** Before Training Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    trainer.log_metrics("train", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"train metrics: {readable}")


    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"eval metrics: {readable}")



    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    special_logging.info("*** Final Evaluation ***")
    logging.info("*** Final Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"train metrics: {readable}")


    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"eval metrics: {readable}")


def predict(trainer, dataset, save_path):
    predictions = trainer.predict(dataset, metric_key_prefix="predict").predictions