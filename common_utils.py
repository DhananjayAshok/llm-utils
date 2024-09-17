import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, disable_caching
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
import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from peft import LoraConfig, get_peft_model, IA3Config

disable_caching()
logger = logging.getLogger(__name__)

def get_metric_report_str(metric_report):
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s

def do_evaluation(trainer, dataset, split, special_logging):
    with torch.no_grad():
        metrics = trainer.evaluate(eval_dataset=dataset, metric_key_prefix=split)
    trainer.log_metrics(split, metrics)
    metric_report = trainer.metrics_format(metrics)
    readable = get_metric_report_str(metric_report)
    special_logging.info(f"{split} metrics: \n{readable}")
    return metric_report

def check_data_args(data_args):
    assert (data_args.train_file is None) == (data_args.validation_file is None), "Cannot use --train_file without --validation_file"
    if data_args.data_file is not None:
        assert data_args.train_file is None, "Cannot use --data_file with --train_file, --validation_file or --test_file"
    elif data_args.train_file is None:
        assert data_args.test_file is not None, "Must specify --test_file if not using --data_file or --train_file and --validation_file"

    if data_args.data_file is not None:
        assert data_args.data_file.split(".")[-1] == "csv", "Data file must be a csv"

    if data_args.train_file is not None:
        assert data_args.train_file.split(".")[-1] == "csv", "Train file must be a csv"

    if data_args.validation_file is not None:
        assert data_args.validation_file.split(".")[-1] == "csv", "Validation file must be a csv"

    if data_args.test_file is not None:
        assert data_args.test_file.split(".")[-1] == "csv", "Test file must be a csv"

    if data_args.test_file is not None:
        assert data_args.prediction_file is not None, "Must specify --prediction_file if using --test_file"

    logdir = os.path.dirname(data_args.log_file)
    if logdir != "" and not os.path.exists(logdir):
        os.makedirs(logdir)
    if data_args.prediction_file is not None:
        prediction_dir = os.path.dirname(data_args.prediction_file)
        if prediction_dir != "" and not os.path.exists(prediction_dir):
            os.makedirs(prediction_dir)
        if os.path.exists(data_args.prediction_file):
            if data_args.prediction_column_name in pd.read_csv(data_args.prediction_file).columns:
                raise ValueError(f"Prediction column already exists in prediction file {data_args.prediction_file}. Please specify a different column name or different file")


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
    if data_args.clear_log:
        if os.path.exists(data_args.log_file):
            os.remove(data_args.log_file)
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
        data_files = {}
    
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file
    if data_args.validation_file is not None:
        data_files["validation"] = data_args.validation_file
    if data_args.test_file is not None:
        data_files["test"] = data_args.test_file


    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=None,
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
        special_logging.warn(f"Could not find an output label column in the dataset with columns {column_names}. This is okay for language modelling, just be sure.")
        pass

    if data_args.input_column_name != "text":
        for key in raw_datasets.keys():
            raw_datasets[key] = raw_datasets[key].rename_column(data_args.input_column_name, "text")

    if data_args.output_column_name is not None and data_args.output_column_name != "label":
        for key in raw_datasets.keys():
            if data_args.output_column_name in raw_datasets[key].features:
                raw_datasets[key] = raw_datasets[key].rename_column(data_args.output_column_name, "label")
    data_args.input_column_name = "text"
    data_args.output_column_name = "label"
    if "train" not in raw_datasets:
        data_args.predict_only = True
    else:
        data_args.predict_only = False
    return special_logging, last_checkpoint, raw_datasets

def activate_peft(model_args, model, tokenizer, task_type):
    if model_args.peft == "lora":
        if "llama".lower() in model_args.model_name_or_path.lower():
            target_modules = ["k_proj", "v_proj", "down_proj"] # Will want to add new stuff here to allow in other models
            peft_config = LoraConfig(lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, r=model_args.lora_r, bias=model_args.lora_bias, task_type=task_type, target_modules=target_modules)
        elif "mistral".lower() in model_args.model_name_or_path.lower():
            peft_config = LoraConfig(lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, r=model_args.lora_r, bias=model_args.lora_bias, task_type=task_type)
        else:
            raise ValueError(f"Model {model_args.model_name_or_path} not supported for Lora")
    elif model_args.peft == "ia3":
        if "llama".lower() in model_args.model_name_or_path.lower():
            target_modules = ["k_proj", "v_proj", "down_proj"]
            peft_config = IA3Config(task_type=task_type, target_modules=target_modules, feedforward_modules=["down_proj"])
        elif "mistral".lower() in model_args.model_name_or_path.lower():
            peft_config = IA3Config(task_type=task_type)
            print(f"IA3 config: {peft_config} UNCHECKED ON MISTRAL")
        else:
            raise ValueError(f"Model {model_args.model_name_or_path} not supported for IA3")

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


def handle_data_sizes(data_args, training_args, raw_datasets):
    train_dataset = None
    eval_dataset = None
    internal_eval_dataset = None
    test_dataset = None
    if not data_args.predict_only:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.shuffle(seed=training_args.seed)
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
    all_metrics = []
    #special_logging.info("*** Before Training Evaluation ***")
    #trainer.model.eval()
    #train_metrics = do_evaluation(trainer, train_dataset, "train", special_logging)
    #all_metrics.append(train_metrics)
    #eval_metrics = do_evaluation(trainer, eval_dataset, "eval", special_logging)
    #all_metrics.append([train_metrics, eval_metrics])

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.model.train()
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    training_metrics = train_result.metrics
    training_metrics["train_samples"] = len(train_dataset)
    trainer.save_model()  # Saves the tokenizer too for easy upload
    trainer.log_metrics("train", training_metrics)
    trainer.save_metrics("train", training_metrics)
    trainer.save_state()

    trainer.model.eval()
    # Evaluation
    special_logging.info("*** Final Evaluation ***")
    logging.info("*** Final Evaluation ***")
    train_metrics = do_evaluation(trainer, train_dataset, "train", special_logging)
    all_metrics.append(train_metrics)
    eval_metrics = do_evaluation(trainer, eval_dataset, "eval", special_logging)
    all_metrics.append([train_metrics, eval_metrics])
    return training_metrics, all_metrics


def predict(data_args, trainer, dataset, special_logging):
    if dataset is None:
        return None
    trainer.model.eval()
    predictions = trainer.predict(dataset, metric_key_prefix="test")
    pred_df = pd.read_csv(data_args.test_file)
    metrics = None
    if hasattr(predictions, "metrics"):
        readable = get_metric_report_str(predictions.metrics)
        special_logging.info(f"test metrics: \n{readable}")
        metrics = predictions.metrics    
    preds = predictions.predictions
    if isinstance(preds, tuple): # assume this is classification idk
        assert len(preds) == 2
        preds = preds[0]
        exped = np.exp(preds)
        softmax = exped / np.sum(exped, axis=1, keepdims=True)
        if preds.shape[1] == 2: # then assume we are in binary classification
            preds = softmax[:, 1]
        else:
            preds = softmax # have not tested this. It might cause the assignment on line 388 to fail because array
    elif isinstance(preds, np.ndarray):
        n_expected = None
        if hasattr(data_args, "max_seq_length"):
            n_expected = data_args.max_seq_length
        elif hasattr(data_args, "val_max_output_length"):
            n_expected = data_args.val_max_output_length
        elif hasattr(data_args, "max_output_length"):
            n_expected = data_args.max_output_length
        else:
            raise ValueError(f"Something is deeply wrong one of the three of these should exist")
        if preds.shape[1] != 2:
            assert preds.shape[0] == len(pred_df) and preds.shape[1] == n_expected, f"Predictions shape {preds.shape} does not match expected shape {len(pred_df), n_expected}"
            preds = trainer.tokenizer.batch_decode(preds, skip_special_tokens=True)
        else:
            exped = np.exp(preds)
            softmax = exped / np.sum(exped, axis=1, keepdims=True)
            preds = softmax[:, 1]
    pred_df[data_args.prediction_column_name] = None
    for i, pred in enumerate(preds):
        pred_df.at[i, data_args.prediction_column_name] = pred
    pred_df.to_csv(data_args.prediction_file, index=False)
    return metrics