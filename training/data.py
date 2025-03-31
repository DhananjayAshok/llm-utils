from utils.log_handling import log_error

import os

import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset, disable_caching
from filelock import FileLock
from tqdm import tqdm
from trl.trainer import ConstantLengthDataset

disable_caching()


def validate_data(dataset, training_kind, pretrain_with_output, logger):
    mandatory_columns = ["input"]
    if training_kind in ["sft", "clf"] or training_kind == "pre" and pretrain_with_output:
        mandatory_columns.append("output")
    elif training_kind in ["dpo, ppo"]:
        mandatory_columns.append("chosen")
        mandatory_columns.append("rejected")
    for split in dataset:
        for column in mandatory_columns:
            if column not in dataset[split].column_names:
                log_error(logger, f"Column {column} not found in {split} split of the dataset with columns {dataset[split].column_names}")
    string_columns = ["input", "chosen", "rejected"]
    int_columns = []
    if training_kind == "clf":
        int_columns.append("output")
    else:
        string_columns.append("output")
    for split in dataset:
        for column in string_columns:
            if column in dataset[split].features:    
                if dataset[split].features[column].dtype != "string":
                    log_error(logger, f"Column {column} in {split} split is not a string")
        for column in int_columns:
            if column in dataset[split].features:
                if dataset[split].features[column].dtype not in ["int64", "int32", "int16", "int8", "int"]:
                    log_error(logger, f"Column {column} in {split} split is not an int")
    logger.debug("Data validation complete")


def shuffle_and_handle_data_sizes(script_args, dataset, data_seed):
    max_train_samples = script_args.max_train_samples
    max_valid_samples = script_args.max_valid_samples
    max_test_samples = script_args.max_test_samples

    dataset["train"] = dataset["train"].shuffle(seed=data_seed)
    dataset["validation"] = dataset["validation"].shuffle(seed=data_seed)
    dataset["test"] = dataset["test"].shuffle(seed=data_seed)

    max_length = script_args.max_length
    # TODO: Unsure if I should handle max_length here


    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    if max_valid_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(max_valid_samples))

    if max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(max_test_samples))

    return dataset



def load_data_splits(extension, script_args, parameters):
    logger = parameters["logger"]
    train_file = script_args.train_file
    validation_file = script_args.validation_file
    test_file = script_args.test_file
    train_split = script_args.train_split
    validation_split = script_args.validation_split
    random_seed = script_args.data_seed
    data_files = {"train": train_file}
    if validation_file is not None:
        data_files["validation"] = validation_file
    if test_file is not None:
        data_files["test"] = test_file
    if extension == "txt":
        dataset = datasets.load_dataset("text", data_files=data_files).rename_column("text", "input")
    else:
        dataset = load_dataset(extension, data_files=data_files)  # should be csv
    validate_data(dataset, script_args.training_kind, script_args.pretrain_with_output, logger)
    if validation_file is None:
        train_val = dataset["train"].train_test_split(test_size=train_split, seed=random_seed)
        dataset["train"] = train_val["train"]
        dataset["validation"] = train_val["test"]
    if test_file is None:
        val_test = dataset["validation"].train_test_split(test_size=validation_split, seed=random_seed)
        dataset["validation"] = val_test["train"]
        dataset["test"] = val_test["test"]
    dataset = shuffle_and_handle_data_sizes(script_args, dataset, random_seed)
    return dataset



def load_data(script_args, parameters):
    training_kind = script_args.training_kind
    train_file = script_args.train_file
    logger = parameters["logger"]
    train_file_extension = train_file.split(".")[-1]
    if training_kind != "pre" and train_file_extension != "csv":
        log_error(logger, f"Training file must be a csv file for {training_kind} training. Got {train_file_extension}")
    else:
        if train_file_extension != "txt":
            log_error(logger, f"Training file must be a txt file or a csv for {training_kind} training. Got {train_file_extension}")
    dataset = load_data_splits(train_file_extension, script_args, parameters)
    return dataset


def str_nested_dict(d, indent=0):
    s = ""
    for k, v in d.items():
        s += "  " * indent + str(k) + "\n"
        if isinstance(v, dict):
            s += str_nested_dict(v, indent + 1)
        else:
            s += "  " * (indent + 1) + str(v) + "\n"
    return s


def log_token_statistics(script_args, dataset, tokenizer, logger):
    """
    Compute token statistics for the dataset and print to log
    """
    columns_to_track = ["input"]
    if script_args.training_kind in ["sft"]:
        columns_to_track.append("output")
    elif script_args.training_kind in ["dpo", "ppo"]:
        columns_to_track.append("chosen")
        columns_to_track.append("rejected")
    column_statistics = {}
    for column in columns_to_track:
        column_statistics[column] = {"total_characters": 0, "total words": 0, "total_tokens": 0, "characters_per_token": 0}
    statistics = {}
    for split in dataset:
        statistics[split] = column_statistics.copy()
        for column in columns_to_track:
            for example in tqdm(dataset[split]):
                text = example[column]
                total_characters += len(text)
                total_words += len(text.split())
                if tokenizer.is_fast:
                    total_tokens += len(tokenizer(text).tokens())
                else:
                    total_tokens += len(tokenizer.tokenize(text))
            statistics[split][column]["total_characters"] = total_characters
            statistics[split][column]["total_words"] = total_words
            statistics[split][column]["total_tokens"] = total_tokens
            statistics[split][column]["characters_per_token"] = total_characters / total_tokens
    logger.debug(str_nested_dict(statistics))
    return statistics


def estimate_chars_token_ratio(dataset, tokenizer, text_preparation_fn, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = text_preparation_fn(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def get_pretraining_data(dataset, tokenizer, script_args):
    prepare_sample_text = lambda x: x["input"]
    if script_args.pretrain_with_output:
        prepare_sample_text = lambda x: f"Input: {x['input']}\nOutput: {x['output']}"

    chars_per_token = estimate_chars_token_ratio(
        dataset["train"],
        tokenizer,
        prepare_sample_text,
        nb_examples=script_args.max_train_samples,
    )

    train_dataset = ConstantLengthDataset(
        tokenizer,
        dataset["train"],
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        dataset["validation"],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )

    test_dataset = ConstantLengthDataset(
        tokenizer,
        dataset["validation"],
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=script_args.seq_length,
        chars_per_token=chars_per_token,
    )



def get_label_list(raw_dataset, split="train"):
    """Get the list of labels from a multi-label dataset"""

    label_list = raw_dataset[split].unique("output")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def infer_label_list(dataset, logger):
    label_list = get_label_list(dataset, split="train")
    for split in ["validation", "test"]:
        if split in dataset:
            val_or_test_labels = get_label_list(dataset, split=split)
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
        log_error(logger, "You need more than one label to do classification.")
    return label_list