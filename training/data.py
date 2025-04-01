from utils.log_handling import log_error

import datasets
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from trl.trainer import ConstantLengthDataset

disable_caching()


def validate_data(dataset, training_kind, pretrain_with_output, logger):
    """
    Check that the dataset has the right columns and data types for the training kind
    """
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
    string_or_int_columns = []
    if training_kind == "clf":
        string_or_int_columns.append("output")
    else:
        string_columns.append("output")
    for split in dataset:
        for column in string_columns:
            if column in dataset[split].features:    
                if dataset[split].features[column].dtype != "string":
                    log_error(logger, f"Column {column} in {split} split is not a string, it is {dataset[split].features[column].dtype}")  
        for column in string_or_int_columns:
            if column in dataset[split].features:    
                if dataset[split].features[column].dtype not in ["string", "int32", "int64", "int"]:
                    log_error(logger, f"Column {column} in {split} split is not a string or int, it is {dataset[split].features[column].dtype}")

def shuffle_and_handle_data_sizes(script_args, dataset, data_seed):
    """
    Shuffle the dataset and cut it to the length specified in the script arguments
    """
    max_train_samples = script_args.max_train_samples
    max_valid_samples = script_args.max_valid_samples
    max_test_samples = script_args.max_test_samples

    dataset["train"] = dataset["train"].shuffle(seed=data_seed)
    dataset["validation"] = dataset["validation"].shuffle(seed=data_seed)
    dataset["test"] = dataset["test"].shuffle(seed=data_seed)

    if max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    if max_valid_samples is not None:
        dataset["validation"] = dataset["validation"].select(range(max_valid_samples))

    if max_test_samples is not None:
        dataset["test"] = dataset["test"].select(range(max_test_samples))

    return dataset



def load_data_splits(extension, script_args, parameters):
    """
    Check that the file extension is valid and return Dataset
    Args:
        extension: the file extension of the data file
        script_args: the parsed script arguments
        parameters: the parameters dictionary from configs

    Returns:
        dataset: HuggingFace Dataset object with train, validation and test splits and 
                    columns (input), (input, output) or (input, chosen, rejected) depending on the training kind
                    It has been shuffled and cut to the length (dataset length not token) specified in args
    """
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
    """
    Load the data from the file arguments and return the dataset

    Args:
        script_args: the parsed script arguments
        parameters: the parameters dictionary from configs

    Returns:
        dataset: HuggingFace Dataset object with train, validation and test splits and 
                 columns (input), (input, output) or (input, chosen, rejected) depending on the training kind

    """
    training_kind = script_args.training_kind
    train_file = script_args.train_file
    logger = parameters["logger"]
    train_file_extension = train_file.split(".")[-1]
    allowed_extensions = ["csv"]
    if training_kind == "pre":
        allowed_extensions.append("txt")
    if train_file_extension not in allowed_extensions:
        log_error(logger, f"Unsupported file extension {train_file_extension}, only {allowed_extensions} are supported for training kind {training_kind}.")
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


def get_label_list(raw_dataset, split="train"):
    """Get the list of labels from a multi-label dataset"""

    label_list = raw_dataset[split].unique("output")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def infer_label_list(dataset, logger):
    """
    Infer the label list from the dataset with special handling for differences between train and val/test labels
    """
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