from utils import log_error, log_warn, log_info

import datasets
from datasets import load_dataset, disable_caching
from tqdm import tqdm
from PIL import Image

disable_caching()

def load_image(image_path):
    """
    Load an image from a file path or URL.
    """
    image = Image.open(image_path)
    return image.convert('RGB')

def validate_data(dataset, training_kind, pretrain_with_output, parameters):
    """
    Check that the dataset has the right columns and data types for the training kind
    """
    mandatory_columns = ["input"]
    if training_kind in ["sft", "clf", "ga", "npo"] or training_kind == "pre" and pretrain_with_output:
        mandatory_columns.append("output")
    elif training_kind in ["dpo"]:
        mandatory_columns.append("chosen")
    if training_kind in ["dpo"]:
        mandatory_columns.append("rejected")
    if training_kind in ["ga"]:
        mandatory_columns.append("forget")
    for split in dataset:
        for column in mandatory_columns:
            if column not in dataset[split].column_names:
                err_string = f"Column {column} not found in {split} split of the dataset with columns {dataset[split].column_names}. Pass --input_column,  --output_column, --chosen_column or --rejected_column to rename the columns in the dataset."
                if training_kind == "pre" and column == "output":
                    err_string = err_string + " For pretraining with argument pretrain_with_output=True. Either set pretrain_with_output=False or provide an output column."
                log_error(err_string, parameters)

    dataset = handle_nans(dataset, mandatory_columns, parameters)
    return dataset # Right now the dtype doesn't seem to update after dropping nans, so just trust the user. 
    string_columns = ["input", "chosen", "rejected"]
    string_or_int_or_bool_columns = []
    if training_kind == "clf":
        string_or_int_or_bool_columns.append("output")
    else:
        string_columns.append("output")
    for split in dataset:
        for column in string_columns:
            if column in dataset[split].features:    
                if dataset[split].features[column].dtype != "string":
                    log_error(logger, f"Column {column} in {split} split is not a string, it is {dataset[split].features[column].dtype}")  
        for column in string_or_int_or_bool_columns:
            if column in dataset[split].features:    
                if dataset[split].features[column].dtype not in ["string", "int32", "int64", "int", bool, "bool", int]:
                    log_error(logger, f"Column {column} in {split} split is not a string, bool or int, it is {dataset[split].features[column].dtype}")
    return dataset


def handle_nans(dataset, check_cols, parameters):
    lengths = {}
    for split in dataset:
        lengths[split] = len(dataset[split])
    def filter(x):
        for col in check_cols:
            if x[col] is None or x[col] == "":
                return False
        return True
    dataset = dataset.filter(filter)
    for split in dataset:
        newlength = len(dataset[split])
        if newlength != lengths[split]:
            log_warn(f"Removed {lengths[split] - newlength}/{lengths[split]} rows with NaN values from {split} split", parameters)
    return dataset


def shuffle_and_handle_data_sizes(script_args, dataset, data_seed):
    """
    Shuffle the dataset and cut it to the length specified in the script arguments
    """
    max_train_samples = script_args.max_train_samples
    max_valid_samples = script_args.max_valid_samples
    max_test_samples = script_args.max_test_samples

    dataset["train"] = dataset["train"].shuffle(seed=data_seed)
    if "validation" in dataset:
        dataset["validation"] = dataset["validation"].shuffle(seed=data_seed)
    if "test" in dataset:
        dataset["test"] = dataset["test"].shuffle(seed=data_seed)

    if max_train_samples is not None and max_train_samples < len(dataset["train"]):
        dataset["train"] = dataset["train"].select(range(max_train_samples))

    if max_valid_samples is not None and "validation" in dataset:
        if max_valid_samples < len(dataset["validation"]):
            dataset["validation"] = dataset["validation"].select(range(max_valid_samples))

    if max_test_samples is not None and "test" in dataset:
        if max_test_samples < len(dataset["test"]):
            dataset["test"] = dataset["test"].select(range(max_test_samples))

    return dataset

def drop_column_if_needed(dataset, column_name):
    for split in dataset:
        if column_name in dataset[split].features:
            dataset = dataset.remove_columns(column_name)
            return dataset
    return dataset

def load_data_splits(extension, script_args):
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
    parameters = script_args.parameters
    train_file = script_args.train_file
    validation_file = script_args.validation_file
    test_file = script_args.test_file
    train_split = script_args.train_validation_split
    validation_split = script_args.validation_test_split
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
    if script_args.input_column != "input":
        dataset = drop_column_if_needed(dataset, "input")
        dataset = dataset.rename_column(script_args.input_column, "input")
    if script_args.output_column != "output":
        dataset = drop_column_if_needed(dataset, "output")
        dataset = dataset.rename_column(script_args.output_column, "output")
    if script_args.chosen_column is not None and script_args.chosen_column != "chosen":
        dataset = drop_column_if_needed(dataset, "chosen")
        dataset = dataset.rename_column(script_args.chosen_column, "chosen")
    if script_args.rejected_column is not None and script_args.rejected_column != "rejected":
        dataset = drop_column_if_needed(dataset, "rejected")
        dataset = dataset.rename_column(script_args.rejected_column, "rejected")
    if script_args.training_kind == "pre" and not script_args.pretrain_with_output:
        dataset = drop_column_if_needed(dataset, "output")
    if script_args.modality == "vlm" and script_args.image_input_column != "image":
        dataset = drop_column_if_needed(dataset, "image")
        dataset = dataset.rename_column(script_args.image_input_column, "image")
    dataset = validate_data(dataset, script_args.training_kind, script_args.pretrain_with_output, parameters)
    # load the image data from the urls in the image column if modality is vlm
    if script_args.modality == "vlm":
        dataset = dataset.map(
            lambda x: {"image": load_image(x["image"])}, # for some reason setting num_proc kills this. 
            desc="Loading images",
        )
    if validation_file is None and train_split is not None:
        if 0 < train_split < 1:
            train_val = dataset["train"].train_test_split(test_size=1-train_split, seed=random_seed)
            dataset["train"] = train_val["train"]
            dataset["validation"] = train_val["test"]
        else:
            log_error("train_validation_split cannot be outside 0 and 1, please provide a valid split.", parameters)
    if test_file is None and validation_split is not None:
        if 0 < validation_split < 1:
            val_test = dataset["validation"].train_test_split(test_size=1-validation_split, seed=random_seed)
            dataset["validation"] = val_test["train"]
            dataset["test"] = val_test["test"]
        else:
            if validation_split == 0: # remove validation split and add test split
                dataset["test"] = dataset.pop("validation")
    dataset = shuffle_and_handle_data_sizes(script_args, dataset, random_seed)
    return dataset


def load_data(script_args):
    """
    Load the data from the file arguments and return the dataset

    Args:
        script_args: the parsed script arguments
        parameters: the parameters dictionary from configs

    Returns:
        dataset: HuggingFace Dataset object with train, validation and test splits and 
                 columns (input), (input, output) or (input, chosen, rejected) depending on the training kind

    """
    parameters = script_args.parameters
    training_kind = script_args.training_kind
    train_file = script_args.train_file
    train_file_extension = train_file.split(".")[-1]
    allowed_extensions = ["csv"]
    if training_kind == "pre":
        allowed_extensions.append("txt")
    if train_file_extension not in allowed_extensions:
        log_error(f"Unsupported file extension {train_file_extension}, only {allowed_extensions} are supported for training kind {training_kind}.", parameters)
    dataset = load_data_splits(train_file_extension, script_args)
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


def log_token_statistics(script_args, dataset, tokenizer, parameters):
    """
    Compute token statistics for the dataset and print to log
    """
    columns_to_track = ["input"]
    if script_args.training_kind in ["sft", "npo", "ga"] or (script_args.training_kind == "pre" and script_args.pretrain_with_output):
        columns_to_track.append("output")
    elif script_args.training_kind in ["dpo"]:
        columns_to_track.append("chosen")
    if script_args.training_kind in ["dpo"]:
        columns_to_track.append("rejected")
    column_statistics = {}
    for column in columns_to_track:
        column_statistics[column] = {"total_characters": 0, "total words": 0, "total_tokens": 0, "characters_per_token": 0}
    statistics = {}
    for split in dataset:
        statistics[split] = column_statistics.copy()
        for column in columns_to_track:
            total_characters = 0
            total_words = 0
            total_tokens = 0
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
    parameters["logger"].debug(str_nested_dict(statistics))
    return statistics


def get_label_list(raw_dataset, split="train"):
    """Get the list of labels from a multi-label dataset"""

    label_list = raw_dataset[split].unique("output")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


def infer_label_list(dataset, parameters):
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
                log_warn(
                    f"Labels {diff} in {split} set but not in training set, adding them to the label list",
                    parameters
                )
                label_list += list(diff)
    # if label is -1, we throw a warning and remove it from the label list
    for label in label_list:
        if label == -1:
            log_warn("Label -1 found in label list, removing it.", parameters)
            label_list.remove(label)

    label_list.sort()
    num_labels = len(label_list)
    if num_labels <= 1:
        log_error("You need more than one label to do classification.", parameters)
    return label_list