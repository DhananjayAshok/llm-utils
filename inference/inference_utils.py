import os
import pandas as pd
import yaml
from utils import log_warn, log_error, log_info, file_makedir
from datetime import datetime, timezone
import torch


def require_gpu(parameters):
    if not torch.cuda.is_available():
        log_error("No GPU available. This script requires a GPU to run.", parameters)
    else:
        # Print Device Count
        num_devices = torch.cuda.device_count()
        log_info(f"Number of available GPU devices: {num_devices}", parameters)


def load_file(file_path, extension, parameters):
    """
    Load a file based on its extension.
    Supported extensions: .csv, .tsv, .jsonl, .parquet
    """
    if not os.path.exists(file_path):
        log_error(f"File {file_path} does not exist", parameters)

    if extension == ".csv":
        return pd.read_csv(file_path)
    elif extension == ".tsv":
        return pd.read_csv(file_path, sep="\t")
    elif extension == ".jsonl":
        return pd.read_json(file_path, lines=True)
    elif extension == ".parquet":
        return pd.read_parquet(file_path)
    else: # should be unreachable
        log_error(f"Should be unreachable, but got an unsupported file extension {extension}", parameters)
        return None

def get_input_file(input_file, input_column, generation_complete_column, parameters):
    if input_file is None:
        log_error("No input file provided", parameters)
    if not os.path.exists(input_file):
        log_error(f"Input file {input_file} does not exist", parameters)
    csv_extensions = [".csv", ".tsv"]
    json_extensions = [".jsonl"]
    parquet_extensions = [".parquet"]
    allowed_extensions = csv_extensions + json_extensions + parquet_extensions
    flag = False
    df = None
    for allowed_extension in allowed_extensions:
        if input_file.endswith(allowed_extension):
            try:
                df = load_file(input_file, allowed_extension, parameters)
                flag = True
                break
            except Exception as e:
                log_error(f"Failed to load input file {input_file} with extension {allowed_extension}: {e}", parameters)
    if not flag:
        log_error(f"Input file {input_file} must be one of {allowed_extensions}", parameters)
    else:
        if input_column not in df.columns:
            log_error(f"Input file must have a column named '{input_column}'. Available columns: {df.columns.tolist()}", parameters)
        if parameters["modality"] == "vlm":
            if parameters["image_input_column"] not in df.columns:
                log_error(f"Input file must have a column named '{parameters['image_input_column']}' for image input. Available columns: {df.columns.tolist()}", parameters)
        if generation_complete_column in df.columns:
            log_error(f"Input file already has a column named '{generation_complete_column}'. This is used to track inference completion, reset it with --generation_complete_column or rename the column in your df", parameters)
        if parameters["output_logits_column"] in df.columns:
            log_error(f"Input file already has a column named '{parameters['output_logits_column']}'. This is used to store the model's output logits/probabilities, reset it with --output_logits_column or rename the column in your df", parameters)
        if parameters["output_column"] in df.columns:
            log_error(f"Input file already has a column named '{parameters['output_column']}'. This will be overwritten with the model's output.", parameters)
        if len(df) == 0:
            log_error(f"Input file {input_file} is empty.", parameters)
        return df
    return None

def get_output_file_path(output_file, input_file, input_df, output_column, parameters):
    # model_name_in_output_file
    if output_file is None:
        # replace the extension of input_file with '_output' before the extension
        start = input_file.rsplit('.', 1)[0]
        end = "_output.jsonl"
        if parameters["model_name_in_output_file"]:
            model_name_end = parameters["model_name"].split("/")[-1]
            end = f"_{model_name_end}_output.jsonl"
        output_file = start + end

    for possible_extension in ["csv", "tsv", "json", "txt", "parquet"]:
        if output_file.endswith(possible_extension):
            output_file = output_file.rsplit('.', 1)[0] + "." + "jsonl"
            break

    if not output_file.endswith(".jsonl"):
        log_error(f"Output file must be a JSON lines file (ending with .jsonl). Got {output_file}", parameters)

    file_makedir(output_file)
    return output_file

def handle_files(input_file, output_file, input_column, generation_complete_column, output_column, ignore_checkpoint,
                 parameters):
    """
    Handles the input and output files.
    Returns
        The output Dataframe (loading in the checkpoint if appropriate)
        output file path we should save it to
    """
    input_df = get_input_file(input_file, input_column, generation_complete_column, parameters)
    output_file_path = get_output_file_path(output_file, input_file, input_df, output_column, parameters)
    if ignore_checkpoint or not os.path.exists(output_file_path):
        input_df[output_column] = None
        input_df[generation_complete_column] = False
        input_df[parameters["output_logits_column"]] = None
        return input_df, output_file_path
    else:
        output_df = load_file(output_file_path, "."+output_file_path.rsplit(".", 1)[-1], parameters)
        for required_column in [input_column, generation_complete_column, output_column]:
            if required_column not in output_df.columns:
                log_error(f"Output file checkpoint at {output_file_path} should have a column named '{required_column}'. Available columns: {output_df.columns.tolist()}", parameters)
        if output_df[generation_complete_column].all():
            log_info(f"All rows in the output file {output_file_path} are already completed. No need to run inference again. Add the --ignore_checkpoint flag to regenerate", parameters)
            exit()
        else:
            start_idx = output_df[output_df[generation_complete_column] == False].index[0]
            log_info(f"Checkpoint detected. Starting inference from index {start_idx}/{len(output_df)}...", parameters)
            return output_df, output_file_path


def discover_prefix_prompt(input_df, input_column, parameters, n_samples=10):
    """
    Discover a prefix prompt from the input column of the input DataFrame.
    """
    if len(input_df) == 0:
        log_error("Input DataFrame is empty. No prefix prompt to discover.", parameters)
        return None
    n_samples = min(n_samples, len(input_df))

    input_texts = input_df[input_column].sample(n_samples, random_state=parameters["random_seed"])
    indexes = input_texts.index.tolist()
    input_texts = input_texts.tolist()
    all_equal = all(text == input_texts[0] for text in input_texts)
    if all_equal:
        log_warn("All input texts are identical. No prefix prompt to discover. This could be okay for VLMs", parameters)
        return ""
    prefix_end_index = -1
    discontinuity_found = False
    limiting_index = None
    while not discontinuity_found:
        next_chars = []
        for i, text in enumerate(input_texts):
            if len(text) <= prefix_end_index + 1:
                limiting_index = indexes[i]
                break
            next_chars.append(text[prefix_end_index + 1])
        discontinuity_found = len(set(next_chars)) > 1
        if not discontinuity_found:
            prefix_end_index += 1

    if limiting_index is not None:
        message =  f"""
        Sampled {n_samples} random input texts and found an input at index {limiting_index} that seems to be a subset of the others.
        This suggests a bug in the creation of the input file.
        Please check the input file and ensure that all inputs are unique.
        \n Input: {input_df.loc[limiting_index, input_column]}
        """
        log_error(message, parameters)
    
    if prefix_end_index == -1:
        return ""

    return input_texts[0][:prefix_end_index] # TODO: Check if this should have a +1

def save_meta_file(meta_vars, output_filepath, parameters, consider_checkpoint=False):
    """
    Save meta information to a file.
    """
    meta_filepath = output_filepath.replace(".jsonl", ".meta.yaml")
    utc_datetime = datetime.now(timezone.utc)
    update_dict = {
        "model_name": parameters["model_name"],
        "num_return_sequences": parameters["num_return_sequences"],
    }
    meta_vars.update(update_dict)
    if consider_checkpoint and os.path.exists(meta_filepath):
        with open(meta_filepath, "r") as f:
            existing_meta = yaml.load(f, Loader=yaml.FullLoader)
        conflicts = []
        for key, value in meta_vars.items():
            if key not in existing_meta or existing_meta[key] != value:
                conflicts.append(f"{key}: {existing_meta.get(key, None)} -> {value}")
        if len(conflicts) > 0:
            conflict_str = ", ".join(conflicts)
            log_error(f"There was already a meta file for this output file at {meta_filepath}. "
                      f"This has conflicting parameters (old -> new): {conflict_str}."
                      f"\nThis will lead in the weird case where outputs are not consistently generated. "
                      f"Sort this out bro idk", parameters)

    meta_vars["timestamp_utc"] = utc_datetime.strftime("%Y-%m-%d %H:%M:%S %Z")
    with open(meta_filepath, "w") as f:
        yaml.dump(meta_vars, f)
    log_info(f"Wrote meta file to {meta_filepath}", parameters)
    return meta_filepath