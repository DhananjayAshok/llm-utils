import os
import pandas as pd
from utils import log_warn, log_error, log_info, file_makedir


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
        if generation_complete_column in df.columns:
            log_error(f"Input file already has a column named '{generation_complete_column}'. This is used to track inference completion, reset it with --generation_complete_column or rename the column in your df", parameters)
        return df
    return None

def get_output_file_path(output_file, input_file, input_df, output_column, parameters):
    if output_file == input_file:
        log_error("Output file cannot be the same as input file", parameters)
    if output_file is None:
        # replace the extension of input_file with '_output' before the extension
        output_file = input_file.rsplit('.', 1)[0] + "_output." + input_file.rsplit('.', 1)[-1]
        if output_column in input_df.columns:
            log_error(f"Output file already has a column named '{output_column}'. This is used to store the model's output, reset it with --output_column or rename the column in your df", parameters)
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
        return input_df, output_file_path
    else:
        output_df = load_file(output_file_path, output_column, parameters)
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
