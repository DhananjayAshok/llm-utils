import click
import os
import subprocess
from utils import log_info, log_warn, log_error
from time import time

def do_batch_size_run(data_df, tmp_path, batch_size, command, hf_command, parameters):
    sample_df = data_df.sample(n=batch_size, random_state=parameters["random_seed"]).reset_index(drop=True)
    sample_df.to_csv(tmp_path, index=False)
    final_command = command + ["hf"] + hf_command + [f"--batch_size", f"{batch_size}"]
    log_info(f"Running command: {' '.join(final_command)}", parameters)
    result = subprocess.run(final_command, capture_output=True, text=True)
    if result.returncode != 0:
        return False
    else:
        return True


@click.command()
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), default="gen")
@click.option("--quantization", type=click.Choice(["none", "8b", "4b"]), default="none", help="The bitsandbytes quantization method to use.")
@click.option("--padding_side", type=click.Choice(["left", "right"]), default="right", help="The padding side to use for the tokenizer.")
@click.option("--batch_size_low", type=int, default=1)
@click.option("--batch_size_high", type=int, default=50)
@click.option("--num_beams", type=int, default=None, help="The number of beams to use for beam search. Set to 1 for greedy decoding.")
@click.option("--cache_implementation", default="dynamic", type=click.Choice(["dynamic", "static", "offloaded", "offloaded_static", "quantized"]), help="The implementation to use for cache.")
@click.option("--cache_prefix", type=bool, default=False, help="If true, will search for a prefix prompt in the input column and precompute its KV cache.")
@click.option("--tmp_dir", default=None, help="The path to the temporary directory to use for the input_file csv. If not provided will use storage_dir/tmp")
@click.option("--avoid_conflicts", default=False, is_flag=True, help="If true, will use the timestamp of the current run to avoid overwriting existing files in the tmp_dir.")
@click.pass_obj
def infer_batch_size(parameters, **kwargs):
    batch_size_low = kwargs.pop("batch_size_low")
    batch_size_high = kwargs.pop("batch_size_high")
    tmp_path = kwargs.pop("tmp_dir")
    avoid_conflicts = kwargs.pop("avoid_conflicts")
    if tmp_path is None:
        tmp_path = os.path.join(parameters["storage_dir"], "tmp")  # should already exist
    tmp_file = "tmp_input.csv"
    if avoid_conflicts:
        tmp_file = tmp_file.replace(".csv", str(time())+".csv")
    tmp_file = os.path.join(tmp_path, tmp_file)
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    if len(data_df) < batch_size_high:
        log_warn(f"Provided batch_size_high value of {batch_size_high} is larger than the dataset size. "
                 f"Setting to {len(data_df)}...")
        batch_size_high = len(data_df)
    if batch_size_high <= batch_size_low+2:
        log_error(f"Got batch_size_low {batch_size_low} not far enough away from batch_size_high {batch_size_high}."
                  f"You must have a separation greater than 2 or theres no point in this script.")
    drop_cols = [parameters["generation_complete_column"], parameters["output_column"]]
    for drop_col in drop_cols:
        if drop_col in data_df.columns:
            data_df.drop(columns=[drop_col], inplace=True)
    relevant_arguments = ["model_name", "input_column", "output_column", "generation_complete_column",
                          "max_new_tokens", "dtype", "num_return_sequences"]
    command = ["python", "infer.py"]
    for arg in relevant_arguments:
        command.append(f"--{arg}")
        command.append(f"{parameters[arg]}")
    for stop_string in parameters["stop_strings"]:
        command.append("--stop_strings")
        command.append(stop_string)
    command.extend(["--input_file", tmp_file])
    hf_command = []
    for key, value in kwargs.items():
        if value is not None:
            hf_command.append(f"--{key}")
            hf_command.append(str(value))
    mid = (batch_size_high - batch_size_low) // 2
    current_high = batch_size_high
    current_low = batch_size_low
    tried_values = {}
    while current_low < mid < current_high:
        passes = do_batch_size_run(data_df, tmp_file, mid, command, hf_command, parameters)
        tried_values[mid] = passes
        if passes: # then search up
            current_low = mid
        else:
            current_high = mid
        mid = (current_high - current_low) // 2
    if mid == batch_size_low:
        low_pass = do_batch_size_run(data_df, tmp_file, mid, command, hf_command, parameters)
        if not low_pass:
            if batch_size_low == 1:
                log_warn(f"No batches fit on GPU. "
                         f"This suggests you cannot run under current configuations. "
                         f"Change num_return_sequences, quantization and cache_implementation etc.", parameters)
            else:
                log_warn(f"Even the lowest attempted batch size ({batch_size_low}) does not fit on GPU. "
                         f"Either there is a bug in the code, or you should try lower values.")
                return
    if mid == batch_size_high:
        high_pass = do_batch_size_run(data_df, tmp_file, mid, command, hf_command, parameters)
        if high_pass:
            log_warn(f"Even the highest attempted batch size ({batch_size_high}) fits on GPU. Try higher.")
            return
    log_info(f"Estimated Maximum Batch Size: {mid}. This is an overestimate, use a smaller one in your eventual run.")
    return

