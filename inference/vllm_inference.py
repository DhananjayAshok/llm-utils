from vllm import LLM, SamplingParams
from utils import log_info
from inference.inference_utils import save_meta_file
import click
import torch


@click.command()
@click.option("--enable_prefix_caching", type=bool, default=True, help="Enable prefix caching for vLLM inference.")
@click.pass_obj
def vllm_inference(parameters, enable_prefix_caching):
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    meta_vars = {}
    temperature = 1.0
    if parameters["temperature"] is not None:
        temperature = parameters["temperature"]
        meta_vars["temperature"] = temperature
    top_p = 1.0
    if parameters["top_p"] is not None:
        top_p = parameters["top_p"]
        meta_vars["top_p"] = top_p
    top_k = -1
    if parameters["top_k"] is not None:
        top_k = parameters["top_k"]
        meta_vars["top_k"] = top_k
    n = 1
    if parameters["num_return_sequences"] is not None:
        n = parameters["num_return_sequences"]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=parameters["max_new_tokens"],
                                     stop=parameters["stop_strings"], n=n, top_k=top_k)
    n_gpus = torch.cuda.device_count()
    save_meta_file(meta_vars, output_filepath, parameters)
    llm = LLM(model=parameters["model_name"], tensor_parallel_size=n_gpus, enable_prefix_caching=enable_prefix_caching)
    if enable_prefix_caching:
        llm.generate(data_df[parameters["input_column"]].iloc[0], sampling_params) # warm up the cache
    outputs = llm.generate(data_df[parameters["input_column"]], sampling_params)
    data_df[parameters["output_column"]] = outputs
    data_df[parameters["generation_complete_column"]] = True
    data_df.to_json(output_filepath, index=False, lines=True, orient="records")
    log_info(f"Saved output to {output_filepath}", parameters)
    return