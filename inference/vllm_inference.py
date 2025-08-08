from utils.log_handling import log_warn
from vllm import LLM, SamplingParams
from utils import log_info, log_warn, log_error
from inference.inference_utils import save_meta_file
import click
import torch


def quick_token_count(text):
    return len(text.split())

@click.command()
@click.option("--enable_prefix_caching", type=bool, default=True, help="Enable prefix caching for vLLM inference.")
@click.option("--max_model_len", type=int, default=1000, help="The maximum sequence length for the model. This is used to set the KV cache size.")
@click.pass_obj
def vllm_inference(parameters, enable_prefix_caching, max_model_len):
    if parameters["max_new_tokens"] is None:
        log_error("--max_new_tokens is required for vLLM inference", parameters)
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    quick_token_count_max = data_df[parameters["input_column"]].apply(quick_token_count).max()
    if quick_token_count_max > max_model_len:
        log_warn(f"Input text length exceeds max model length ({quick_token_count_max} > {max_model_len}). This run may fail, consider increasing --max_model_len value after vllm command.", parameters)
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
    llm = LLM(model=parameters["model_name"], tensor_parallel_size=n_gpus, enable_prefix_caching=enable_prefix_caching,
              max_model_len=max_model_len)
    if enable_prefix_caching:
        llm.generate(data_df[parameters["input_column"]].iloc[0], sampling_params) # warm up the cache
    outputs = llm.generate(data_df[parameters["input_column"]], sampling_params)
    output_texts = []
    for output in outputs:
        internal_outputs = []
        for out_text in output.outputs:
            internal_outputs.append(out_text.text)
        output_texts.append(internal_outputs)
    for i in range(len(output_texts)):
        data_df.at[i, parameters["output_column"]] = output_texts[i]
    data_df[parameters["generation_complete_column"]] = True
    data_df.to_json(output_filepath, index=False, lines=True, orient="records")
    log_info(f"Saved output to {output_filepath}", parameters)
    return