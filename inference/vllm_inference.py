from utils.log_handling import log_warn
from vllm import LLM, SamplingParams
from utils import log_info, log_warn, log_error
from inference.inference_utils import save_meta_file, require_gpu
from transformers import AutoTokenizer
import click
import torch
from tqdm import tqdm

def infer_max_input_length(model_name, data_df, start_idx, parameters):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_lens = data_df.loc[start_idx:, parameters["input_column"]].apply(lambda x: len(tokenizer(x, padding=False, truncation=False)["input_ids"]))
    max_input_length = tokenized_lens.max()
    return max_input_length


@click.command()
@click.option("--enable_prefix_caching", type=bool, default=True, help="Enable prefix caching for vLLM inference.")
@click.option("--max_model_len", type=int, default=None, help="The maximum sequence length for the model. If not set, will automatically infer it.")
@click.option("--vllm_max_n", type=int, default=5, help="I don't know why, but when we ask vLLM to do inference on too many points at once it produces garbage output, and gives no warnings. This parameter must be tweaked manually, unforunately.")
@click.pass_obj
def vllm_inference(parameters, enable_prefix_caching, max_model_len, vllm_max_n):
    require_gpu(parameters)
    if parameters["max_new_tokens"] is None:
        log_error("--max_new_tokens is required for vLLM inference", parameters)
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    meta_vars = {}
    temperature = 0.0
    if parameters.get("temperature", None) is not None:
        temperature = parameters["temperature"]
        meta_vars["temperature"] = temperature
    top_p = 1.0
    if parameters.get("top_p", None) is not None:
        top_p = parameters["top_p"]
        meta_vars["top_p"] = top_p
    top_k = -1
    if parameters.get("top_k", None) is not None:
        top_k = parameters["top_k"]
        meta_vars["top_k"] = top_k
    n = 1
    if parameters["num_return_sequences"] is not None:
        n = parameters["num_return_sequences"]
    repetition_penalty = parameters["repetition_penalty"]
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=parameters["max_new_tokens"],
                                     stop=parameters["stop_strings"], n=n, top_k=top_k, repetition_penalty=repetition_penalty)
    n_gpus = torch.cuda.device_count()
    save_meta_file(meta_vars, output_filepath, parameters)
    start_idx = data_df[data_df[parameters["generation_complete_column"]] == False].index.min()
    max_input_length = infer_max_input_length(parameters["model_name"], data_df, start_idx, parameters)
    if max_model_len is None:
        max_model_len = max_input_length + parameters["max_new_tokens"] + 100 # give some buffer for generation
        log_info(f"Inferred max_input_length of {max_input_length} from data. Setting max_model_len to {max_model_len}", parameters)
    elif max_model_len < max_input_length + parameters["max_new_tokens"]:
        log_warn(f"Warning: provided max_model_len of {max_model_len} is less than the max input length + max_new_tokens ({max_input_length + parameters['max_new_tokens']}). This may cause errors.", parameters)
    llm = LLM(model=parameters["model_name"], tensor_parallel_size=n_gpus, enable_prefix_caching=enable_prefix_caching, max_num_seqs=vllm_max_n+5,
              max_model_len=max_model_len)
    if enable_prefix_caching:
        llm.generate(data_df[parameters["input_column"]].iloc[0], sampling_params) # warm up the cache
    if start_idx != 0:
        log_info(f"Resuming from index {start_idx}", parameters)
    batch_size = vllm_max_n
    save_every = int(parameters["checkpoint_every"] * ((len(data_df) - start_idx) / batch_size))+1
    for i in tqdm(range(start_idx, len(data_df), batch_size), desc="Performing vLLM inference"):
        input_texts = data_df[parameters["input_column"]].loc[i:i+batch_size].tolist()
        outputs = llm.generate(input_texts, sampling_params, use_tqdm=False)
        data_df.loc[i:i+batch_size-1, parameters["generation_complete_column"]] = True
        append_i = 0
        for output in outputs:
            internal_outputs = []
            for out_text in output.outputs:
                internal_outputs.append(out_text.text)
            data_df.at[i+append_i, parameters["output_column"]] = internal_outputs
            append_i += 1
        if (i % save_every == 0 and i > 0) or i >= len(data_df) - batch_size:
            data_df.to_json(output_filepath, index=False, orient="records", lines=True)
    log_info(f"Saved output to {output_filepath}", parameters)
    return