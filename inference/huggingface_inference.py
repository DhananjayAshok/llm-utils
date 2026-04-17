from utils import log_error, log_warn, log_info, log_dict
import click
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
                          AutoModelForImageTextToText, AutoProcessor,
                          GenerationConfig, set_seed)
import torch
import copy
from inference.inference_utils import discover_prefix_prompt, save_meta_file, require_gpu
from utils.vlm_utils import get_single_vlm_message_list, get_vlm_message_list
from tqdm import tqdm
import numpy as np
from PIL import Image
from skimage import io
import os


def get_model(parameters, quantization, model_kind):
    if parameters["modality"] == "vlm":
        return get_vlm(parameters, quantization, model_kind)
    model_name = parameters["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=parameters["padding_side"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    parameters["tokenizer"] = tokenizer
    parameters["pad_token_id"] = tokenizer.pad_token_id
    dtype = parameters["dtype"]
    if model_kind == "gen":
        load_class = AutoModelForCausalLM
    elif model_kind == "clf":
        load_class = AutoModelForSequenceClassification
    else:
        raise ValueError(f"Invalid model kind {model_kind}. Must be 'gen' or 'clf'.")
    if quantization == "none":
        model = load_class.from_pretrained(model_name, device_map="auto", dtype=dtype)
    else:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=quantization == "4b", load_in_8bit=quantization == "8b")
        model = load_class.from_pretrained(model_name, device_map="auto", dtype=dtype, quantization_config=quantization_config)
    return model.eval()

def handle_replace_stop_strings(data_df, parameters):
    tokenizer = parameters["tokenizer"]
    if parameters["modality"] == "vlm" and not hasattr(tokenizer, "eos_token"):
        eos_token = parameters["tokenizer"].tokenizer.eos_token
    else:
        eos_token = tokenizer.eos_token
    if eos_token is None:
        log_warn("Tokenizer does not have an eos token. Cannot replace stop strings.", parameters)
        return
    stop_strings = parameters["stop_strings"]
    for stop_string in stop_strings:
        replace_func = lambda x: x.replace(stop_string, " " + eos_token + " ")
        data_df[parameters["input_column"]] = data_df[parameters["input_column"]].apply(replace_func)
    return




def get_vlm(parameters, quantization, model_kind):
    """
    This function is a placeholder for loading Vision Language Models (VLMs).
    Currently, it only supports text generation models.
    """
    dtype = parameters["dtype"]
    model_name = parameters["model_name"]
    padding_side = parameters["padding_side"]
    quant_dict = {}
    if quantization != "none":
        from transformers import BitsAndBytesConfig
        quant_dict["quantization_config"] = BitsAndBytesConfig(load_in_4bit=quantization == "4b", load_in_8bit=quantization == "8b")
    else:
        pass
    processor = AutoProcessor.from_pretrained(model_name, padding_side=padding_side)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    parameters["tokenizer"] = processor
    parameters["pad_token_id"] = processor.tokenizer.pad_token_id
    if model_kind == "clf":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, dtype=dtype,
                                                                    device_map="auto", trust_remote_code=True, **quant_dict)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_name, dtype=dtype,
                                                                    device_map="auto", trust_remote_code=True, **quant_dict)
        return model.eval()



def get_inputs(data_df, start, end, model, parameters):
    if parameters["modality"] == "lm":
        inputs = data_df.loc[start:end, parameters["input_column"]].tolist()
        inputs = parameters["tokenizer"](inputs, padding=True, truncation=True, return_tensors="pt").to(model.device)
        return inputs
    elif parameters["modality"] == "vlm":
        messages = get_vlm_message_list(data_df.loc[start:end].reset_index(drop=True))
        inputs = parameters["tokenizer"].apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        ).to(model.device)
        return inputs
    else:
        raise ValueError(f"Bro what did you do.")


def log_discrepancies(generation_parameters, original_generation_config, parameters):
    discrepancies = {}
    keys = generation_parameters.keys()
    for key in keys:
        if key in ["max_new_tokens"]:
            continue
        if hasattr(original_generation_config, key):
            original_val = getattr(original_generation_config, key)
            new_val = generation_parameters[key]
            if original_val != new_val:
                discrepancies[key] = (original_val, new_val)
    if len(discrepancies) > 0:
        log_info("You have changed the following generation config parameters from their original values: (original_val, new_val)", parameters)
        log_dict(discrepancies, parameters=parameters)


@click.command()
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), default="gen")
@click.option("--quantization", type=click.Choice(["none", "8b", "4b"]), default="none", help="The bitsandbytes quantization method to use.")
@click.option("--padding_side", type=click.Choice(["left", "right"]), default="right", help="The padding side to use for the tokenizer.")
@click.option("--batch_size", type=int, default=1)
@click.option("--num_beams", type=int, default=None, help="The number of beams to use for beam search. Set to 1 for greedy decoding.")
@click.option("--num_beam_groups", type=int, default=None, help="The number of beam groups to use for group beam search. Set to 1 for standard beam search.")
@click.option("--diversity_penalty", type=float, default=0.2, help="The diversity penalty to use for group beam search. Set to 0.0 for no diversity penalty.")
@click.option("--cache_implementation", default="dynamic", type=click.Choice(["dynamic", "static", "offloaded", "offloaded_static", "quantized"]), help="The implementation to use for cache.")
@click.option("--cache_prefix", type=bool, default=False, help="If true, will search for a prefix prompt in the input column and precompute its KV cache.")
@click.option("--replace_stop_strings", type=bool, default=True, help="If set, will replace stop strings in the input text with the models eos token.")
@click.option("--track_output_perplexity", type=bool, default=False)
@click.option("--output_perplexity_column", type=str, default="output_perplexity")
@click.option("--track_input_perplexity", type=bool, default=False)
@click.option("--input_perplexity_column", type=str, default="input_perplexity")
@click.option("--debug", type=bool, default=True, help="If set, will print the first generated output for a sanity check")
@click.pass_obj
def hf_inference(parameters, model_kind, quantization, padding_side, batch_size, num_beams, num_beam_groups, diversity_penalty, cache_implementation, cache_prefix, replace_stop_strings, track_output_perplexity, output_perplexity_column, track_input_perplexity, input_perplexity_column, debug):
    require_gpu(parameters)
    if model_kind == "gen":
        if parameters["max_new_tokens"] is None:
            log_error("--max_new_tokens is required for Generative LM inference", parameters)
    torch.set_grad_enabled(False)
    set_seed(parameters["random_seed"])
    parameters["padding_side"] = padding_side
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    meta_vars = {
                 "quantization": quantization,
                 "cache_prefix": cache_prefix,
                 "cache_implementation": cache_implementation}
    model = get_model(parameters, quantization, model_kind)
    if replace_stop_strings:
        handle_replace_stop_strings(data_df, parameters)
    track_scores = track_input_perplexity or track_output_perplexity
    generation_parameter_keys = ["max_new_tokens", "temperature", "do_sample", "top_p", "top_k", "num_return_sequences"]
    generation_parameters  = {key: parameters[key] for key in generation_parameter_keys if key in parameters}
    if model_kind == "gen":
        generation_parameters["repetition_penalty"] = parameters["repetition_penalty"]
    meta_vars.update(generation_parameters)
    if parameters["modality"] == "vlm":
        if cache_prefix:
            log_warn("Prefix caching is not supported for VLMs. Deactivating ...", parameters)
            cache_prefix = False
    if parameters["num_return_sequences"] > 1 or batch_size > 1:
        if cache_prefix:
            log_warn("Prefix caching does not seem to work with num_return_sequences > 1 or batch_size > 1. Deactivating ...")
            cache_prefix = False
    if num_beams is not None:
        generation_parameters["num_beams"] = num_beams
        meta_vars["num_beams"] = num_beams
    if num_beam_groups is not None:
        if num_beam_groups > 1:
            if track_input_perplexity or track_output_perplexity:
                second_string = "You may encounter an error."
                if parameters["dtype"] != "float32":
                    second_string = ("It seems to only work with float32 dtype. "
                                     "Change the dtype and try again if you really want perplexity tracking, "
                                     "but that might fail too. For now, switching off tracking....")
                    track_scores = False
                log_warn(f"The output_score argument required for perplexity tracking is weird with grouped beam search." + second_string, parameters)
            generation_parameters["num_beam_groups"] = num_beam_groups
            generation_parameters["diversity_penalty"] = diversity_penalty
            meta_vars["num_beam_groups"] = num_beam_groups
            meta_vars["diversity_penalty"] = diversity_penalty
    try:
        original_generation_config = GenerationConfig.from_pretrained(parameters["model_name"])
        if hasattr(original_generation_config, "pad_token_id") and original_generation_config.pad_token_id is not None:
            generation_parameters['pad_token_id'] = original_generation_config.pad_token_id
        else:
            generation_parameters['pad_token_id'] = parameters["pad_token_id"]
        log_discrepancies(generation_parameters, original_generation_config, parameters)
    except Exception as e:
        log_warn(f"Could not load generation config from {parameters['model_name']}. Will fall back to default...",
                 parameters)
        generation_parameters["pad_token_id"] = parameters["pad_token_id"]
    start_idx = data_df[data_df[parameters["generation_complete_column"]] == False].index.min()
    checkpointed = start_idx != 0
    save_meta_file(meta_vars, output_filepath, parameters, consider_checkpoint=checkpointed)
    save_every = int(parameters["checkpoint_every"] * ((len(data_df) - start_idx) / batch_size))+1
    log_warn(f"Saving every {save_every} batches", parameters)
    prompt_cache = None
    prefix_text = discover_prefix_prompt(data_df, parameters["input_column"], parameters)
    if cache_prefix:
        if prefix_text is None:
            log_warn(f"Prefix caching is enabled but no prefix prompt could be discovered. "
                     f"Running inference without prefix caching...", parameters)
        else:
            prefix_inputs = parameters["tokenizer"]([prefix_text], padding=True, truncation=True, return_tensors="pt").to(model.device)
            prompt_cache = model(**prefix_inputs, cache_implementation=cache_implementation).past_key_values # had past_key_values=prompt_cache
            del prefix_inputs
            log_info(f"Prefix prompt discovered and KV cache precomputed.\nPrefix: {prefix_text}", parameters)

    for i in tqdm(range(start_idx, len(data_df), batch_size)):
        inputs = get_inputs(data_df, i, i + batch_size - 1, model, parameters)
        input_length = inputs["input_ids"].shape[1]
        n_items_in_batch = inputs['input_ids'].shape[0]
        if prompt_cache is not None:
            past_key_values = copy.deepcopy(prompt_cache)
            inputs["past_key_values"] = past_key_values
        else:
            inputs["cache_implementation"] = cache_implementation
        if model_kind == "gen":
            output = model.generate(**inputs, tokenizer=parameters["tokenizer"], output_scores=track_scores,
                                    return_dict_in_generate=True, trust_remote_code=True,
                                    **generation_parameters)
            output_sequences = output.sequences
            output_normed_perplexity = None
            input_normed_perplexity = None
            if track_output_perplexity:
                output_normed_perplexity = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True).mean().detach().cpu().numpy().item()
            if track_input_perplexity:
                raise NotImplementedError
            output_only = output_sequences[:, input_length:]
            out = parameters["tokenizer"].batch_decode(output_only, skip_special_tokens=True)
            for out_i in range(len(out)):
                for stop_string in parameters["stop_strings"]:
                    out[out_i] = out[out_i].replace(stop_string, "")
            out = np.array(out)
            out_reshaped = out.reshape(n_items_in_batch, parameters["num_return_sequences"]).tolist()
            for counter, j in enumerate(range(i, i+n_items_in_batch)):
                data_df.at[j, parameters["output_column"]] = out_reshaped[counter]
            if i == start_idx and debug:
                output_str = "\n[Output]: ".join(out_reshaped[0])
                input_str = data_df.loc[i, parameters["input_column"]]
                if prefix_text is not None:
                    input_str = "(Common prefix removed...) + " + input_str[len(prefix_text):]  # remove prefix from input
                log_info(f"First generated output for sanity check: \nInput: {input_str} \nOutput(s): {output_str}", parameters)
            if track_output_perplexity:
                data_df.loc[i, output_perplexity_column] = output_normed_perplexity
            if track_input_perplexity:
                data_df.loc[i, input_perplexity_column] = input_normed_perplexity
        elif model_kind == "clf":
            output = model(**inputs)
            out = output.logits.detach().cpu().float().numpy() # .argmax(dim=-1)
            for counter, j in enumerate(range(i, i+n_items_in_batch)):
                data_df.at[j, parameters["output_column"]] = out[counter].argmax().item()
                data_df.at[j, parameters["output_logits_column"]] = out[counter].tolist()
        data_df.loc[i:i+batch_size-1, parameters["generation_complete_column"]] = True
        if (i % save_every == 0 and i > 0) or i >= len(data_df) - batch_size:
            data_df.to_json(output_filepath, index=False, orient="records", lines=True)
        del inputs
        del output
    log_info(f"Saved output to {output_filepath}", parameters)
    return
