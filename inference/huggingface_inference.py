from utils import log_error, log_warn, log_info, log_dict
import click
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification,
                          DynamicCache, StaticCache, OffloadedCache, OffloadedStaticCache,
                          QuantizedCache, QuantizedCacheConfig, GenerationConfig)
import torch
import copy
from inference.inference_utils import discover_prefix_prompt

from tqdm import tqdm
import numpy as np
import os


def get_cache(cache_implementation, model, batch_size, num_beams=1):
    if cache_implementation == "dynamic":
        return DynamicCache()
    elif cache_implementation == "static":
        return StaticCache(config=model.config, max_batch_size=batch_size, max_cache_len=1024, device=model.device, dtype=model.dtype)
    elif cache_implementation == "offloaded":
        return OffloadedCache()
    elif cache_implementation == "offloaded_static":
        return OffloadedStaticCache(config=model.config, max_batch_size=batch_size, max_cache_len=1024, device=model.device, dtype=model.dtype)
    elif cache_implementation == "quantized":
        config = QuantizedCacheConfig(compute_dtype=model.dtype,device=model.device)
        return QuantizedCache(config)
    else:
        raise ValueError(f"Invalid cache implementation {cache_implementation}. Must be one of 'dynamic', 'static', 'offloaded', 'offloaded_static', or 'quantized'.")


def get_model(parameters, quantization, model_kind):
    model_name = parameters["model_name"]
    dtype = parameters["dtype"]
    if model_kind == "gen":
        load_class = AutoModelForCausalLM
    elif model_kind == "clf":
        load_class = AutoModelForSequenceClassification
    else:
        raise ValueError(f"Invalid model kind {model_kind}. Must be 'gen' or 'clf'.")
    if quantization == "none":
        model = load_class.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)
    else:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=quantization == "4b", load_in_8bit=quantization == "8b")
        model = load_class.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, quantization_config=quantization_config)
    return model.eval()


def log_discrepancies(generation_config, original_generation_config, parameters):
    discrepancies = {}
    keys = dir(generation_config)
    keys = [key for key in keys if not key.startswith("_")]
    for key in keys:
        original_val = getattr(original_generation_config, key)
        new_val = getattr(generation_config, key)
        if original_val != new_val:
            discrepancies[key] = (original_val, new_val)
    if len(discrepancies) > 0:
        log_info("You have changed the following generation config parameters from their original values: (original_val, new_val)", parameters)
        log_dict(discrepancies, parameters)


@click.command()
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), default="gen")
@click.option("--quantization", type=click.Choice(["none", "8b", "4b"]), default="none", help="The bitsandbytes quantization method to use.")
@click.option("--padding_side", type=click.Choice(["left", "right"]), default="right", help="The padding side to use for the tokenizer.")
@click.option("--batch_size", type=int, default=1)
@click.option("--cache_implementation", default="dynamic", type=click.Choice(["dynamic", "static", "offloaded", "offloaded_static"]), help="The implementation to use for cache.")
@click.option("--cache_prefix", type=bool, default=True, help="If true, will search for a prefix prompt in the input column and precompute its KV cache.")
@click.option("--checkpoint_every", type=float, default=0.2)
@click.option("--track_output_perplexity", type=bool, default=False)
@click.option("--output_perplexity_column", type=str, default="output_perplexity")
@click.option("--track_input_perplexity", type=bool, default=False)
@click.option("--input_perplexity_column", type=str, default="input_perplexity")
@click.pass_obj
def hf_inference(parameters, quantization, padding_side, model_kind, batch_size, cache_implementation, cache_prefix, checkpoint_every, track_output_perplexity, output_perplexity_column, track_input_perplexity, input_perplexity_column):
    torch.set_grad_enabled(False)
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    tokenizer = AutoTokenizer.from_pretrained(parameters["model_name"], padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_model(parameters, quantization, model_kind)
    track_scores = track_input_perplexity or track_output_perplexity
    try:
        original_generation_config = GenerationConfig.from_pretrained(parameters["model_name"])
        generation_config, unused_args = GenerationConfig.from_pretrained(parameters["model_name"], **parameters,
                                                                          pad_token_id=tokenizer.eos_token_id,
                                                                          tokenizer=tokenizer,
                                                                          output_scores=track_scores,
                                                                          return_dict_in_generate=True,
                                                                          return_unused_kwargs=True)
        log_discrepancies(generation_config, original_generation_config, parameters)
    except Exception as e:
        log_warn(f"Could not load generation config from {parameters['model_name']}. Will fall back to default...",
                 parameters)
        generation_config = GenerationConfig(**parameters)

    start_idx = data_df[data_df[parameters["generation_complete_column"]] == False].index.min()
    save_every = int(checkpoint_every * ((len(data_df) - start_idx) / batch_size))+1
    log_warn(f"Saving every {save_every} batches", parameters)
    prompt_cache = None
    if cache_prefix:
        prefix_text = discover_prefix_prompt(data_df, parameters["input_column"], parameters)
        if prefix_text is None:
            log_warn(f"Prefix caching is enabled but no prefix prompt could be discovered. "
                     f"Running inference without prefix caching...", parameters)
        else:
            prompt_cache = get_cache(cache_implementation=cache_implementation, model=model, batch_size=batch_size)
            prefix_inputs = tokenizer([prefix_text], padding=True, truncation=True, return_tensors="pt").to(model.device)
            prompt_cache = model(**prefix_inputs, past_key_values=prompt_cache).past_key_values
            del prefix_inputs
            log_info(f"Prefix prompt discovered and KV cache precomputed.\nPrefix: {prefix_text}", parameters)

    for i in tqdm(range(start_idx, len(data_df), batch_size)):
        prompts = data_df.loc[i:i+batch_size-1, parameters["input_column"]].tolist()
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        if prompt_cache is not None:
            past_key_values = copy.deepcopy(prompt_cache)
            inputs["past_key_values"] = past_key_values
        else:
            inputs["cache_implementation"] = cache_implementation
        output = model.generate(**inputs, generation_config=generation_config)
        output_sequences = output.sequences
        output_normed_perplexity = None
        input_normed_perplexity = None
        if track_output_perplexity:
            output_normed_perplexity = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True).mean().detach().cpu().numpy().item()
        if track_input_perplexity:
            raise NotImplementedError
        output_only = output_sequences[:, input_length:]
        out = tokenizer.batch_decode(output_only, skip_special_tokens=True)
        for out_i in range(len(out)):
            for stop_string in parameters["stop_strings"]:
                out[out_i] = out[out_i].replace(stop_string, "")

        data_df.loc[i:i+batch_size-1, parameters["output_column"]] = out
        data_df.loc[i:i+batch_size-1, parameters["generation_complete_column"]] = True
        del inputs
        del output
        if track_output_perplexity:
            data_df.loc[i, output_perplexity_column] = output_normed_perplexity
        if track_input_perplexity:
            data_df.loc[i, input_perplexity_column] = input_normed_perplexity

        if (i % save_every == 0 and i > 0) or i >= len(data_df) - batch_size:
            data_df.to_json(output_filepath, index=False, orient="records", lines=True)
    return 
