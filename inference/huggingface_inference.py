from utils import log_error, log_info
import click
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def setup_hidden_directory(save_hidden, output_filepath, output_hidden_dir):
    if save_hidden:
        if output_hidden_dir is None:
            output_hidden_dir = os.path.join(os.path.dirname(output_filepath), "hidden_states")
            os.makedirs(output_hidden_dir, exist_ok=True)
    return


def get_model(parameters, model_kind):
    model_name = parameters["model_name"]
    dtype = parameters["dtype"]
    quantization = parameters["quantization"]
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
    return model



def get_hidden_states_lists(config, layers_to_track, parameters):
    hidden_states_list = {}
    track_layers = []
    if layers_to_track == "mid+":
        midplus = int(0.65 * config.num_hidden_layers)
        track_layers.append(midplus)
    elif layers_to_track == "many":
        track_layers = list(range(1, config.num_hidden_layers-1, 4))
    elif layers_to_track == "all":
        track_layers = list(range(1, config.num_hidden_layers-1))
    elif layers_to_track.isdigit():
        track_layers = [int(layers_to_track)]
    else:
        log_error(parameters["logger"], f"Invalid layers_to_track {layers_to_track}")
    for layer in track_layers:
        hidden_states_list[layer] = []
    return hidden_states_list, track_layers





@click.command()
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), default="gen")
@click.option("--quantization", type=click.Choice(["none", "8b", "4b"]), default="none", help="The bitsandbytes quantization method to use.")
@click.option("--batch_size", type=int, default=1)
@click.option("--checkpoint_every", type=float, default=0.2)
@click.option("--track_output_perplexity", type=bool, default=False)
@click.option("--output_perplexity_column", type=str, default="output_perplexity")
@click.option("--track_input_perplexity", type=bool, default=False)
@click.option("--input_perplexity_column", type=str, default="input_perplexity")
@click.option("--save_hidden", type=bool, default=False)
@click.option("--output_hidden_dir", type=str, default=None)
@click.option("--track_layers", type=str, default="mid+", help="How to track MLP layer outputs. 'mid+' means the 65th percentile layer, 'many' means every one ever four layers, 'all' means all layers and a specific number means that layer.")
@click.option('--track_token', type=click.Choice(["input", "output"], case_sensitive=False), default="input", help="Whether to track hidden state embeddings of the last input or output token.")
@click.pass_obj
def hf_inference(parameters, model_kind, batch_size, checkpoint_every, track_output_perplexity, output_perplexity_column, track_input_perplexity, input_perplexity_column, save_hidden, output_hidden_dir, track_layers, track_token):
    data_df, output_filepath = parameters["output_df"], parameters["output_filepath"]
    setup_hidden_directory(save_hidden, output_filepath, output_hidden_dir)
    tokenizer = AutoTokenizer.from_pretrained(parameters["model_name"])
    tokenizer.pad_token = tokenizer.eos_token
    model = get_model(parameters, model_kind)
    config = model.config

    all_layers_to_track = []
    hidden_states_lists = {}
    if save_hidden:
        if batch_size > 1:
            log_error(parameters["logger"], f"Batch size {batch_size} is greater than 1. This is not supported for hidden state tracking.")
        hidden_states_lists, all_layers_to_track = get_hidden_states_lists(config, track_layers, parameters)

    start_idx = data_df[data_df[parameters["generation_complete_column"]] == False].index.min()

    save_every = int(checkpoint_every * ((len(data_df) - start_idx) / batch_size))+1
    log_info(f"Saving every {save_every} batches", parameters)
    for i in tqdm(range(start_idx, len(data_df), batch_size)):
        prompts = data_df.loc[i:i+batch_size-1, parameters["input_column"]].tolist()
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1] # TODO: Check this works for batch size > 1
        output = model.generate(**inputs, max_new_tokens=parameters["max_new_tokens"], stop_strings=parameters["stop_strings"], pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_attentions=False, output_hidden_states=save_hidden, output_scores=track_input_perplexity or track_output_perplexity, return_dict_in_generate=True)
        output_sequences = output.sequences
        if save_hidden:
            for layer in all_layers_to_track:
                hidden_states = output.hidden_states[0][layer][0, -1].detach().cpu().numpy()  # TODO: Check that this works for classification too
                hidden_states_lists[layer].append(hidden_states)
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
        if track_output_perplexity:
            data_df.loc[i, output_perplexity_column] = output_normed_perplexity
        if track_input_perplexity:
            data_df.loc[i, input_perplexity_column] = input_normed_perplexity
        data_df.loc[i, "inference_completed"] = True

        if (i % save_every == 0 and i > 0) or i == len(data_df) - 1:
            data_df.to_csv(output_filepath, index=False)
            if save_hidden:
                for layer in all_layers_to_track:
                    array = np.array(hidden_states_lists[layer])
                    np.save(os.path.join(output_hidden_dir, f"hidden_states_{layer}.npy"), array)
    return 
