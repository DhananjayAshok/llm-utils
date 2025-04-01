from utils.log_handling import log_error
import click
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def setup_directories(save_hidden, output_csv_path, output_hidden_dir, parameters):
    makedirs = []
    if save_hidden:
        if output_hidden_dir is None:
            output_hidden_dir = os.path.join(os.path.dirname(output_csv_path), "hidden_states")
            if os.path.exists(output_hidden_dir):
                log_error(parameters["logger"], f"Output hidden directory {output_hidden_dir} already exists.")
        
        makedirs.append(output_hidden_dir)
    for directory in makedirs:
        if not os.path.exists(directory) and not os.path.dirname(directory) == "":
            os.makedirs(directory)
    return output_hidden_dir


def get_data(data_path, input_column, output_column, parameters):
    data_df = pd.read_csv(data_path)
    if input_column not in data_df.columns:
        log_error(parameters["logger"], f"Dataframe must have a '{input_column}' column with columns {data_df.columns}")
    if output_column in data_df.columns:
        log_error(parameters["logger"], f"Dataframe already has an output column {output_column} with columns {data_df.columns}")
    return data_df


def get_model(model_name, model_kind):
    if model_kind == "gen":
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    elif model_kind == "clf":
        model = AutoModelForSequenceClassification.from_pretrained(model_name, device_map="auto")
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
        

def get_checkpoint_file(output_csv_path, parameters):
    if os.path.exists(output_csv_path):
        df = pd.read_csv(output_csv_path)
        if "inference_completed" not in df.columns:
            parameters["logger"].warning(f"Found existing output file {output_csv_path} but it does not have an 'inference_completed' column (columns {df.columns}). Restarting from scratch.")
            return None, 0
        else:
            completed_rows = df[df["inference_completed"]]
            if len(completed_rows) == 0:
                return None, 0
            start_idx = completed_rows.index.max() + 1
            return df, start_idx
    else:
        return None, 0
            




@click.command()
@click.option("--model_name", type=str, required=True)
@click.option("--model_kind", type=click.Choice(["gen", "clf"], case_sensitive=False), default="gen")
@click.option("--data_path", type=str, required=True)
@click.option("--output_csv_path", type=str, required=True)
@click.option("--input_column", type=str, default="input")
@click.option("--output_column", type=str, default="output")
@click.option("--save_every", type=int, default=500)
@click.option('--restart_from_checkpoint', type=bool, default=True)
@click.option('--stop_idx', type=int, default=None)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_strings", type=str, default="[STOP]")
@click.option("--remove_stop_strings", type=bool, default=True)
@click.option("--track_output_perplexity", type=bool, default=False)
@click.option("--output_perplexity_column", type=str, default="output_perplexity")
@click.option("--track_input_perplexity", type=bool, default=False)
@click.option("--input_perplexity_column", type=str, default="input_perplexity")
@click.option("--save_hidden", type=bool, default=False)
@click.option("--output_hidden_dir", type=str, required=False)
@click.option("--track_layers", type=str, default="mid+", help="How to track MLP layer outputs. 'mid+' means the 65th percentile layer, 'many' means every one ever four layers, 'all' means all layers and a specific number means that layer.")
@click.option('--track_token', type=click.Choice(["input", "output"], case_sensitive=False), default="input", help="Whether to track hidden state embeddings of the last input or output token.")
@click.pass_obj
def hf_inference(parameters, model_name, model_kind, data_path, output_csv_path, input_column, output_column, save_every, restart_from_checkpoint, stop_idx, max_new_tokens, stop_strings, remove_stop_strings, track_output_perplexity, output_perplexity_column, track_input_perplexity, input_perplexity_column, save_hidden, output_hidden_dir, track_layers, track_token):
    output_hidden_dir = setup_directories(save_hidden, output_csv_path, output_hidden_dir, parameters)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_model(model_name, model_kind)
    config = model.config


    data_df = get_data(data_path, input_column, output_column, parameters)
    hidden_states_lists, all_layers_to_track = get_hidden_states_lists(config, track_layers, parameters)

    if restart_from_checkpoint:
        new_data_df, start_idx = get_checkpoint_file(output_csv_path, parameters)
        if new_data_df is not None:
            data_df = new_data_df
    else:
        start_idx = 0

    if start_idx == 0:
        data_df["inference_completed"] = False

    if stop_idx is not None:
        stop_idx = min(stop_idx, len(data_df))
    else:
        stop_idx = len(data_df)

    for i in tqdm(range(start_idx, stop_idx)):
        prompt = data_df.loc[i, input_column]
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1]
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=stop_strings, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_attentions=True, output_hidden_states=True, output_scores=True, return_dict_in_generate=True)
        output_sequences = output.sequences
        output_normed_perplexity = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True).mean().detach().cpu().numpy().item()  # TODO: Use this to get perplexity of the input too.
        if save_hidden:
            for layer in all_layers_to_track:
                hidden_states = output.hidden_states[0][layer][0, -1].detach().cpu().numpy()  # TODO: Check that this works for classification too
                hidden_states_lists[layer].append(hidden_states)

        output_only = output_sequences[0, input_length:]
        out = tokenizer.decode(output_only, skip_special_tokens=True)
        if remove_stop_strings:
            for stop_string in stop_strings:
                out = out.replace(stop_string, "")

        data_df.loc[i, output_column] = out
        data_df.loc[i, "perplexity"] = output_normed_perplexity
        data_df.loc[i, "inference_completed"] = True

        if (i % save_every == 0 and i > 0) or i == stop_idx - 1:
            data_df.to_csv(output_csv_path, index=False)
            if save_hidden:
                for layer in all_layers_to_track:
                    array = np.array(hidden_states_lists[layer])
                    np.save(os.path.join(output_hidden_dir, f"hidden_states_{layer}.npy"), array)
    return 
