from utils.log_handling import log_error
import click
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
import pandas as pd
from tqdm import tqdm
import numpy as np
import os


def setup_directories(save_hidden, output_csv_path, output_hidden_dir, parameters):
    makedirs = [os.path.dirname(output_csv_path)]
    if save_hidden:
        if output_hidden_dir is None:
            output_hidden_dir = os.path.join(os.path.dirname(output_csv_path), "hidden_states")
            if os.path.exists(output_hidden_dir):
                log_error(parameters["logger"], f"Output hidden directory {output_hidden_dir} already exists.")
        
        makedirs.append(output_hidden_dir)
    for directory in makedirs:
        if not os.path.exists(directory) and not os.path.dirname(directory) == "":
            os.makedirs(directory)
    return


def get_data(data_path, input_column, output_column, output_perplexity_column, input_perplexity_column, parameters):
    data_df = pd.read_csv(data_path)
    if input_column not in data_df.columns:
        log_error(parameters["logger"], f"Dataframe must have a '{input_column}' column with columns {data_df.columns}")
    if output_column in data_df.columns:
        log_error(parameters["logger"], f"Dataframe already has an output column {output_column} with columns {data_df.columns}")
    if output_perplexity_column is not None and output_perplexity_column in data_df.columns:
        log_error(parameters["logger"], f"Dataframe already has an output perplexity column {output_perplexity_column} with columns {data_df.columns}")
    if input_perplexity_column is not None and input_perplexity_column in data_df.columns:
        log_error(parameters["logger"], f"Dataframe already has an input perplexity column {input_perplexity_column} with columns {data_df.columns}")
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
@click.option("--input_csv_path", type=str, required=True)
@click.option("--output_csv_path", type=str, default=None, help="If not provided, will be set to the input CSV path with '_output' appended before the file extension.")
@click.option("--input_column", type=str, default="input")
@click.option("--output_column", type=str, default="output")
@click.option("--batch_size", type=int, default=1)
@click.option("--save_every", type=int, default=0.2)
@click.option('--restart_from_checkpoint', type=bool, default=True)
@click.option('--stop_idx', type=int, default=None)
@click.option("--max_new_tokens", type=int, default=10)
@click.option("--stop_strings", type=str, default=["[STOP]"], multiple=True)
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
def hf_inference(parameters, model_name, model_kind, input_csv_path, output_csv_path, input_column, output_column, batch_size, save_every, restart_from_checkpoint, stop_idx, max_new_tokens, stop_strings, remove_stop_strings, track_output_perplexity, output_perplexity_column, track_input_perplexity, input_perplexity_column, save_hidden, output_hidden_dir, track_layers, track_token):
    if output_csv_path is None:
        output_csv_path = input_csv_path.replace(".csv", "_output.csv")
    setup_directories(save_hidden, output_csv_path, output_hidden_dir, parameters)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = get_model(model_name, model_kind)
    config = model.config

    if not track_input_perplexity:
        input_perplexity_column = None
    if not track_output_perplexity:
        output_perplexity_column = None
    data_df = get_data(input_csv_path, input_column, output_column, output_perplexity_column, input_perplexity_column, parameters)
    if save_hidden:
        if batch_size > 1:
            log_error(parameters["logger"], f"Batch size {batch_size} is greater than 1. This is not supported for hidden state tracking.")
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

    save_every = int(save_every * ((stop_idx - start_idx) / batch_size))
    for i in tqdm(range(start_idx, stop_idx, batch_size)):
        prompts = data_df.loc[i:i+batch_size-1, input_column].tolist()
        inputs = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(model.device)
        input_length = inputs["input_ids"].shape[1] # TODO: Check this works for batch size > 1
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, stop_strings=stop_strings, pad_token_id=tokenizer.eos_token_id, tokenizer=tokenizer, output_attentions=False, output_hidden_states=save_hidden, output_scores=track_input_perplexity or track_output_perplexity, return_dict_in_generate=True)
        output_sequences = output.sequences
        if save_hidden:
            for layer in all_layers_to_track:
                hidden_states = output.hidden_states[0][layer][0, -1].detach().cpu().numpy()  # TODO: Check that this works for classification too
                hidden_states_lists[layer].append(hidden_states)
        if track_output_perplexity:
            output_normed_perplexity = model.compute_transition_scores(output.sequences, output.scores, normalize_logits=True).mean().detach().cpu().numpy().item()
        if track_input_perplexity:
            input_normed_perplexity = None # TODO: Use this to get perplexity of the input too.
            pass

        output_only = output_sequences[:, input_length:]
        out = tokenizer.batch_decode(output_only, skip_special_tokens=True)
        if remove_stop_strings:
            for out_i in range(len(out)):
                for stop_string in stop_strings:
                    out[out_i] = out[out_i].replace(stop_string, "")

        data_df.loc[i:i+batch_size-1, output_column] = out
        if track_output_perplexity:
            data_df.loc[i, output_perplexity_column] = output_normed_perplexity
        if track_input_perplexity:
            data_df.loc[i, input_perplexity_column] = input_normed_perplexity
        data_df.loc[i, "inference_completed"] = True

        if (i % save_every == 0 and i > 0) or i == stop_idx - 1:
            data_df.to_csv(output_csv_path, index=False)
            if save_hidden:
                for layer in all_layers_to_track:
                    array = np.array(hidden_states_lists[layer])
                    np.save(os.path.join(output_hidden_dir, f"hidden_states_{layer}.npy"), array)
    return 
