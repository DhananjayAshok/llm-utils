import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, set_seed
import torch
import numpy as np
import os
torch.set_grad_enabled(False)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--model_kind', type=str, required=True, choices=['causal-lm', 'seq-classification', 'seq2seq-lm'])
    parser.add_argument('--input_file_path', type=str, required=True)
    parser.add_argument('--output_file_path', type=str, required=False)  
    parser.add_argument('--input_column', type=str, default=None)
    parser.add_argument('--output_column', type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_predictions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--repetition_penalty", type=float, default=0.01)

    args = parser.parse_args()
    return args

def get_df(args):
    assert args.input_file_path.endswith('.csv'), 'Input file must be a CSV file'
    df = pd.read_csv(args.input_file_path)
    if args.input_column is not None:
        assert args.input_column in df.columns, f'Column {args.input_column} not found in input file'
    else:
        for option in ["text", "sentence", "input"]:
            if option in df.columns:
                args.input_column = option
                break
        if args.input_column is None:
            raise ValueError(f"Input column not found in input file with columns {df.columns}")
    if args.output_column is None:
        args.output_column = 'prediction'
    if args.output_file_path is None:
        assert args.output_column not in df.columns, f'Column {args.output_column} already exists in input file. Please specify an output file path (can repeat input_file_path to overwrite)'
        args.output_file_path = args.input_file_path
    output_dir = os.path.dirname(args.output_file_path)
    if output_dir != "" and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return df

def get_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if args.model_kind == 'causal-lm':
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, device_map="auto")
        # set pad token id to eos token id to avoid generating padding tokens
        model.config.pad_token_id = model.config.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
    elif args.model_kind == 'seq-classification':
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, device_map="auto")
    elif args.model_kind == 'seq2seq-lm':
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, device_map="auto")
    model.eval()
    return tokenizer, model

def predict(args, df, tokenizer, model):
    if args.max_predictions is None:
        args.max_predictions = len(df)
    if args.seed is not None:
        set_seed(args.seed)
    df[args.output_column] = None
    for i, row in tqdm(df.iterrows(), total=args.max_predictions):
        if i >= args.max_predictions:
            df = df.iloc[:i-1]
            break
        text = row[args.input_column]
        if text is None or pd.isna(text) or text == "":
            continue
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=args.max_seq_length).to(model.device)
        if args.model_kind == 'causal-lm':
            inp_shape = inputs['input_ids'].shape[1]
            outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, pad_token_id=tokenizer.eos_token_id)
            outputs = outputs[0, inp_shape:]
            prediction = tokenizer.decode(outputs, skip_special_tokens=True)
        elif args.model_kind == 'seq-classification':
            outputs = model(**inputs)
            logits = outputs.logits.detach().cpu().numpy()[0]
            exped = np.exp(logits)
            softmax = exped / np.sum(exped)
            # if there are only two classes return the probability of the positive class
            if len(softmax) == 2:
                prediction = softmax[1]
            else:
                prediction = softmax
        elif args.model_kind == 'seq2seq-lm':
            outputs = model.generate(**inputs, max_length=100)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        del outputs
        del inputs
        df.at[i, args.output_column] = prediction
    # drop nans in the input or output column
    nanrows = df[args.input_column].isna() | df[args.output_column].isna()
    df = df[~nanrows].reset_index(drop=True)
    df.to_csv(args.output_file_path, index=False)
    
if __name__ == '__main__':
    args = parse_args()
    df = get_df(args)
    tokenizer, model = get_model(args)
    predict(args, df, tokenizer, model)
