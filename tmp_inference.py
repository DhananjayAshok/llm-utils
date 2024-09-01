from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("cheat_model", use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("cheat_model", attn_implementation="sdpa", device_map="auto", use_cache=False)
model.eval()
torch.set_grad_enabled(False)
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(model.device)
    output = model.generate(**inputs, max_new_tokens=5)
    output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    del inputs
    return output

import pandas as pd
df = pd.read_csv("data/l/valid.csv")
df['response'] = None
for i in tqdm(range(len(df))):
    df['response'][i] = generate_response(df['prompt'][i])


