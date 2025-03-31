"""
	1. Load the data
	2. get the base model and tokenizer with bitsandbytes config
	3. Debug print some details of the data
	4. Handle classification separately
	5. Implement SFT, DPO and Pre (packing diff is all)
"""
from utils.parameter_handling import load_parameters
from training.data import load_data, log_token_statistics
from training.model import get_model_tokenizer, get_peft_model_tokenizer
from training.trainers import get_trainer


import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import logging
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, DPOConfig, PPOConfig



default_parameters = load_parameters()

@dataclass
class ScriptArguments:
    training_kind: str = field(metadata={"help": "the kind of training to do. Options: sft, dpo, clf, pre"})
    train_file: str = field(metadata={"help": "the training file"})
    model_name: str = field(default="meta-llama/Llama-2-7b-hf", metadata={"help": "the model name"})
    pretrain_with_output: bool = field(default=True, metadata={"help": "If true, will look for output column during pretraining and try to pretrain on the whole thing after concatenating with a standard template."})
    validation_file: Optional[str] = field(default=None, metadata={"help": "the validation file to use for internal model selection, early stopping etc."})
    test_file: Optional[str] = field(default=None, metadata={"help": "the test file to measure final fit. If not provided, a random split of the validation file is used."})
    train_split: Optional[float] = field(default=0.8, metadata={"help": "the split of the training file to use for training if validation file is not provided"})
    validation_split: Optional[float] = field(default=0.2, metadata={"help": "the split of the validation file to use for internal model selection, early stopping etc."})


    max_train_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of training samples to use"})
    max_valid_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of validation samples to use"})
    max_test_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of test samples to use"})

    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    max_input_length: Optional[int] = field(default=1024, metadata={"help": "the maximum input length to be used only for classification training"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})

    # BitsAndBytesConfig
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "whether to use BitsAndBytes"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "the model dtype. Set to bfloat16 if using BitsAndBytes"})



    # LoraConfig
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use Lora"})
    do_lora: Optional[bool] = field(default=True, metadata={"help": "whether to use lora"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # Seeds    # Log
    log_verbose: Optional[bool] = field(default=False, metadata={"help": "print summary stats of data and processing information."})





if __name__ == "__main__":
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()
    if script_args.training_kind in ["pre", "sft"]:
        parser = HfArgumentParser((ScriptArguments, SFTConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "dpo":
        parser = HfArgumentParser((ScriptArguments, DPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "ppo":
        parser = HfArgumentParser((ScriptArguments, PPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    else:
        pass

    script_args.seed = training_args.seed
    script_args.data_seed = training_args.data_seed

    if script_args.training_kind == "pre":
        training_args.packing = True
    default_parameters['random_seed'] = script_args.data_seed
    set_seed(training_args.seed)
    if script_args.log_verbose:
        default_parameters["logger"].setLevel(logging.DEBUG)

    dataset = load_data(script_args, default_parameters)
    model, tokenizer = None, None
    if script_args.training_kind == "clf" and script_args.use_peft:
        model, tokenizer = get_peft_model_tokenizer(script_args, dataset)
    else:
        model, tokenizer = get_model_tokenizer(script_args, dataset)

    if script_args.log_verbose:
        log_token_statistics(script_args, dataset, tokenizer, default_parameters["logger"])

    trainer = get_trainer(script_args, training_args, dataset, model, tokenizer)



    trainer.train()
    trainer.save_model(training_args.output_dir)

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)