import logging
from dataclasses import dataclass, field
import os
import warnings
import random
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments
from trl.commands.cli_utils import  TrlParser
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
        set_seed,

)
from trl import setup_chat_format
from peft import LoraConfig
import pandas as pd

from trl import (
   SFTTrainer)

# Comment in if you want to use the Llama 3 instruct template but make sure to add modules_to_save
# LLAMA_3_CHAT_TEMPLATE="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"


# ACCELERATE_USE_FSDP=1 FSDP_CPU_RAM_EFFICIENT_LOADING=1 torchrun --nproc_per_node=4 ./scripts/run_fsdp_qlora.py --config llama_3_70b_fsdp_qlora.yaml

@dataclass
class ScriptArguments:
    train_file: str = field(
        default=None,
        metadata={
            "help": "Path to the dataset folder with a train and valid csv file inside them"
        },
    )
    validation_file: str = field(
        default=None,
        metadata={"help": "Path to the dataset folder with a train and valid csv file inside them"},
    )
    max_train_samples: int = field(
        default=None,
        metadata={"help": "The maximum number of training samples to use"}
    )
    model_name_or_path: str = field(
        default=None, metadata={"help": "Model ID to use for SFT training"}
    )
    max_seq_length: int = field(
        default=512, metadata={"help": "The maximum sequence length for SFT Trainer"}
    )


def training_function(script_args, training_args):
    ################
    # Dataset
    ################
    
    train_dataset = load_dataset(
        "csv",
        data_files=script_args.train_file,
        split="train",
    )
    test_dataset = load_dataset(
        "csv",
        data_files=script_args.validation_file,
        split="train",
    )

    # shuffle the training dataset
    train_dataset = train_dataset.shuffle(seed=training_args.seed)
    # if max_train_samples is set, we only use a subset of the training dataset
    train_size = len(train_dataset)
    test_size = len(test_dataset)
    train_dataset = train_dataset.filter(lambda x: pd.notnull(x["text"]))
    test_dataset = test_dataset.filter(lambda x: pd.notnull(x["text"]))
    if len(train_dataset) != train_size:
        warnings.warn(f"Removed {train_size - len(train_dataset)} samples from the training dataset because they had missing text")
    if len(test_dataset) != test_size:
        warnings.warn(f"Removed {test_size - len(test_dataset)} samples from the test dataset because they had missing text")
    if script_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), script_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    

    ################
    # Model & Tokenizer
    ################

    # Tokenizer        
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Model    
    torch_dtype = torch.bfloat16
    quant_storage_dtype = torch.bfloat16

    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_storage=quant_storage_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        attn_implementation="sdpa", # use sdpa, alternatively use "flash_attention_2"
        device_map="auto",
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
#        quantization_config=quantization_config,
#        torch_dtype=quant_storage_dtype,
    )
    
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    ################
    # PEFT
    ################

    # LoRA config based on QLoRA paper & Sebastian Raschka experiment
    peft_config = LoraConfig(
        lora_alpha=8,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        dataset_text_field="text",
        eval_dataset=test_dataset,
        peft_config=peft_config,
        max_seq_length=script_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )
    if trainer.accelerator.is_main_process:
        trainer.model.print_trainable_parameters()

    ##########################
    # Train model
    ##########################
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)

    ##########################
    # SAVE MODEL FOR SAGEMAKER
    ##########################
    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model()
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_and_config()    
    
    # set use reentrant to False
    if training_args.gradient_checkpointing:
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}
    # set seed
    set_seed(training_args.seed)
  
    # launch training
    training_function(script_args, training_args)