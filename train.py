"""
	1. Load the data
	2. get the base model and tokenizer with bitsandbytes config
	3. Debug print some details of the data
	4. Handle classification separately
	5. Implement SFT, DPO and Pre (packing diff is all)
"""
from utils import load_parameters, log_error, log_info, log_warn
from training.data import load_data, log_token_statistics
from training.model import get_model_tokenizer, get_peft_model_tokenizer
from training.trainers import get_trainer


import os
from dataclasses import dataclass, field
from typing import Optional
import logging

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, DPOConfig, PPOConfig



default_parameters = load_parameters()

@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "the model name"})    
    training_kind: str = field(metadata={"help": "the kind of training to do. Options: sft, dpo, clf, pre"})

    train_file: str = field(metadata={"help": "the training file"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "the validation file to use for internal model selection, early stopping etc."})
    test_file: Optional[str] = field(default=None, metadata={"help": "the test file to measure final fit. If not provided and validation_test split is set, then a random split of the validation file is used."})
    train_validation_split: Optional[float] = field(default=None, metadata={"help": "the split of the training file to use for training if validation file is not provided"})
    validation_test_split: Optional[float] = field(default=None, metadata={"help": "the split of the validation file to use for internal model selection, early stopping etc. The rest is used as test split"})

    input_column: str = field(default="input", metadata={"help": "the input column name"})
    output_column: str = field(default="output", metadata={"help": "the output column name"})
    chosen_column: str = field(default=None, metadata={"help": "the chosen column name for preference training"})
    rejected_column: str = field(default=None, metadata={"help": "the rejected column name for preference training"})
    pretrain_with_output: bool = field(default=True, metadata={"help": "If true, will look for output column during pretraining and try to pretrain on the whole thing after concatenating with a standard template."})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of training samples to use"})
    max_valid_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of validation samples to use"})
    max_test_samples: Optional[int] = field(default=None, metadata={"help": "the maximum number of test samples to use"})

    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    max_input_length: Optional[int] = field(default=512, metadata={"help": "the maximum input length to be used only for classification training"})
    evaluate_before_training: Optional[bool] = field(default=False, metadata={"help": "whether to evaluate before training"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers for huggingface datasets"})

    # LoraConfig
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use Lora"})
    do_lora: Optional[bool] = field(default=True, metadata={"help": "whether to use lora"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})

    # BitsAndBytesConfig
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "whether to use BitsAndBytes"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "the model dtype. Set to bfloat16 if using BitsAndBytes"})


    # Log
    log_verbose: Optional[bool] = field(default=False, metadata={"help": "print summary stats of data and processing information."})

    # Training Setup
    using_deepspeed: bool = field(default=False, metadata={"help": "whether you are using deepspeed"})



def override_defaults(training_args):
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 2
    if training_args.save_steps is None:
        training_args.save_steps = 1000
    if training_args.logging_strategy is None and training_args.logging_steps is None:
        training_args.logging_strategy = "steps"
        training_args.logging_steps = 10
    training_args.report_to = "wandb"



if __name__ == "__main__":
    # Parse arguments. The arguments we expect will depend on the training kind, so we have to parse the args twice. 
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0] # return_remaining_strings stops error out on unknown args
    if script_args.training_kind in ["pre", "sft"]:
        if script_args.training_kind == "sft":
            default_parameters["logger"].warn("SFT is supported, but it works pretty badly. I think this has to do with the data collater class and is hence a bit more involved to fix.") # TODO: Fix SFT
        parser = HfArgumentParser((ScriptArguments, SFTConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "dpo":
        parser = HfArgumentParser((ScriptArguments, DPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "ppo":
        parser = HfArgumentParser((ScriptArguments, PPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "clf":
        parser = HfArgumentParser((ScriptArguments, TrainingArguments))
        script_args, training_args = parser.parse_args_into_dataclasses()
    else:
        log_error(default_parameters["logger"], f"Training kind {script_args.training_kind} not supported. Please use one of sft, dpo, clf, pre.")

    script_args.seed = training_args.seed
    script_args.data_seed = training_args.data_seed
    override_defaults(training_args)
    if script_args.use_bnb:
        training_args.bf16 = True
        script_args.model_dtype = "bfloat16"

    # set up basic arguments
    if script_args.training_kind == "pre":
        training_args.packing = True
    default_parameters['random_seed'] = script_args.data_seed
    set_seed(training_args.seed)
    if script_args.log_verbose:
        default_parameters["logger"].setLevel(logging.DEBUG)
    script_args.logger = default_parameters["logger"]


    dataset = load_data(script_args, default_parameters)


    model, tokenizer = None, None
    if script_args.training_kind == "clf" and script_args.use_peft:     
        # TRL takes in peft_config instead of model, so we load the peft model only for classification which uses Trainer directly
        model, tokenizer = get_peft_model_tokenizer(script_args, dataset)
    else:
        model, tokenizer = get_model_tokenizer(script_args, dataset)

    if script_args.log_verbose:
        log_token_statistics(script_args, dataset, tokenizer, default_parameters["logger"])


    trainer, dataset = get_trainer(script_args, training_args, dataset, model, tokenizer)

    if script_args.training_kind == "clf" and script_args.evaluate_before_training and "test" in dataset:
        trainer.evaluate(dataset["test"])

    trainer.train()

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    #trainer.model.to('cpu')
    if script_args.use_peft:
        trainer.model = trainer.model.merge_and_unload()
    trainer.model.save_pretrained(output_dir)

    if "test" in dataset:
        trainer.model.to("cuda")
        trainer.evaluate(dataset["test"])