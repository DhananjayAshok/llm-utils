"""
	1. Load the data
	2. get the base model and tokenizer with bitsandbytes config
	3. Debug print some details of the data
	4. Handle classification separately
	5. Implement SFT, DPO and Pre (packing diff is all)
"""
from utils import load_parameters, log_error, log_info, log_warn
from training.data import load_data, log_token_statistics
from training.model import get_model_processor, get_peft_model_processor
from training.trainers import get_trainer
from accelerate import Accelerator


import os
import yaml
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional, List
import logging

from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, DPOConfig, KTOConfig, CPOConfig
from datetime import datetime, timezone



def save_args(script_args, training_args):
    parameters = script_args.parameters
    output_dir = training_args.output_dir
    output_path = os.path.join(output_dir, "run_args.yaml")
    parameters.update(asdict(script_args))
    if "accelerator" in parameters:
        parameters.pop("accelerator") # cant serialize this
    parameters.update(asdict(training_args))
    with open(output_path, "w") as f:
        yaml.dump(parameters, f)
    log_info(f"Saved script arguments to {output_path}\n"
             f"HOWEVER: You can also find a config file in <run_location>/wandb/<run_name>/files/config.yaml",
             parameters)



default_parameters = load_parameters()
default_parameters["run_start_time"] = datetime.now(timezone.utc).strftime("%Y-%m-%d-H%H-M%M-S%S")

@dataclass
class ScriptArguments:
    model_name: str = field(metadata={"help": "the model name"})
    training_kind: str = field(metadata={"help": "the kind of training to do. Options: clf, pre, sft, dpo, kto, cpo"})
    train_file: str = field(metadata={"help": "the training file"})

    modality: str = field(default="lm", metadata={"help": "the modality of the model. Options: lm, vlm"})

    validation_file: Optional[str] = field(default=None, metadata={"help": "the validation file to use for internal model selection, early stopping etc."})
    test_file: Optional[str] = field(default=None, metadata={"help": "the test file to measure final fit. If not provided and validation_test split is set, then a random split of the validation file is used."})
    train_validation_split: Optional[float] = field(default=None, metadata={"help": "the split of the training file to use for training if validation file is not provided. The rest is used as validation split"})
    validation_test_split: Optional[float] = field(default=None, metadata={"help": "the split of the validation file to use for internal model selection, early stopping etc. The rest is used as test split"})
    eval_max_new_tokens: Optional[int] = field(default=512, metadata={"help": "the maximum number of tokens to use during evaluation."})

    input_column: str = field(default="input", metadata={"help": "the input column name"})
    image_input_column: str = field(default="image", metadata={"help": "the image input column name. Should contain paths to image files. Only used for VLMs."})
    ga_forget_column: str = field(default="forget", metadata={"help": "the forget column name for GA training. Must be a boolean column indicating whether to forget the sample or not."})
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

    # Training Arguments
    class_weights: Optional[List[float]] = field(default=None, metadata={"help": "the class weights to use for classification training. If not provided, will be uniform."})
    auto_infer_class_weights: Optional[bool] = field(default=False, metadata={"help": "whether to automatically infer class weights from the training data. Only used if class_weights is not provided. Will override class_weights if both are provided."})
    early_stopping_patience: Optional[int] = field(default=None, metadata={"help": "the number of steps to wait for improvement before stopping training"})
    early_stopping_threshold: Optional[float] = field(default=0.0, metadata={"help": "the threshold for early stopping. If the validation loss does not improve by this amount, training will stop."})

    # PEFT Config
    use_peft: Optional[bool] = field(default=True, metadata={"help": "whether to use PEFT"})
    do_lora: Optional[bool] = field(default=True, metadata={"help": "whether to use lora"})
    lora_alpha: Optional[float] = field(default=16, metadata={"help": "the lora alpha parameter"})
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": "the lora dropout parameter"})
    lora_r: Optional[int] = field(default=8, metadata={"help": "the lora r parameter"})
    lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": "the lora target modules. If not provided, will use all-linear."})

    # BitsAndBytesConfig
    use_bnb: Optional[bool] = field(default=False, metadata={"help": "whether to use BitsAndBytes"})
    model_dtype: Optional[str] = field(default="float16", metadata={"help": "the model dtype. Set to bfloat16 if using BitsAndBytes"})

    # Checkpoint logic
    overwrite_final : Optional[bool] = field(default=False, metadata={"help": "whether to overwrite output_dir/final_checkpoint if it exists"})

    # Log
    log_verbose: Optional[bool] = field(default=False, metadata={"help": "print summary stats of data and processing information."})
    n_eval_output_batches: Optional[int] = field(default=1, metadata={"help": "the number of evaluation batches to use for logging outputs."})
    do_debug: Optional[bool] = field(default=False, metadata={"help": "whether to run in debug mode. This allows you to set breakpoints etc inside the conditional, without breaking other running scripts that use this repo."})


def search_for_checkpoint(output_dir, parameters):
    if not os.path.exists(output_dir):
        return None
    final_exists = os.path.exists(output_dir + "/final_checkpoint")
    checkpoints = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if final_exists:
        log_info(f"Detected final_checkpoint, but cannot restart from final checkpoint as trainer information is not stored. Looking for others.", parameters)
    if len(checkpoints) == 0:
        log_info(f"{output_dir} exists but no checkpoint found, starting from scratch.", parameters)
        return None
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))
    log_info(f"Found {len(checkpoints)} checkpoints in {output_dir}. Resuming from {checkpoints[-1]}.", parameters)
    return checkpoints[-1]


def override_defaults(training_args, parameters=default_parameters):
    if script_args.training_kind != "clf":
        if script_args.validation_test_split is not None or script_args.test_file is not None:
            log_warn(f"Test split evaluation is not supported for {script_args.training_kind}, removing test set related arguments", default_parameters)
            script_args.validation_test_split = None
            script_args.test_file = None
    if training_args.resume_from_checkpoint is not None and not isinstance(training_args.resume_from_checkpoint, bool):
        if training_args.resume_from_checkpoint.lower().strip() in ["true", "1", "false", "0"]:
            training_args.resume_from_checkpoint = bool(training_args.resume_from_checkpoint)
        else: # then it is a path
            if not os.path.exists(training_args.resume_from_checkpoint):
                log_error(f"resume_from_checkpoint is set to {training_args.resume_from_checkpoint} but this path does not exist.", parameters)
            training_args.resume_from_checkpoint = training_args.resume_from_checkpoint
    if training_args.resume_from_checkpoint == True:
        if not os.path.exists(training_args.output_dir):
            log_warn(f"resume_from_checkpoint is set to True but output_dir {training_args.output_dir} does not exist. Starting training from scratch.", parameters)
            training_args.resume_from_checkpoint = False
        else:
            log_warn("Trying to resume from checkpoint.", parameters)
            available_checkpoint = search_for_checkpoint(training_args.output_dir, parameters)
            if available_checkpoint is None:
                training_args.resume_from_checkpoint = False
            else:
                training_args.resume_from_checkpoint = available_checkpoint # TODO: Debug, this might not work for classification as it may need to load it instead of the model. 
    elif training_args.resume_from_checkpoint is None or training_args.resume_from_checkpoint == False:
        training_args.resume_from_checkpoint = False
    if os.path.exists(training_args.output_dir + "/final_checkpoint"):
        if script_args.overwrite_final:
            log_warn(f"final_checkpoint already exists in {training_args.output_dir} but overwrite_final is set to True. Will end up overwriting final checkpoint after training...", parameters)
        else:
            log_info(f"final_checkpoint already exists in {training_args.output_dir}. To force overwrite, set overwrite_final to True.", parameters)
            sys.exit(0)
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 2
    if training_args.save_steps is None:
        training_args.save_steps = 1000
    if training_args.logging_strategy is None and training_args.logging_steps is None:
        training_args.logging_strategy = "steps"
        training_args.logging_steps = 10
    if training_args.output_dir is None:
        training_args.output_dir = parameters["tmp_dir"] + "/" + parameters["run_start_time"] + "/"
    os.makedirs(training_args.output_dir, exist_ok=True)
    training_args.report_to = "wandb"
    if script_args.training_kind == "ga":
        script_args.remove_unused_columns = False # needed to keep forget column



if __name__ == "__main__":
    # Parsing and setting up the arguments
    # region
    # Parse arguments. The arguments we expect will depend on the training kind, so we have to parse the args twice. 
    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]  # return_remaining_strings stops error out on unknown args
    accelerator = Accelerator()
    distributed = accelerator.num_processes > 1
    if script_args.training_kind in ["pre", "sft", "ga", "npo"]:
        parser = HfArgumentParser((ScriptArguments, SFTConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
        if script_args.training_kind == "pre":
            training_args.packing = True
            training_args.completion_only_loss = False
    elif script_args.training_kind in ["dpo"]:
        parser = HfArgumentParser((ScriptArguments, DPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "kto":
        raise NotImplementedError(f"I think there is a bug in the KTO automatic dataset conversion")
        parser = HfArgumentParser((ScriptArguments, KTOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "cpo":
        parser = HfArgumentParser((ScriptArguments, CPOConfig))
        script_args, training_args = parser.parse_args_into_dataclasses()
    elif script_args.training_kind == "clf":
        parser = HfArgumentParser((ScriptArguments, TrainingArguments))
        script_args, training_args = parser.parse_args_into_dataclasses()
    else:
        if accelerator.is_main_process:
            log_error(f"Training kind {script_args.training_kind} not supported. Please use one of sft, dpo, clf, pre.", default_parameters)

    script_args.seed = training_args.seed
    script_args.data_seed = training_args.data_seed
    script_args.distributed = distributed
    if accelerator.is_main_process:
        if training_args.resume_from_checkpoint is None:
            log_warn("resume_from_checkpoint is not set, defaulting to True. This will search for the output directory and if not found, will start training from scratch. To avoid this, explicitly set --resume_from_checkpoint arg", default_parameters)        
            training_args.resume_from_checkpoint = True
    override_defaults(training_args)
    if accelerator.is_main_process:
        log_info(f"Saving to: {training_args.output_dir}", default_parameters)
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
    script_args.parameters = default_parameters
    if accelerator.is_main_process:
        save_args(script_args, training_args)
    script_args.accelerator = accelerator
    # endregion

    dataset = load_data(script_args)

    model, processor = None, None
    if script_args.training_kind == "clf" and script_args.use_peft:     
        # TRL takes in peft_config instead of model, so we load the peft model only for classification which uses Trainer directly
        model, processor = get_peft_model_processor(script_args, dataset)
    else:
        model, processor = get_model_processor(script_args, dataset)

    if script_args.log_verbose:
        log_token_statistics(script_args, dataset, processor, script_args.parameters)

    trainer, dataset = get_trainer(script_args, training_args, dataset, model, processor)

    if script_args.evaluate_before_training and "test" in dataset:
        trainer.evaluate(dataset["test"], metric_key_prefix="test") # This will fail if run in FSDP: https://github.com/huggingface/transformers/issues/39961

    if accelerator.is_main_process:
        log_info(f"Training starting with {len(trainer.train_dataset)} samples ...")
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    if accelerator.is_main_process:
        log_info("Training completed. Saving model, this may take some time depending on your setup ...")

    if "test" in dataset:
        trainer.evaluate(dataset["test"], metric_key_prefix="test")

    output_dir = os.path.join(training_args.output_dir, "final_checkpoint")
    trainer.processing_class.save_pretrained(output_dir)
    trainer.model.config.save_pretrained(output_dir)
    if script_args.use_peft:
        trainer.model = trainer.model.merge_and_unload()

    is_main_process = accelerator.is_main_process
    save_function = accelerator.save
    state_dict = accelerator.get_state_dict(trainer.model)
    trainer.model.save_pretrained(output_dir, is_main_process=is_main_process, state_dict=state_dict, save_function=save_function)
    if accelerator.is_main_process:
        log_info(f"Model saved to {output_dir}", script_args.parameters)
    accelerator.end_training()