#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
import pandas as pd
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_xla_available,
    set_seed,
)
from peft import LoraConfig, TaskType, get_peft_model, IA3Config
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import send_example_telemetry



logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def get_metric_report_str(trainer, metrics):
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to trust the execution of code from datasets/models defined on the Hub."
                " This option should only be set to `True` for repositories you trust and in which you have read the"
                " code, as it will execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=True,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded. "
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )
    peft: str = field(
        default=None,
        metadata={"help": "Kind of PEFT to use. Options are None, lora, ia3."},
    )
    lora_alpha: int = field(
        default=16,
        metadata={"help": "The alpha value to use for LoRA."},
    )
    lora_dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout value to use for LoRA."},
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "The r value to use for LoRA."},
    )
    lora_bias: str = field(
        default="none",
        metadata={"help": "The bias value to use for LoRA."},
    )    

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_internal_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of internal evaluation examples to this "
                "value if set."
            )
        },
    )
    data_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    train_val_split: Optional[float] = field(
        default=0.1,
        metadata={"help": "The ratio of the val subset of the dataset to the training subset"},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    log_file: Optional[str] = field(
        default="clm_ft.log", metadata={"help": "The file to write special logs to."}
    )
    grid_log: bool = field(
        default=False, metadata={"help": "Is this script running gridsearch. If False then deletes previous special log_file"}
    )
    metric: Optional[str] = field(
        default=None, metadata={"help": "The metric to use for evaluation. If None, will be inferred from the dataset.", 
                                "choices": [""]} #TODO: Add choices
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    max_seq_length: Optional[int] = field(
        default=280, metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.data_file is not None:
            assert self.train_file is None and self.validation_file is None, "Cannot use --data_file with --train_file, --validation_file or --test_file"
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(" train file and validation file must be specified. If you want to use single df use data_file argument w train_val_split")

        if self.data_file is None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv"], "`train_file` should be a csv file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv) as `train_file`."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Manually force training args if not set
    training_args.do_train = True
    training_args.do_eval = True
    training_args.eval_strategy = "epoch"
    training_args.auto_find_batch_size  = True
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 2
    if training_args.seed is None:
        training_args.seed = 42

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if not data_args.grid_log:
        if os.path.exists(data_args.log_file):
            os.remove(data_args.log_file)
    special_logging = logging.getLogger("special")
    special_logging.setLevel(logging.DEBUG)
    handler = logging.FileHandler(data_args.log_file)
    formatter = logging.Formatter("%(levelname)s - %(name)s - %(message)s")
    handler.setFormatter(formatter)
    special_logging.addHandler(handler)

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(transformers.logging.ERROR)
    transformers.utils.logging.set_verbosity(transformers.logging.ERROR)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Training hyperparameters
    special_logging.info(f"*** Hyperparameters ***")
    special_logging.info(f"learning_rate: {training_args.learning_rate}")
    if training_args.learning_rate is None or training_args.learning_rate == 5e-05:
        special_logging.warning("\tUsing default learning rate (5e-5). Should likely mess around with it")
    special_logging.info(f"weight_decay: {training_args.weight_decay}")
    if training_args.weight_decay is None or training_args.weight_decay == 0.0:
        special_logging.warning("\tUsing default weight decay (0.0). Should likely mess around with it")
    special_logging.info(f"adam_beta1: {training_args.adam_beta1}")
    if training_args.adam_beta1 is None or training_args.adam_beta1 == 0.9:
        special_logging.warning("\tUsing default adam_beta1 (0.9). Should likely mess around with it")
    special_logging.info(f"adam_beta2: {training_args.adam_beta2}")
    if training_args.adam_beta2 is None or training_args.adam_beta2 == 0.999:
        special_logging.warning("\tUsing default adam_beta2 (0.999). Should likely mess around with it")
    special_logging.info(f"lr_scheduler_type: {training_args.lr_scheduler_type}")
    if training_args.lr_scheduler_type is None or training_args.lr_scheduler_type == "linear":
        special_logging.warning("\tUsing default lr_scheduler_type (linear). Should likely mess around with it")

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.data_file is not None:
        df = pd.read_csv(data_args.data_file)
        df = df.sample(frac=1).reset_index(drop=True)
        n_train = int(len(df) * (1 - data_args.train_val_split))
        train_df = df[:n_train]
        val_df = df[n_train:].reset_index(drop=True)
        save_name_path = data_args.data_file.replace(".csv", "")
        train_df.to_csv(f"{save_name_path}_train.csv", index=False)
        val_df.to_csv(f"{save_name_path}_validation.csv", index=False)
        data_files = {"train": f"{save_name_path}_train.csv", "validation": f"{save_name_path}_validation.csv"}
    else:
        data_files = {"train": data_args.train_file, "validation": data_args.validation_file}


    raw_datasets = load_dataset(
        "csv",
        data_files=data_files,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=model_args.low_cpu_mem_usage,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code, device_map="auto")
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model_args.peft == "lora":
        target_modules = ["k_proj", "v_proj", "down_proj"]
        peft_config = LoraConfig(lora_alpha=model_args.lora_alpha, lora_dropout=model_args.lora_dropout, r=model_args.lora_r, bias=model_args.lora_bias, task_type=TaskType.SEQ_CLS, target_modules=target_modules)
    elif model_args.peft == "ia3":
        peft_config = IA3Config(task_type=TaskType.SEQ_CLS, target_modules=["k_proj", "v_proj", "down_proj"], feedforward_modules=["down_proj"])
    #model.add_adapter(peft_config)
    if model_args.peft is not None:
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.eos_token_id

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = list(raw_datasets["train"].features)
    text_column_name = None
    for name in ["text", "sentence", "input"]:
        if name in column_names:
            text_column_name = name
            break
    output_column_name = None
    for name in ["target", "output"]:
        if name in column_names:
            output_column_name = name
            break
    if text_column_name is None:
        raise ValueError(f"Could not find a text column in the dataset with columns {column_names}. Please make sure the dataset has a text column.")

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    additional_tokens_on_tokenize = len(tokenizer("hello")['input_ids']) - 1 
    if additional_tokens_on_tokenize != 1:
        special_logging.warning(f"Tokenizer adds {additional_tokens_on_tokenize} tokens to the input. This is unexpected.")
        logger.warning(f"Tokenizer adds {additional_tokens_on_tokenize} tokens to the input. This is unexpected.")

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs = examples[text_column_name]
        targets = examples[output_column_name] if output_column_name is not None else ""
        to_tok = [inputs[i] + targets[i] for i in range(len(inputs))]
        model_inputs = tokenizer(to_tok, max_length=data_args.max_seq_length, padding="max_length", truncation=True)
        labels = model_inputs["input_ids"].copy()
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        labels = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels]
        # if targets is not None then replace all input_only_ids with -100 to ignore them in the loss
        if output_column_name is not None:
                input_only_ids = tokenizer(inputs, max_length=data_args.max_seq_length, padding='do_not_pad', truncation=True)['input_ids']
                inp_lengths = [len(i) for i in input_only_ids]
                for i in range(len(labels)):
                    length = inp_lengths[i]
                    for j in range(length):
                        labels[i][j] = -100
        model_inputs["labels"] = labels
        return model_inputs

    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/process#map

    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Tokenizing dataset",
            )
        else:
            lm_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
            )

    train_dataset = lm_datasets["train"].shuffle(seed=training_args.seed)
    train_dataset = train_dataset.shuffle(seed=data_args.seed)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))

    eval_dataset = lm_datasets["validation"].shuffle(seed=training_args.seed)
    internal_eval_dataset = None
    if data_args.max_internal_eval_samples is not None:
        max_internal_eval_samples = min(len(eval_dataset), data_args.max_internal_eval_samples)
        internal_eval_dataset = eval_dataset.select(range(max_internal_eval_samples))
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    if internal_eval_dataset is None:
        internal_eval_dataset = eval_dataset

    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir) # TODO: Look at appropriate metric

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics but we need to shift the labels
        labels = labels[:, 1:].reshape(-1)
        preds = preds[:, :-1].reshape(-1)
        return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=internal_eval_dataset,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
    )

    special_logging.info("*** Before Training Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    trainer.log_metrics("train", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"train metrics: {readable}")


    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"eval metrics: {readable}")

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics

    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    special_logging.info("*** Final Evaluation ***")
    logging.info("*** Final Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"train metrics: {readable}")


    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"eval metrics: {readable}")

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-generation"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()