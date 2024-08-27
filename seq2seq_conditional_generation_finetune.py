#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import datasets
import evaluate
import nltk  # Here to have a nice missing dependency error message early on
#nltk.download('punkt_tab')
import numpy as np
import pandas as pd
from datasets import load_dataset
from filelock import FileLock

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from peft import LoraConfig, TaskType, get_peft_model, IA3Config

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def get_metric_report_str(trainer, metrics):
    metric_report = trainer.metrics_format(metrics)
    s = ""
    for key in metric_report:
        s += f"{key}: {metric_report[key]}\n"
    return s


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
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
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_input_length` exceeds "
                "the model's position embeddings."
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


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
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
    print_examples: bool = field(
        default=False, metadata={"help": "Print some examples to logger to check data."}
    )
    log_file: Optional[str] = field(
        default="clf_ft.log", metadata={"help": "The file to write special logs to."}
    )
    grid_log: bool = field(
        default=False, metadata={"help": "Is this script running gridsearch. If False then deletes previous special log_file"}
    )
    input_column: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the input."},
    )
    output_column: Optional[str] = field(
        default="target",
        metadata={"help": "The name of the column in the datasets containing the output."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_input_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_output_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_output_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_output_length`. "
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
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
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id. "
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    metric: Optional[str] = field(
        default="exact_match", metadata={"help": "The metric to use for evaluation.", 
                                "choices": ["bleu", "rogue", "exact_match"]}
    )

    def __post_init__(self):
        if self.data_file is not None:
            assert self.train_file is None and self.validation_file is None, "Cannot use --data_file with --train_file, --validation_file or --test_file"
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(" train file and validation file must be specified. If you want to use single df use data_file argument w train_val_split")

        if self.data_file is None:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."
        if self.val_max_output_length is None:
            self.val_max_output_length = self.max_output_length



def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Manually force training args if not set
    training_args.do_train = True
    training_args.do_eval = True
    training_args.predict_with_generate = True
    training_args.eval_strategy = "epoch"
    training_args.auto_find_batch_size  = True
    if training_args.save_total_limit is None:
        training_args.save_total_limit = 2
    if training_args.seed is None:
        training_args.seed = 42

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_summarization", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if not data_args.grid_log:
        if os.path.exists(data_args.log_file):
            os.remove(data_args.log_file)
    logdir = os.path.dirname(data_args.log_file)
    if logdir != "" and not os.path.exists(logdir):
        os.makedirs(logdir)
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

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
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
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained( #TODO: Get various models in here. 
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
        device_map="auto"
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_input_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                "Increasing the model's number of position embedding vectors from"
                f" {model.config.max_position_embeddings} to {data_args.max_input_length}."
            )
            model.resize_position_embeddings(data_args.max_input_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_input_length)
        else:
            raise ValueError(
                f"`--max_input_length` is set to {data_args.max_input_length}, but the model only has"
                f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                f" `--max_input_length` to {model.config.max_position_embeddings} or to automatically resize the"
                " model's position encodings by passing `--resize_position_embeddings`."
            )

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
    # We need to tokenize inputs and targets.
    column_names = raw_datasets["train"].column_names

    # Get the column names for input/target.
    text_column = data_args.input_column
    if text_column not in column_names:
        raise ValueError(
            f"--input_column' value '{data_args.input_column}' needs to be one of: {', '.join(column_names)}"
        )
    summary_column = data_args.output_column
    if summary_column not in column_names:
        raise ValueError(
            f"--output_column' value '{data_args.output_column}' needs to be one of: {', '.join(column_names)}"
        )

    # Temporarily set max_output_length for training.
    max_output_length = data_args.max_output_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        # remove pairs where at least one record is None

        inputs, targets = [], []
        for i in range(len(examples[text_column])):
            if examples[text_column][i] and examples[summary_column][i]:
                inputs.append(examples[text_column][i])
                targets.append(examples[summary_column][i])

        model_inputs = tokenizer(inputs, max_length=data_args.max_input_length, padding=padding, truncation=True)

        # Tokenize targets with the `text_target` keyword argument
        labels = tokenizer(text_target=targets, max_length=max_output_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    train_dataset = raw_datasets["train"].shuffle(seed=training_args.seed)
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(train_dataset), data_args.max_train_samples)
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    max_output_length = data_args.val_max_output_length
    eval_dataset = raw_datasets["validation"].shuffle(seed=training_args.seed)
    if data_args.max_eval_samples is not None:
        max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
        eval_dataset = eval_dataset.select(range(max_eval_samples))
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )
    internal_eval_dataset = None
    if data_args.max_internal_eval_samples is not None:
        max_internal_eval_samples = min(len(eval_dataset), data_args.max_internal_eval_samples)
        internal_eval_dataset = eval_dataset.select(range(max_internal_eval_samples))
    if internal_eval_dataset is None:
        internal_eval_dataset = eval_dataset

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    ) #TODO: Get other model types here

    # Metric TODO: Customize
    metric = evaluate.load(data_args.metric, cache_dir=model_args.cache_dir)

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        if data_args.metric == "rogue":
            # rougeLSum expects newline after each sentence
            preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
            labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        kwargs = {}
        if data_args.metric == "rouge":
            kwargs = {"use_stemmer": True}
        result = metric.compute(predictions=decoded_preds, references=decoded_labels, **kwargs)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_output_length
    )

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=internal_eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
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


    special_logging.info("*** After Training Evaluation ***")
    metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train")
    trainer.log_metrics("train", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"train metrics: {readable}")


    metrics = trainer.evaluate(eval_dataset=eval_dataset)
    trainer.log_metrics("eval", metrics)
    readable = get_metric_report_str(trainer, metrics)
    special_logging.info(f"eval metrics: {readable}")

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "summarization"}

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)

    return


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()