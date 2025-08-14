import torch
from transformers import Trainer, default_data_collator, EarlyStoppingCallback
from trl import SFTTrainer, DPOTrainer, KTOTrainer, CPOTrainer
import numpy as np
from training.model import get_peft_config


def get_callback_list(script_args):
    callbacks = []
    if script_args.early_stopping_patience is not None:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=script_args.early_stopping_patience,
                early_stopping_threshold=script_args.early_stopping_threshold,
            )
        )
    return callbacks

class WeightedTrainer(Trainer):
    """
    Trainer subclass that allows for weighted loss functions
    """
    def __init__(self, *args, **kwargs):
        if "class_weights" in kwargs:
            class_weights = kwargs.pop("class_weights")
        else:
            class_weights = None
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights)
        else:
            self.class_weights = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if self.class_weights is not None:        
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.dtype).to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1)) 
        return (loss, outputs) if return_outputs else loss
    

def compute_recall(preds, labels, label):
    """
    Compute the recall for a specific label
    """
    tp = ((preds == label) & (labels == label)).sum()
    fn = ((preds != label) & (labels == label)).sum()
    return tp / (tp + fn) if (tp + fn) > 0 else -1

def compute_precision(preds, labels, label):
    """
    Compute the precision for a specific label
    """
    tp = ((preds == label) & (labels == label)).sum()
    fp = ((preds == label) & (labels != label)).sum()
    return tp / (tp + fp) if (tp + fp) > 0 else -1


def compute_clf_metrics(p):
    """
    Compute the accuracy of a classification model.
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = (preds == p.label_ids).mean()
    result = {"accuracy": acc}
    distinct_labels = set(p.label_ids)
    for label in distinct_labels:
        recall = compute_recall(preds, p.label_ids, label)
        precision = compute_precision(preds, p.label_ids, label)
        result[f"recall_{label}"] = recall
        result[f"precision_{label}"] = precision
    return result



def lm_clf_preprocess_function(examples, tokenizer, max_length, label2id):
    """
    Preprocess function for classification tasks.
    Tokenizes the input text, and converts the label to the corresponding id.
    """
    # Tokenize the texts
    result = tokenizer(examples["input"], padding="max_length", max_length=max_length, truncation=True)
    result["label"] = [(label2id[str(l)] if l != -1 else -1) for l in examples["output"]]
    return result

def vlm_clf_preprocess_function(examples, processor, max_length, label2id):
    raise NotImplementedError()

def process_clf(script_args, training_args, dataset, model, processor):
    label2id = model.config.label2id
    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        if script_args.modality == "lm":
            dataset = dataset.map(
                lambda x: lm_clf_preprocess_function(x, processor, script_args.max_input_length, label2id),
                batched=True,
                num_proc=script_args.num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        elif script_args.modality == "vlm":
            dataset = dataset.map(
                lambda x: vlm_clf_preprocess_function(x, processor, script_args.max_input_length, label2id),
                batched=True,
                num_proc=script_args.num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
    return dataset

def get_clf_trainer(script_args, training_args, dataset, model, processor):
    dataset = process_clf(script_args, training_args, dataset, model, processor)
    callbacks = get_callback_list(script_args)
    trainer = WeightedTrainer(
        model=model,
        class_weights=script_args.class_weights,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        compute_metrics=compute_clf_metrics,
        callbacks=callbacks,
        processing_class=processor,
        data_collator=default_data_collator,
    )
    return trainer, dataset


def prepare_sample_text(example, input_col="prompt", output_col="completion"):
    if output_col not in example:
        return example[input_col]
    return f"Input: {example[input_col]} \nOutput: {example[output_col]}"


def get_trl_renamed_train_val_dataset(dataset):
    """
    Rename the columns of the dataset to match the expected format for TRL trainers.
    """
    train_dataset = dataset["train"].rename_column("input", "prompt")
    if "output" in dataset["train"].features:
        train_dataset = train_dataset.rename_column("output", "completion")
    if "validation" in dataset:
        validation_dataset = dataset["validation"].rename_column("input", "prompt")
        if "output" in dataset["validation"].features:
            validation_dataset = validation_dataset.rename_column("output", "completion")
    else:
        validation_dataset = None
    return train_dataset, validation_dataset


def convert_vlm_conversational_format(text):
    ret = [
        {
            "content": [
                {"type": "image"}, 
                {"type": "text", "text": text}
            ]
        }
    ]
    return ret


def get_trl_vlm_format_train_val_dataset(dataset):
    """
    Convert to the expected conversational format for TRL trainers.
    """
    dataset = dataset.map(lambda x: {"images": [x["image"]]}, 
                          remove_columns=["image"],
                          num_proc=1,
                          desc="Converting images to list format")
    dataset = dataset.map(lambda x: {"input": convert_vlm_conversational_format(x["input"])},
                          num_proc=1,
                          desc="Converting input text to conversational format")
    train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    return train_dataset, validation_dataset



def get_pre_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        formatting_func=prepare_sample_text,
        processing_class=processor,
        completion_only_loss=False,
        args=training_args,
        callbacks=callbacks,
    )
    return trainer, dataset



def get_sft_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        processing_class=processor,
        args=training_args,
        callbacks=callbacks,
    )
    return trainer, dataset


def get_dpo_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        callbacks=callbacks,
        peft_config=peft_config,
    )   
    return trainer, dataset

def get_kto_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)
    trainer = KTOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        callbacks=callbacks,
        peft_config=peft_config,
    )   
    return trainer, dataset

def get_cpo_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)
    trainer = CPOTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=processor,
        callbacks=callbacks,
        peft_config=peft_config,
    )   
    return trainer, dataset

def get_trainer(script_args, training_args, dataset, model, processor):
    if script_args.training_kind == "clf":
        trainer, dataset = get_clf_trainer(script_args, training_args, dataset, model, processor)
    else:
        peft_config = None
        if script_args.use_peft:
            peft_config = get_peft_config(script_args)
        if script_args.training_kind == "pre":
            trainer, dataset = get_pre_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "sft":
            trainer, dataset = get_sft_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "dpo":
            trainer, dataset = get_dpo_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "kto":
            trainer, dataset = get_kto_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "cpo":
            trainer, dataset = get_cpo_trainer(script_args, training_args, dataset, model, processor, peft_config)
        else:
            raise ValueError(f"Training kind {script_args.training_kind} not supported")
    return trainer, dataset