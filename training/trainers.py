import torch
from transformers import Trainer, default_data_collator
from trl import SFTTrainer, DPOTrainer
import numpy as np
from training.model import get_peft_config


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
    TODO: Add other metrics like F1, precision and recall
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



def clf_preprocess_function(examples, tokenizer, max_length, label2id):
    """
    Preprocess function for classification tasks.
    Tokenizes the input text, and converts the label to the corresponding id.
    """
    # Tokenize the texts
    result = tokenizer(examples["input"], padding="max_length", max_length=max_length, truncation=True)
    result["label"] = [(label2id[str(l)] if l != -1 else -1) for l in examples["output"]]
    return result

def process_clf(script_args, training_args, dataset, model, tokenizer):
    label2id = model.config.label2id
    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            lambda x: clf_preprocess_function(x, tokenizer, script_args.max_input_length, label2id),
            batched=True,
            num_proc=script_args.num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    return dataset

def get_clf_trainer(script_args, training_args, dataset, model, tokenizer):
    dataset = process_clf(script_args, training_args, dataset, model, tokenizer)
    trainer = WeightedTrainer(
        model=model,
        class_weights=script_args.class_weights,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        compute_metrics=compute_clf_metrics,
        processing_class=tokenizer, # getting processing_class warning Deprication
        data_collator=default_data_collator,
    )
    return trainer, dataset


def prepare_sample_text(example, input_col="input", output_col="output"):
    if output_col is None:
        return example[input_col]
    return f"Input: {example[input_col]} \nOutput: {example[output_col]}"



def get_pre_trainer(script_args, training_args, dataset, model, tokenizer, peft_config):
    input_col = "input"
    output_col = None
    if "output" in dataset["train"].features:
        output_col = "output"

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        peft_config=peft_config,
        formatting_func=lambda x: prepare_sample_text(x, input_col, output_col),
        processing_class=tokenizer,
        args=training_args,
    )
    return trainer, dataset



def get_sft_trainer(script_args, training_args, dataset, model, tokenizer, peft_config):
    response_template = "\nOutput: "
    raise NotImplementedError
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        peft_config=peft_config,
        formatting_func=prepare_sample_text,
        processing_class=tokenizer,
        data_collator=collator,
        args=training_args,
    )
    return trainer, dataset


def get_dpo_trainer(script_args, training_args, dataset, model, tokenizer, peft_config):
    dataset = dataset.rename_column("input", "prompt")
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if "validation" in dataset else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )   
    return trainer, dataset

def get_trainer(script_args, training_args, dataset, model, tokenizer):
    if script_args.training_kind == "clf":
        trainer, dataset = get_clf_trainer(script_args, training_args, dataset, model, tokenizer)
    else:
        peft_config = None
        if script_args.use_peft:
            peft_config = get_peft_config(script_args)
        if script_args.training_kind == "pre":
            trainer, dataset = get_pre_trainer(script_args, training_args, dataset, model, tokenizer, peft_config)
        elif script_args.training_kind == "sft":
            trainer, dataset = get_sft_trainer(script_args, training_args, dataset, model, tokenizer, peft_config)
        elif script_args.training_kind == "dpo":
            trainer, dataset = get_dpo_trainer(script_args, training_args, dataset, model, tokenizer, peft_config)
        else:
            raise ValueError(f"Training kind {script_args.training_kind} not supported")
    return trainer, dataset