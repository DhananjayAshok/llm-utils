import torch
from transformers import Trainer, default_data_collator
from trl import SFTConfig, SFTTrainer, DPOConfig, DPOTrainer, DataCollatorForCompletionOnlyLM
from trl.trainer import ConstantLengthDataset
import numpy as np


class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss()
        #loss_fct = torch.nn.CrossEntropyLoss(weight=torch.tensor([100, 0.01]).to(model.device))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    


def compute_clf_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    acc = (preds == p.label_ids).mean()
    result = {"accuracy": acc}
    return result



def clf_preprocess_function(examples, tokenizer, max_length, label2id):
    # Tokenize the texts
    result = tokenizer(examples["input"], padding=True, max_length=max_length, truncation=True)
    result["output"] = [(label2id[str(l)] if l != -1 else -1) for l in examples["output"]]
    return result

def process_clf(script_args, training_args, dataset, model, tokenizer):
    label2id = model.config.label2id
    # Running the preprocessing pipeline on all the datasets
    with training_args.main_process_first(desc="dataset map pre-processing"):
        dataset = dataset.map(
            lambda x: clf_preprocess_function(x, tokenizer, script_args.max_length, label2id),
            batched=True,
            num_proc=script_args.preprocessing_num_workers,
            load_from_cache_file=False,
            desc="Running tokenizer on dataset",
        )
    return dataset

def get_clf_trainer(script_args, training_args, dataset, model, tokenizer):
    dataset = process_clf(script_args, training_args, dataset, model, tokenizer)
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_clf_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )
    return trainer


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
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_length=script_args.max_length,
        formatting_func=lambda x: prepare_sample_text(x, input_col, output_col),
        processing_class=tokenizer,
        args=training_args,
    )
    return trainer



def get_sft_trainer(script_args, training_args, dataset, model, tokenizer, peft_config):
    response_template = "\nOutput: "
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        max_length=script_args.max_length,
        formatting_func=prepare_sample_text,
        processing_class=tokenizer,
        data_collator=collator,
        args=training_args,
    )
    return trainer


def get_dpo_trainer(script_args, training_args, dataset, model, tokenizer, peft_config):
    trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
        max_length=script_args.max_length,
    )   
    return trainer
