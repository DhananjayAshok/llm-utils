import torch
from transformers import Trainer, default_data_collator, EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer, DPOTrainer, KTOTrainer, CPOTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from training.unlearning import GATrainer, NPOTrainer
import numpy as np
from training.model import get_peft_config
from utils.vlm_utils import infer_vlm_kind, get_single_vlm_text, get_vlm_text
from utils import log_info
import wandb

class SampleLoggingCallback(TrainerCallback):
    def __init__(self, training_kind, modality, n_eval_output_batches: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_kind = training_kind
        self.modality = modality
        self.n_eval_output_batches = n_eval_output_batches
        self.input_ids_key_name = "input_ids"
        if self.training_kind in ["dpo", "npo"]:
            self.input_ids_key_name = "prompt_input_ids"
        self.output_ids_key_name = "labels"
        if self.training_kind in ["dpo"]:
            self.output_ids_key_name = "chosen_input_ids"
        if self.training_kind in ["npo"]:
            self.output_ids_key_name = "rejected_input_ids"
        self.table = wandb.Table(columns=["global_step", "item_id", "input", "target_output", "model_output"], log_mode="MUTABLE")

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
         # TODO: This might fail for VLMs. Needs testing.
        batch = next(iter(eval_dataloader))
        all_input_texts = []
        all_targets = []
        all_outputs = []
        processor = kwargs.get("processing_class")
        for i, batch in enumerate(eval_dataloader):
            if i >= self.n_eval_output_batches:
                break

        # Generate output
        if self.training_kind == "clf":
            input_texts = processor.batch_decode(batch[self.input_ids_key_name], skip_special_tokens=True)        
            all_input_texts.extend(input_texts)        
            targets = batch[self.output_ids_key_name]
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            all_outputs.extend(preds)
        else:
            starting_indices = (batch[self.output_ids_key_name] != -100).int().argmax(dim=1)
            real_input_texts = []
            real_targets = []
            for j, start_idx in enumerate(starting_indices):
                input_ids = batch[self.input_ids_key_name][j][:start_idx]
                text = processor.decode(input_ids, skip_special_tokens=True)
                real_input_texts.append(text)
                output_ids = batch[self.output_ids_key_name][j][start_idx:]
                output_text = processor.decode(output_ids[output_ids != -100], skip_special_tokens=True)
                real_targets.append(output_text)
            all_input_texts.extend(real_input_texts)
            all_targets.extend(real_targets)
            current_padding_side = processor.padding_side
            processor.padding_side = "left"
            input_ids = processor(batch[self.input_ids_key_name], return_tensors="pt", padding=True).input_ids
            processor.padding_side = current_padding_side
            input_ids.to(model.device)
            input_length = input_ids.shape[1]
            gen_kwargs = {}
            if self.modality == "vlm":
                gen_kwargs = {"pixel_values": batch["pixel_values"]} # TODO: This might fail for some models / learning algorithms. Needs testing. 
            outputs = model.generate(input_ids=input_ids, **gen_kwargs)
            outputs = outputs[:, input_length:]
            output_texts = processor.batch_decode(outputs, skip_special_tokens=True)                                                                                                                                        
            all_outputs.extend(output_texts)
        for j, values in enumerate(zip(all_input_texts, all_targets, all_outputs)):
            input_text, target, output = values
            self.table.add_data(state.global_step, j, input_text, target, output)
        wandb.log({"Sample Outputs": self.table})
        return

def get_callback_list(script_args):
    callbacks = [SampleLoggingCallback(script_args.training_kind, script_args.modality, script_args.n_eval_output_batches)]
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
            self.class_weights = self.class_weights / self.class_weights.sum() # normalize
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


def vlm_clf_preprocess_function(examples, processor, max_length, label2id, vlm_kind):
    vlm_texts = get_vlm_text(vlm_kind=vlm_kind, input_texts=examples["input"].list())
    result = processor(text=vlm_texts, images=examples["image"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
    result["label"] = [(label2id[str(l)] if l != -1 else -1) for l in examples["output"]]
    return result

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
            vlm_kind = infer_vlm_kind(model_name=None, config=model.config)
            dataset = dataset.map(
                lambda x: vlm_clf_preprocess_function(x, processor, script_args.max_input_length, label2id, vlm_kind),
                batched=True,
                num_proc=script_args.num_workers,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
    return dataset


def infer_class_weights(train_dataset):
    """
    Infer class weights from the training dataset.
    """
    labels = train_dataset["label"]
    label_counter = {}
    for label in labels:
        if label not in label_counter:
            label_counter[label] = 0
        label_counter[label] += 1
    n_labels = len(label_counter)
    class_weights = []
    for i in range(n_labels):
        count = label_counter[i]
        class_weight = 1 / (count)
        class_weights.append(class_weight)
    return class_weights


def get_clf_trainer(script_args, training_args, dataset, model, processor):
    dataset = process_clf(script_args, training_args, dataset, model, processor)
    callbacks = get_callback_list(script_args)
    if script_args.auto_infer_class_weights:
        script_args.class_weights = infer_class_weights(dataset["train"])
        log_info(f"Inferred class weights: {script_args.class_weights}. This is pre-normalization.", script_args.parameters)

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
    return f"{example[input_col]}\nOutput: {example[output_col]}"


def get_trl_renamed_train_val_dataset(dataset):
    """
    Rename the columns of the dataset to match the expected format for TRL trainers.
    """
    train_dataset = dataset["train"].rename_column("input", "prompt")
    if "output" in dataset["train"].features:
        train_dataset = train_dataset.rename_column("output", "completion")
    else:
        # make a dummy completion column with empty strings. Seems to work for pretraining. 
        train_dataset = train_dataset.add_column("completion", [""] * len(train_dataset))
    if "validation" in dataset:
        validation_dataset = dataset["validation"].rename_column("input", "prompt")
        if "output" in dataset["validation"].features:
            validation_dataset = validation_dataset.rename_column("output", "completion")
    else:
        validation_dataset = None
    return train_dataset, validation_dataset


def get_trl_vlm_format_train_val_dataset(dataset, model):
    """
    Convert to the expected conversational format for TRL trainers.
    """
    model_config = model.config
    vlm_kind = infer_vlm_kind(model_name=None, config=model_config)
    dataset = dataset.map(lambda x: {"input": get_single_vlm_text(vlm_kind, x["input"])},
                          num_proc=1,
                          desc="Converting input text to VLM format")
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
        args=training_args,
        callbacks=callbacks,
    )
    return trainer, dataset


def get_sft_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset, model)
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

def get_ga_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset, model)
    callbacks = get_callback_list(script_args)
    training_args.remove_unused_columns = False
    trainer = GATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        processing_class=processor,
        args=training_args,
        callbacks=callbacks,
        data_collator=ga_data_collator
    )
    return trainer, dataset

def ga_data_collator(batch):
    #ref: https://github.com/huggingface/trl/blob/c26b375ca3dd47e9cd9fdfd820e89bb4af669186/trl/trainer/sft_trainer.py#L118
    sft_collator = DataCollatorForLanguageModeling(pad_token_id=0) #TODO: change; some models have no pad tokens, there is a pad_free argument but I need to verify that this will still work correctly if I use it 
    batch_data = sft_collator(batch)
    batch_data["forget"] = torch.tensor([example["forget"] for example in batch], dtype=torch.float)
    return batch_data


def get_npo_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset, validation_dataset = get_trl_vlm_format_train_val_dataset(dataset)
    callbacks = get_callback_list(script_args)
    trainer = NPOTrainer(
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
        elif script_args.training_kind == "ga":
            trainer, dataset = get_ga_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "npo":
            trainer, dataset = get_npo_trainer(script_args, training_args, dataset, model, processor, peft_config)
        else:
            raise ValueError(f"Training kind {script_args.training_kind} not supported")
    return trainer, dataset
