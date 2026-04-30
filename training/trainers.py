import torch
from transformers import Trainer, default_data_collator, EarlyStoppingCallback, TrainerCallback
from trl import SFTTrainer, DPOTrainer
from trl.trainer.sft_trainer import DataCollatorForLanguageModeling
from trl.trainer.sft_trainer import prepare_multimodal_messages
from training.unlearning import GATrainer, NPOTrainer
import numpy as np
from training.model import get_peft_config
from training.data import drop_column_if_needed
from utils import log_info
from PIL import Image
import wandb


class StopOnZeroLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        # Check if the training loss is available and is approximately zero
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log and last_log["loss"] < 1e-9:  # Using a small threshold for "zero"
                print(f"Training loss reached zero ({last_log['loss']}), stopping training...")
                control.should_training_stop = True
        return control

class SampleLoggingCallback(TrainerCallback):
    def __init__(self, training_kind, modality, n_eval_output_batches: int, eval_max_new_tokens: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_kind = training_kind
        self.modality = modality
        self.n_eval_output_batches = n_eval_output_batches
        self.eval_max_new_tokens = eval_max_new_tokens
        self.input_ids_key_name = "input_ids"
        self.output_ids_key_name = "labels"        
        if self.training_kind in ["dpo"]:
            self.input_ids_key_name = "prompt_input_ids"
            self.output_ids_key_name = "chosen_input_ids"
            self.rejected_ids_key_name = "rejected_input_ids"
        base_columns = ["global_step", "item_id", "input"]
        if self.modality == "vlm":
            base_columns.append("input_images")
        if self.training_kind in ["clf", "sft", "ga", "npo", "pre"]:
            base_columns.extend(["target_output", "model_output"])
        elif self.training_kind in ["dpo"]:
            base_columns.extend(["chosen_output", "rejected_output", "model_output"])
        self.table = wandb.Table(columns=base_columns, log_mode="MUTABLE")

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        batch = next(iter(eval_dataloader))
        all_input_texts = []
        all_targets = [] # also all_chosens
        all_outputs = []
        all_rejecteds = []
        all_images = []
        processor = kwargs.get("processing_class")
        for i, batch in enumerate(eval_dataloader):
            if i >= self.n_eval_output_batches:
                break

        if self.training_kind == "clf":
            input_texts = processor.batch_decode(batch[self.input_ids_key_name], skip_special_tokens=True)        
            all_input_texts.extend(input_texts)        
            targets = batch[self.output_ids_key_name]
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=-1).detach().cpu().numpy().tolist()
            all_outputs.extend(preds)
        else:
            if self.training_kind in ["sft", "ga", "npo", "dpo"]:
                starting_indices = (batch[self.output_ids_key_name] != -100).int().argmax(dim=1)
            elif self.training_kind in ["pre"]:
                # then starting_indices is the halfway point of the input ids
                starting_indices = (batch[self.output_ids_key_name].shape[1]//2 * torch.ones(batch[self.output_ids_key_name].shape[0], dtype=torch.int)).to(batch[self.output_ids_key_name].device)
            real_input_texts = []
            real_targets = []
            real_rejecteds = []
            for j, start_idx in enumerate(starting_indices):
                if self.training_kind in ["sft", "ga", "pre", "npo"]:
                    input_ids = batch[self.input_ids_key_name][j][:start_idx]
                elif self.training_kind in ["dpo"]:
                    input_ids = batch[self.input_ids_key_name][j] # start_idx is always 0
                text = processor.decode(input_ids, skip_special_tokens=True)
                real_input_texts.append(text)
                if self.training_kind in ["sft", "ga", "npo", "dpo"]:
                    output_ids = batch[self.output_ids_key_name][j][start_idx:]
                elif self.training_kind in ["pre"]:
                    output_ids = batch[self.output_ids_key_name][j][start_idx:start_idx+self.eval_max_new_tokens]
                output_text = processor.decode(output_ids[output_ids != -100], skip_special_tokens=True)
                real_targets.append(output_text)
                if self.training_kind in ["dpo"]:
                    rejected_ids = batch[self.rejected_ids_key_name][j][start_idx:]
                    rejected_text = processor.decode(rejected_ids[rejected_ids != -100], skip_special_tokens=True)
                    real_rejecteds.append(rejected_text)
                else:
                    real_rejecteds.append("")
            all_input_texts.extend(real_input_texts)
            all_targets.extend(real_targets)
            all_rejecteds.extend(real_rejecteds)
            if self.modality == "vlm":
                # Collect images for wandb logging
                if "image_paths" in batch:
                    for path_str in batch["image_paths"]:
                        if isinstance(path_str, (list, tuple)):
                            paths = [p for p in path_str if p]
                        else:
                            paths = [p.strip() for p in path_str.split(",") if p.strip()]
                        all_images.append(wandb.Image(Image.open(paths[0]).convert("RGB")))
                # Run generation using image tensors already in the batch.
                # Forward all tensor batch keys except text/label keys — this picks up
                # pixel_values, image_grid_thw, and any other model-specific image keys.
                skip_keys = {self.input_ids_key_name, "attention_mask", "labels", "image_paths"}
                prompt_ids_list = [
                    batch[self.input_ids_key_name][j][:starting_indices[j]].tolist()
                    for j in range(len(starting_indices))
                ]
                max_prompt_len = max(len(p) for p in prompt_ids_list)
                pad_id = getattr(processor, "tokenizer", processor).pad_token_id or 0
                padded = torch.full((len(prompt_ids_list), max_prompt_len), pad_id, dtype=torch.long)
                attn = torch.zeros_like(padded)
                for j, ids in enumerate(prompt_ids_list):
                    padded[j, max_prompt_len - len(ids):] = torch.tensor(ids)
                    attn[j, max_prompt_len - len(ids):] = 1
                gen_inputs = {
                    "input_ids": padded.to(model.device),
                    "attention_mask": attn.to(model.device),
                }
                for k, v in batch.items():
                    if k not in skip_keys and hasattr(v, "to"):
                        gen_inputs[k] = v.to(model.device)
                gen_out = model.generate(**gen_inputs, max_new_tokens=self.eval_max_new_tokens, do_sample=False)
                gen_out = gen_out[:, max_prompt_len:]
                all_outputs.extend(processor.batch_decode(gen_out, skip_special_tokens=True))
            else:
                current_padding_side = processor.padding_side
                processor.padding_side = "left"
                inputs = processor(all_input_texts, return_tensors="pt", padding=True).to(model.device)
                processor.padding_side = current_padding_side
                input_length = inputs['input_ids'].shape[1]
                gen_kwargs = {"max_new_tokens": self.eval_max_new_tokens, "do_sample": False}
                outputs = model.generate(**inputs, **gen_kwargs)
                outputs = outputs[:, input_length:]
                output_texts = processor.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(output_texts)
        for j, values in enumerate(zip(all_input_texts, all_targets, all_rejecteds, all_outputs)):
            input_text, target, rejected, output = values
            img = all_images[j] if all_images else None
            if self.training_kind in ["clf", "sft", "ga", "pre", "npo"]:
                if self.modality == "vlm":
                    self.table.add_data(state.global_step, j, input_text, img, target, output)
                else:
                    self.table.add_data(state.global_step, j, input_text, target, output)
            elif self.training_kind in ["dpo"]:
                if self.modality == "vlm":
                    self.table.add_data(state.global_step, j, input_text, img, target, rejected, output)
                else:
                    self.table.add_data(state.global_step, j, input_text, target, rejected, output)
        wandb.log({"Sample Outputs": self.table})
        return

def get_callback_list(script_args):
    callbacks = [
        SampleLoggingCallback(script_args.training_kind, script_args.modality, script_args.n_eval_output_batches, script_args.eval_max_new_tokens), 
                 ]
    if script_args.training_kind != "ga":
        callbacks.append(StopOnZeroLossCallback())
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


def vlm_clf_preprocess_function(examples, processor, max_length, label2id):
    result = processor.apply_chat_template(
            examples["messages"],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            padding=True,
            return_tensors="pt",
        )
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
            dataset = dataset.map(
                lambda x: vlm_clf_preprocess_function(x, processor, script_args.max_input_length, label2id),
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


class VLMSFTDataCollator:
    """
    Custom data collator for VLM SFT training.

    Reads raw input / output / image columns directly so we never have to store
    complex nested dicts or PIL images through HuggingFace datasets Arrow storage
    (which causes type-inference and nesting bugs).

    Loads PIL images from paths at collation time, builds proper chat-template
    messages, and calls the processor — mirroring what TRL's
    DataCollatorForVisionLanguageModeling does internally.
    """

    def __init__(self, processor, max_length=None):
        self.processor = processor
        self.max_length = max_length

    def _load_images(self, image_value):
        """Return a list of PIL Images from a path, list of paths, or existing PIL images."""
        if isinstance(image_value, str):
            paths = [p.strip() for p in image_value.split(",") if p.strip()]
        elif isinstance(image_value, (list, tuple)):
            paths = list(image_value)
        else:
            return [image_value]
        result = []
        for p in paths:
            if isinstance(p, str):
                result.append(Image.open(p).convert("RGB"))
            else:
                result.append(p)
        return result

    def __call__(self, examples):
        all_messages = []
        all_prompt_messages = []
        all_images = []   # list-of-lists: [[PIL, ...], [PIL, ...], ...]

        for example in examples:
            pil_images = self._load_images(example["image"])

            user_content = [{"type": "image"} for _ in pil_images]
            user_content.append({"type": "text", "text": example["input"]})
            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": example["output"]},
            ]
            prompt_messages = [
                {"role": "user", "content": user_content},
            ]
            prepared = prepare_multimodal_messages(messages, pil_images)
            prepared_prompt = prepare_multimodal_messages(prompt_messages, pil_images)
            all_messages.append(prepared)
            all_prompt_messages.append(prepared_prompt)
            all_images.append(pil_images)

        texts = self.processor.apply_chat_template(
            all_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        if isinstance(texts, str):
            texts = [texts]

        prompt_texts = self.processor.apply_chat_template(
            all_prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]

        processor_kwargs = dict(
            text=texts,
            images=all_images,
            padding=True,
            return_tensors="pt",
            add_special_tokens=False,
        )
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length
            processor_kwargs["truncation"] = True
        output = self.processor(**processor_kwargs)

        # Tokenize prompt-only per example (no padding) to get exact prompt lengths,
        # including any image tokens inserted by the processor.
        prompt_lens = []
        for pt, imgs in zip(prompt_texts, all_images):
            p_out = self.processor(
                text=pt,
                images=imgs,
                return_tensors="pt",
                add_special_tokens=False,
            )
            prompt_lens.append(p_out["input_ids"].shape[1])

        labels = output["input_ids"].clone()
        labels[output["attention_mask"] == 0] = -100
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, :prompt_len] = -100
        output["labels"] = labels
        output["image_paths"] = [example["image"] for example in examples]
        return output


def _vlm_max_length(model):
    """Return the text context length from a VLM model config, or None if not found."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", cfg)
    return getattr(text_cfg, "max_position_embeddings", None)


class VLMGADataCollator(VLMSFTDataCollator):
    """
    Extends VLMSFTDataCollator for Gradient Ascent (GA) training.
    Passes through the `forget` column as a float tensor alongside
    the standard VLM inputs/labels.
    """

    def __call__(self, examples):
        output = super().__call__(examples)
        output["forget"] = torch.tensor(
            [example["forget"] for example in examples], dtype=torch.float
        )
        return output


def get_trl_renamed_train_val_dataset(dataset):
    """
    Rename the columns of the dataset to match the expected format for TRL trainers.
    """
    dataset = drop_column_if_needed(dataset, "prompt")
    dataset = drop_column_if_needed(dataset, "completion")
    po_cols = ["chosen", "rejected"]
    for col in po_cols:        
        dataset = drop_column_if_needed(dataset, col)
    train_dataset = dataset["train"].rename_column("input", "prompt")
    train_dataset = train_dataset.map(lambda x: {"prompt": x["prompt"] + " \n"}, num_proc=1, desc="Adding endline and space after prompt")
    if "output" in dataset["train"].features:
        train_dataset = train_dataset.rename_column("output", "completion")
        train_dataset = train_dataset.map(lambda x: {"completion": "Output: " + x["completion"]}, num_proc=1, desc="Adding Output before completion")
    else:
        # make a dummy completion column with empty strings. Seems to work for pretraining. 
        train_dataset = train_dataset.add_column("completion", [""] * len(train_dataset))
    for col in po_cols:
        if col in dataset["train"].features:
            train_dataset = train_dataset.map(lambda x: {col: "Output: " + x[col]}, num_proc=1, desc=f"Adding Output before {col}")
    if "validation" in dataset:
        validation_dataset = dataset["validation"].rename_column("input", "prompt")
        validation_dataset = validation_dataset.map(lambda x: {"prompt": x["prompt"] + " \n"}, num_proc=1, desc="Adding endline and space after prompt")
        if "output" in dataset["validation"].features:
            validation_dataset = validation_dataset.rename_column("output", "completion")
            validation_dataset = validation_dataset.map(lambda x: {"completion": "Output: " + x["completion"]}, num_proc=1, desc="Adding Output before completion")
        else:
            validation_dataset = validation_dataset.add_column("completion", [""] * len(validation_dataset))
        for col in po_cols:
            if col in dataset["validation"].features:
                validation_dataset = validation_dataset.map(lambda x: {col: "Output: " + x[col]}, num_proc=1, desc=f"Adding Output before {col}")
    else:
        validation_dataset = None
    return train_dataset, validation_dataset


def get_pre_trainer(script_args, training_args, dataset, model, processor, peft_config):
    callbacks = get_callback_list(script_args)
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
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
    else:
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"] if "validation" in dataset else None
        training_args.remove_unused_columns = False
        data_collator = VLMSFTDataCollator(processor, max_length=_vlm_max_length(model))
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=data_collator,
            args=training_args,
            callbacks=callbacks,
        )
    return trainer, dataset


def get_sft_trainer(script_args, training_args, dataset, model, processor, peft_config):
    callbacks = get_callback_list(script_args)
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            peft_config=peft_config,
            processing_class=processor,
            args=training_args,
            callbacks=callbacks,
        )
    else:
        # Use a custom collator that reads raw input/output/image columns directly,
        # bypassing HF datasets' Arrow type inference for nested PIL/dict structures.
        # remove_unused_columns must be False so the trainer keeps our raw text columns.
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"] if "validation" in dataset else None
        training_args.remove_unused_columns = False
        data_collator = VLMSFTDataCollator(processor, max_length=_vlm_max_length(model))
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            peft_config=peft_config,
            processing_class=processor,
            data_collator=data_collator,
            args=training_args,
            callbacks=callbacks,
        )
    return trainer, dataset


def get_dpo_trainer(script_args, training_args, dataset, model, processor, peft_config):
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
    else:
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"] if "validation" in dataset else None
        training_args.remove_unused_columns = False
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


def get_ga_trainer(script_args, training_args, dataset, model, processor, peft_config):
    callbacks = get_callback_list(script_args)
    training_args.remove_unused_columns = False
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
        data_collator = ga_data_collator
    else:
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"] if "validation" in dataset else None
        data_collator = VLMGADataCollator(processor, max_length=_vlm_max_length(model))
    trainer = GATrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        peft_config=peft_config,
        processing_class=processor,
        args=training_args,
        callbacks=callbacks,
        data_collator=data_collator,
    )
    return trainer, dataset

def ga_data_collator(batch):
    #ref: https://github.com/huggingface/trl/blob/c26b375ca3dd47e9cd9fdfd820e89bb4af669186/trl/trainer/sft_trainer.py#L118
    sft_collator = DataCollatorForLanguageModeling(pad_token_id=0) #TODO: change; some models have no pad tokens, there is a pad_free argument but I need to verify that this will still work correctly if I use it 
    batch_data = sft_collator(batch)
    batch_data["forget"] = torch.tensor([example["forget"] for example in batch], dtype=torch.float)
    return batch_data


def get_npo_trainer(script_args, training_args, dataset, model, processor, peft_config):
    callbacks = get_callback_list(script_args)
    if script_args.modality == "lm":
        train_dataset, validation_dataset = get_trl_renamed_train_val_dataset(dataset)
        trainer = NPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=processor,
            callbacks=callbacks,
            peft_config=peft_config,
        )
    else:
        train_dataset = dataset["train"]
        validation_dataset = dataset["validation"] if "validation" in dataset else None
        training_args.remove_unused_columns = False
        data_collator = VLMSFTDataCollator(processor, max_length=_vlm_max_length(model))
        trainer = NPOTrainer(
            model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            processing_class=processor,
            callbacks=callbacks,
            peft_config=peft_config,
            data_collator=data_collator,
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
        elif script_args.training_kind == "ga":
            trainer, dataset = get_ga_trainer(script_args, training_args, dataset, model, processor, peft_config)
        elif script_args.training_kind == "npo":
            trainer, dataset = get_npo_trainer(script_args, training_args, dataset, model, processor, peft_config)
        else:
            raise ValueError(f"Training kind {script_args.training_kind} not supported")
    return trainer, dataset
