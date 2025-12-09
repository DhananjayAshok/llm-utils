from trl import DPOTrainer
from trl.trainer.dpo_trainer import *
import torch
from torch import nn
import torch.nn.functional as F
from trl import SFTTrainer
import copy
from trl.trainer.sft_trainer import entropy_from_logits, PeftType


def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
    # get the sum loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels).sum(dim=-1)
    return loss

class NPOTrainer(SFTTrainer):
    """
    Trainer for Negative Preference Optimization (NPO) method.

    This class is a wrapper around the [`trl.SFTTrainer`] class and inherits all of its attributes and methods. It is largely based off the trl SFT code
    """

    _tag_names = ["trl", "npo"]

    def __init__(self, *args, beta=0.1, **kwargs):
        model = args[0]
        self.oracle_model = copy.deepcopy(model).eval()
        self.beta = beta
        super().__init__(*args, **kwargs)

    def super_compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        #ref: https://github.com/huggingface/transformers/blob/514de24abfd4416aeba6a6455ad5920f57f3567d/src/transformers/trainer.py#L2759C30-L2759C63
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        # forward pass
        outputs = model(**inputs)
        forget_loss_current = get_batch_loss(outputs.logits, labels) 
        logits = outputs.get("logits")
        with torch.no_grad():
            forget_outputs_oracle = self.oracle_model(**inputs)
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = get_batch_loss(forget_logits_oracle, labels)
        neg_log_ratios = forget_loss_current - forget_loss_oracle
        loss = -F.logsigmoid(self.beta * neg_log_ratios).mean() * 2 / self.beta 
        return (loss, outputs) if return_outputs else loss

    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        ref: https://github.com/huggingface/trl/blob/23da61d5894f2886764bc62b3b76fd4797d401f1/trl/trainer/sft_trainer.py#L1129
        """
        mode = "train" if self.model.training else "eval"
        
        # Set aside labels as it will be dropped by super().compute_loss() if a custom `compute_loss_func` is used.
        # This can be removed when this issue is fixed.
        # When using CP or SP, labels are pre-shifted, we must use shift_labels instead.
        labels = inputs["labels"] if "shift_labels" not in inputs else None

        # If not set, defaults from model config and may warn since cache isn't compatible with gradient checkpointing
        inputs["use_cache"] = False
        # Request token accuracy from Liger kernel and set token scaling if using DFT loss
        if self.args.use_liger_kernel:
            inputs["return_token_accuracy"] = True
            inputs["use_token_scaling"] = self.args.loss_type == "dft"

        (loss, outputs) = self.super_compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
        )

        # Compute entropy
        if not self.args.use_liger_kernel:  # liger doesn't return logits
            with torch.no_grad():
                per_token_entropy = entropy_from_logits(outputs.logits)
                # When using Prompt Tuning, skip the virtual tokens in logits before entropy computation, since they
                # do not correspond to actual input tokens.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    per_token_entropy = per_token_entropy[:, self.num_virtual_tokens :]
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]
                    entropy = torch.sum(per_token_entropy * attention_mask) / attention_mask.sum()
                elif "position_ids" in inputs:
                    entropy = torch.mean(per_token_entropy)
                else:
                    raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
                entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

        if mode == "train":
            # When using padding-free, the attention_mask is not present in the inputs, instead we have cu_seq_lens_q,
            # cu_seq_lens_k, and max_length_k, max_length_q and position_ids.
            if "attention_mask" in inputs:
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
            elif "position_ids" in inputs:
                local_num_tokens = torch.tensor(inputs["position_ids"].size(1), device=inputs["position_ids"].device)
                num_tokens_in_batch = self.accelerator.gather_for_metrics(local_num_tokens).sum().item()
            else:
                raise ValueError("Expected 'attention_mask' or 'position_ids' in inputs.")
            self._total_train_tokens += num_tokens_in_batch
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        if self.args.use_liger_kernel:
            token_accuracy = self.accelerator.gather_for_metrics(outputs.token_accuracy).mean().item()
            self._metrics[mode]["mean_token_accuracy"].append(token_accuracy)
        else:
            # Compute accuracy from logits using argmax (traditional method)
            with torch.no_grad():
                if "shift_labels" in inputs:
                    # When using CP or SP, labels are pre-shifted. We must use these (and cannot manually shift) because:
                    # - The first discarded token from inputs["labels"] actually belongs to process n-1
                    # - The last logits require the label from process n+1
                    shift_logits = outputs.logits.contiguous()
                    shift_labels = inputs["shift_labels"]
                else:
                    shift_logits = outputs.logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()

                # Prompt Tuning and P-Tuning output logits for virtual tokens but Prefix-Tuning does not.
                if (
                    self.num_virtual_tokens > 0
                    and model.peft_config[model.active_adapter].peft_type != PeftType.PREFIX_TUNING
                ):
                    shift_logits = shift_logits[:, self.num_virtual_tokens :, :]

                # Get predictions
                predictions = shift_logits.argmax(dim=-1)

                # Create mask for non-padding tokens (assuming ignore_index is -100)
                mask = shift_labels != -100

                # Calculate accuracy only on non-padding tokens
                correct_predictions = (predictions == shift_labels) & mask
                total_tokens = mask.sum()
                correct_tokens = correct_predictions.sum()

                # Gather the correct_tokens and total_tokens across all processes
                correct_tokens = self.accelerator.gather_for_metrics(correct_tokens)
                total_tokens = self.accelerator.gather_for_metrics(total_tokens)

                # Compute the mean token accuracy and log it
                total_sum = total_tokens.sum()
                accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
                self._metrics[mode]["mean_token_accuracy"].append(accuracy)

        # Log auxiliary loss if enabled (applies to both Liger and non-Liger)
        if self.aux_loss_enabled:
            aux_loss = outputs.aux_loss
            aux_loss = self.accelerator.gather_for_metrics(aux_loss).mean().item()
            self._metrics[mode]["aux_loss"].append(aux_loss)

        return (loss, outputs) if return_outputs else loss
