import torch.nn as nn
import torch
from trl import SFTTrainer

class GATrainer(SFTTrainer):
    """
    Trainer for Gradient Ascent (GA) method.

    This class is a wrapper around the [`~trl.SFTTrainer`] class and inherits all of its attributes and methods. 
    The only difference is that the loss returned is the negative of the loss computed by the parent class,
    effectively turning the gradient descent into gradient ascent.
    """
    
    def __init__(self, *args, epsilon: float = 0.1, ignore_index: int = -100, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.ignore_index = ignore_index

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        #ref: https://github.com/huggingface/transformers/blob/514de24abfd4416aeba6a6455ad5920f57f3567d/src/transformers/trainer.py#L2759C30-L2759C63
        outputs = model(**inputs) 
        labels = inputs.get("labels")
        logits = outputs.get("logits") 

        #ref: https://github.com/huggingface/transformers/blob/514de24abfd4416aeba6a6455ad5920f57f3567d/src/transformers/trainer_pt_utils.py#L497
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        log_probs = -nn.functional.log_softmax(logits, dim=-1) 
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1) 

        padding_mask = labels.eq(self.ignore_index) 
        labels = torch.clamp(labels, min=0)

        nll_loss = log_probs.gather(dim=-1, index=labels) 
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32) 

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        #changes to the HF code begin below
        #if the example consists only of pad tokens (where the numerator will always be zero), we clamp denominator to 1 to avoid division by zero
        num_active_per_example = (padding_mask.size(1) - padding_mask.long().sum(dim=1)).clamp(min=1) 
        nll_loss = nll_loss.sum(dim=1) / num_active_per_example.squeeze(-1)
        smoothed_loss = smoothed_loss.sum(dim=1) / (num_active_per_example.squeeze(-1) * log_probs.shape[-1])
        
        per_example_loss = (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss
        forget_mask = inputs.get("forget").to(per_example_loss.device).bool()
        per_example_loss = torch.where(forget_mask, -per_example_loss, per_example_loss)

        loss = per_example_loss.mean() 
        return (loss, outputs) if return_outputs else loss

