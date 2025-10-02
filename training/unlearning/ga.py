from trl import SFTTrainer

class GATrainer(SFTTrainer):
    """
    Trainer for Gradient Ascent (GA) method.

    This class is a wrapper around the [`~trl.SFTTrainer`] class and inherits all of its attributes and methods. 
    The only difference is that the loss returned is the negative of the loss computed by the parent class,
    effectively turning the gradient descent into gradient ascent.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute training loss and additionally compute token accuracies
        """
        loss_return = super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)
        if return_outputs:
            loss, outputs = loss_return
        else:
            loss = loss_return
        loss = -loss # TODO: I think you might want to enable retention in GA by only flipping the loss for some instances. I think inputs.get will be useful. 
        return (loss, outputs) if return_outputs else loss