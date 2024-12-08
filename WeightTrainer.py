import torch
from transformers import Trainer
from torch import nn

class WeightTrainer(Trainer):
    """
        WeightTrainer is a custom Trainer subclass for the HuggingFace Transformer library used to compute custom loss. 
        By default all models return loss in the first element.  WeightTrainer provides class weights to PyTorch's nn.CrossEntropy.
        Custom weights must be calculated first, e.g. sklearn's class_weight.
        Note: this class does NOT handle label smoothing. Enhancements would be required.
    """

    def __init__(self, class_weights, *args, **kwargs):
        # class_weights must be torch.float
        self.class_weights = class_weights
        super().__init__(*args, **kwargs)


    def compute_loss(self,
            model,
            inputs,
            return_outputs=False,
            num_items_in_batch = None
    ):

        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs['logits']
        # calc negative log-likelihood of predicted class distribution
        criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
                reduction='mean',       # default: mean over batch
                label_smoothing=0.0,    # default: no label smoothing
                                          
        )
        loss = criterion(logits, labels)

        return (loss, outputs) if return_outputs else loss