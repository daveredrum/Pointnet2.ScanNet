import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets, weights):
        assert inputs.size(0) == targets.size(0) == weights.size(0)
        
        loss = F.cross_entropy(input=inputs, target=targets, reduction="none", ignore_index=self.ignore_index)
        loss = torch.mean(loss * weights)

        return loss