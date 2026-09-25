import torch
import torch.nn as nn

class SourceOnlyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, source_logits, source_labels, target_features=None):
        loss = self.ce(source_logits, source_labels)
        return loss, {"loss_cls": loss.item(), "loss_transfer": 0.0}
