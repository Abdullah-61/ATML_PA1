import torch.nn as nn
from shared.pacs_protocol import NUM_CLASSES

class LinearClassifierHead(nn.Module):
    def __init__(self, in_features=512, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)
