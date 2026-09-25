import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        net = resnet18(weights=weights)
        self.features = nn.Sequential(*list(net.children())[:-1])
        self.out_features = 512

    def forward(self, x):
        feat = self.features(x)
        return feat.view(feat.size(0), -1)

def freeze_batchnorm_stats(model):
    """
    Batch-normalization policy: Freeze all running means and variances
    at ImageNet pretrained values while leaving gamma and beta trainable.
    """
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.eval()
            m.track_running_stats = False
