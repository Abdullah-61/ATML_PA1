import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights, vit_b_16, ViT_B_16_Weights
import open_clip

class ViTClassTokenExtractor(nn.Module):
    def __init__(self, vit_module):
        super().__init__()
        self.vit = vit_module

    def forward(self, x):
        x = self.vit._process_input(x)
        b = x.shape[0]
        batch_class_token = self.vit.class_token.expand(b, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.vit.encoder(x)
        return x[:, 0]  # [B, 768] CLS token

def load_backbones(device):
    # 1. ResNet-50 (IMAGENET1K_V2)
    r50_weights = ResNet50_Weights.IMAGENET1K_V2
    r50 = resnet50(weights=r50_weights)
    r50.fc = nn.Identity()  # GAP output (2048-d)
    r50_norm = transforms.Normalize(mean=r50_weights.transforms().mean, std=r50_weights.transforms().std)

    # 2. ViT-B/16 (IMAGENET1K_V1)
    vit_weights = ViT_B_16_Weights.IMAGENET1K_V1
    vit_raw = vit_b_16(weights=vit_weights)
    vit = ViTClassTokenExtractor(vit_raw)  # CLS token (768-d)
    vit_norm = transforms.Normalize(mean=vit_weights.transforms().mean, std=vit_weights.transforms().std)

    # 3. OpenCLIP ViT-B/32 (pretrained='openai')
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    clip_norm = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                     std=(0.26862954, 0.26130258, 0.27577711))

    # Freeze all backbones
    for m in [r50, vit, clip_model]:
        m.to(device)
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    backbones = {"resnet50": r50, "vit_b16": vit, "clip": clip_model}
    norms = {"resnet50": r50_norm, "vit_b16": vit_norm, "clip": clip_norm}
    return backbones, norms
