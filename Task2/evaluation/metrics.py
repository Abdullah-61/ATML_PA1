import torch
from sklearn.metrics import accuracy_score, f1_score

@torch.no_grad()
def evaluate_dataset(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    all_preds, all_targets = [], []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        logits = classifier(feats)
        preds = logits.argmax(dim=1).cpu()
        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(labels.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    return acc, macro_f1
