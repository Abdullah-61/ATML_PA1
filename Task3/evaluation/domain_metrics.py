import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from shared.pacs_protocol import SEED, SOURCE_DOMAINS

@torch.no_grad()
def evaluate_domain(backbone, classifier, loader, device):
    backbone.eval()
    classifier.eval()
    all_preds, all_targets = [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)
        preds = classifier(feats).argmax(dim=1).cpu()
        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(labels.numpy().tolist())
    acc = accuracy_score(all_targets, all_preds) * 100.0
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100.0
    return acc, f1

@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    feats_list = []
    for imgs, _, _ in loader:
        imgs = imgs.to(device)
        feats_list.append(backbone(imgs).cpu())
    return torch.cat(feats_list, dim=0).numpy()

def compute_source_domain_separability(domain_features_dict):
    min_count = min(len(f) for f in domain_features_dict.values())
    rng = np.random.RandomState(SEED)
    
    x_list, y_list = [], []
    for idx, d in enumerate(SOURCE_DOMAINS):
        feats = domain_features_dict[d]
        chosen = rng.choice(len(feats), size=min_count, replace=False)
        x_list.append(feats[chosen])
        y_list.append(np.full(min_count, idx, dtype=int))
        
    x = np.concatenate(x_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )
    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(x_train, y_train)
    return float(accuracy_score(y_test, clf.predict(x_test)) * 100.0)

def measure_sharpness_proxy(backbone, classifier, datasets, device, rho=0.05):
    backbone.eval()
    classifier.eval()
    ce = torch.nn.CrossEntropyLoss()
    
    rng = np.random.RandomState(SEED)
    diag_imgs, diag_labels = [], []
    
    for d in SOURCE_DOMAINS:
        val_set = datasets["source_val"][d]
        indices = rng.choice(len(val_set), size=32, replace=False)
        for idx in indices:
            img, label, _ = val_set[idx]
            diag_imgs.append(img)
            diag_labels.append(label)
            
    imgs = torch.stack(diag_imgs, dim=0).to(device)
    labels = torch.tensor(diag_labels, dtype=torch.long, device=device)

    backbone.zero_grad()
    classifier.zero_grad()
    feats = backbone(imgs)
    logits = classifier(feats)
    base_loss = ce(logits, labels)
    base_loss.backward()

    params = [p for p in list(backbone.parameters()) + list(classifier.parameters()) if p.grad is not None]
    grad_norm = torch.norm(torch.stack([p.grad.norm(2) for p in params]), 2)
    
    perturbations = {}
    with torch.no_grad():
        scale = rho / (grad_norm + 1e-12)
        for p in params:
            e = p.grad * scale
            p.add_(e)
            perturbations[p] = e

    with torch.no_grad():
        pert_feats = backbone(imgs)
        pert_logits = classifier(pert_feats)
        pert_loss = ce(pert_logits, labels)

    with torch.no_grad():
        for p in params:
            p.sub_(perturbations[p])

    return float((pert_loss - base_loss).item())
