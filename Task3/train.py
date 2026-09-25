import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from shared.pacs_protocol import set_seed, SEED, SOURCE_DOMAINS
from shared.pacs import build_or_load_splits
from task3.models.backbone import ResNet18Backbone, freeze_batchnorm_stats
from task3.models.classifier_head import LinearClassifierHead
from task3.methods.dan_dg import DANDGLoss
from task3.methods.sam import SAM
from task3.evaluation.domain_metrics import evaluate_domain

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_task3_model(method="dan_dg", lambda_mmd=1.0, rho=0.05):
    set_seed(SEED)
    datasets = build_or_load_splits("./pacs_data")
    
    source_train_loaders = {
        d: DataLoader(datasets["source_train"][d], batch_size=8, shuffle=True, drop_last=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }
    source_val_loaders = {
        d: DataLoader(datasets["source_val"][d], batch_size=64, shuffle=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }

    backbone = ResNet18Backbone().to(DEVICE)
    classifier = LinearClassifierHead(in_features=backbone.out_features).to(DEVICE)
    params = list(backbone.parameters()) + list(classifier.parameters())

    max_source_steps = max(len(loader) for loader in source_train_loaders.values())
    total_epochs = 30
    patience = 5
    best_val_macro_f1 = -1.0
    best_weights = None
    no_improve_epochs = 0
    history = {"epochs": [], "loss_cls": [], "loss_mmd": [], "mean_val_f1": []}

    if method == "dan_dg":
        criterion = DANDGLoss(lambda_mmd=lambda_mmd)
        optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    elif method == "sam":
        criterion = nn.CrossEntropyLoss()
        optimizer = SAM(params, torch.optim.AdamW, rho=rho, lr=1e-4, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown method: {method}")

    print("========================================================")
    print(f"Starting Task 3 Training: {method.upper()} (Max {total_epochs} epochs, {max_source_steps} steps/epoch)")
    print("========================================================")

    for epoch in range(1, total_epochs + 1):
        backbone.train()
        classifier.train()
        freeze_batchnorm_stats(backbone)

        iter_sources = {d: iter(source_train_loaders[d]) for d in SOURCE_DOMAINS}
        ep_cls_loss, ep_mmd_loss = 0.0, 0.0

        for step in range(max_source_steps):
            domain_imgs, domain_labels = {}, {}
            cat_imgs, cat_labels = [], []
            for d in SOURCE_DOMAINS:
                try:
                    imgs, labels, _ = next(iter_sources[d])
                except StopIteration:
                    iter_sources[d] = iter(source_train_loaders[d])
                    imgs, labels, _ = next(iter_sources[d])
                domain_imgs[d] = imgs.to(DEVICE)
                domain_labels[d] = labels.to(DEVICE)
                cat_imgs.append(domain_imgs[d])
                cat_labels.append(domain_labels[d])

            batch_imgs = torch.cat(cat_imgs, dim=0)
            batch_labels = torch.cat(cat_labels, dim=0)

            if method == "dan_dg":
                optimizer.zero_grad()
                feats = backbone(batch_imgs)
                logits = classifier(feats)
                
                feats_by_dom = {d: feats[i*8:(i+1)*8] for i, d in enumerate(SOURCE_DOMAINS)}
                loss, log_dict = criterion(logits, batch_labels, feats_by_dom)
                loss.backward()
                optimizer.step()
                
                freeze_batchnorm_stats(backbone)
                ep_cls_loss += log_dict["loss_cls"]
                ep_mmd_loss += log_dict["loss_mmd"]

            elif method == "sam":
                # Forward / Backward pass 1
                feats = backbone(batch_imgs)
                logits = classifier(feats)
                loss1 = criterion(logits, batch_labels)
                loss1.backward()
                optimizer.first_step(zero_grad=True)

                # Forward / Backward pass 2
                freeze_batchnorm_stats(backbone)
                feats2 = backbone(batch_imgs)
                logits2 = classifier(feats2)
                loss2 = criterion(logits2, batch_labels)
                loss2.backward()
                optimizer.second_step(zero_grad=True)

                freeze_batchnorm_stats(backbone)
                ep_cls_loss += loss1.item()

        val_f1s = []
        for d in SOURCE_DOMAINS:
            _, f1 = evaluate_domain(backbone, classifier, source_val_loaders[d], DEVICE)
            val_f1s.append(f1)
        mean_val_f1 = float(np.mean(val_f1s))

        history["epochs"].append(epoch)
        history["loss_cls"].append(ep_cls_loss / max_source_steps)
        history["loss_mmd"].append(ep_mmd_loss / max_source_steps if method == "dan_dg" else 0.0)
        history["mean_val_f1"].append(mean_val_f1)

        print(f"Epoch {epoch:02d} | Cls Loss: {ep_cls_loss/max_source_steps:.4f} | MMD Loss: {ep_mmd_loss/max_source_steps:.4f} | Source Val Mean-F1: {mean_val_f1:.2f}% | Worst: {min(val_f1s):.2f}%")

        if mean_val_f1 > best_val_macro_f1:
            best_val_macro_f1 = mean_val_f1
            best_weights = {
                "backbone": copy.deepcopy(backbone.state_dict()),
                "classifier": copy.deepcopy(classifier.state_dict()),
                "epoch": epoch,
                "best_val_macro_f1": best_val_macro_f1
            }
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch} (Best Mean Source-Val Macro-F1: {best_val_macro_f1:.2f}%)")
                break

    ckpt_name = f"{method}_model.pth" if (lambda_mmd == 1.0 and rho == 0.05) else f"{method}_rho_{rho}.pth"
    ckpt_path = os.path.join("task3/results/checkpoints", ckpt_name)
    torch.save(best_weights, ckpt_path)

    curve_path = os.path.join("task3/results/curves", f"{method}_training_curves.json")
    with open(curve_path, "w") as fp:
        json.dump(history, fp, indent=2)

    print(f"Saved checkpoint: {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["dan_dg", "sam"], required=True)
    parser.add_argument("--lambda_mmd", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=0.05)
    args = parser.parse_args()
    train_task3_model(method=args.method, lambda_mmd=args.lambda_mmd, rho=args.rho)
