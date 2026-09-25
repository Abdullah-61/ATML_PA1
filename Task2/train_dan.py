import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import torch
import numpy as np
from torch.utils.data import DataLoader
from shared.pacs_protocol import set_seed, SEED, SOURCE_DOMAINS
from shared.pacs import build_or_load_splits
from task2.models.backbone import ResNet18Backbone, freeze_batchnorm_stats
from task2.models.classifier_head import LinearClassifierHead
from task2.methods.dan import DANLoss
from task2.evaluation.metrics import evaluate_dataset

def train_dan(data_root="./pacs_data", checkpoint_save_dir="task2/results/checkpoints"):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution device: {device}")

    datasets = build_or_load_splits(data_root)

    # 8 examples per source domain (24 total source per iteration)
    source_train_loaders = {
        d: DataLoader(datasets["source_train"][d], batch_size=8, shuffle=True, drop_last=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }
    # 24 unlabeled target examples per iteration
    target_adapt_loader = DataLoader(datasets["target_adapt"], batch_size=24, shuffle=True, drop_last=False, num_workers=2)

    source_val_loaders = {
        d: DataLoader(datasets["source_val"][d], batch_size=64, shuffle=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }

    # ResNet-18 + Linear Classifier Head
    backbone = ResNet18Backbone().to(device)
    classifier = LinearClassifierHead(in_features=backbone.out_features).to(device)

    criterion = DANLoss(lambda_mmd=1.0)
    params = list(backbone.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)

    max_source_steps = max(len(loader) for loader in source_train_loaders.values())
    if max_source_steps == 0:
        max_source_steps = 1
    patience = 5
    best_val_macro_f1 = -1.0
    best_weights = None
    no_improve_epochs = 0

    print(f"Starting DAN Training (Max 30 source epochs, {max_source_steps} steps/epoch, lambda_mmd=1.0)...")

    for epoch in range(1, 31):
        backbone.train()
        classifier.train()
        freeze_batchnorm_stats(backbone)

        iter_sources = {d: iter(source_train_loaders[d]) for d in SOURCE_DOMAINS}
        iter_target = iter(target_adapt_loader)

        running_cls_loss = 0.0
        running_mmd_loss = 0.0

        for step in range(max_source_steps):
            s_imgs, s_labels = [], []
            for d in SOURCE_DOMAINS:
                try:
                    imgs, labels, _ = next(iter_sources[d])
                except StopIteration:
                    iter_sources[d] = iter(source_train_loaders[d])
                    imgs, labels, _ = next(iter_sources[d])
                s_imgs.append(imgs)
                s_labels.append(labels)

            try:
                t_imgs, _, _ = next(iter_target)
            except StopIteration:
                iter_target = iter(target_adapt_loader)
                t_imgs, _, _ = next(iter_target)

            s_imgs = torch.cat(s_imgs, dim=0).to(device)
            s_labels = torch.cat(s_labels, dim=0).to(device)
            t_imgs = t_imgs.to(device)

            optimizer.zero_grad()

            # Extract 512-d features immediately before the linear classifier
            s_feats = backbone(s_imgs)
            t_feats = backbone(t_imgs)
            s_logits = classifier(s_feats)

            loss, log_dict = criterion(s_logits, s_labels, s_feats, t_feats)
            loss.backward()
            optimizer.step()

            freeze_batchnorm_stats(backbone)
            running_cls_loss += log_dict["loss_cls"]
            running_mmd_loss += log_dict["loss_mmd"]

        # Validation across 3 source domains
        val_f1_list = []
        for d in SOURCE_DOMAINS:
            _, val_f1 = evaluate_dataset(backbone, classifier, source_val_loaders[d], device)
            val_f1_list.append(val_f1)

        mean_val_f1 = float(np.mean(val_f1_list))
        avg_cls = running_cls_loss / max_source_steps
        avg_mmd = running_mmd_loss / max_source_steps
        print(f"Epoch {epoch:02d} | Cls Loss: {avg_cls:.4f} | MMD Loss: {avg_mmd:.4f} | Source Val Macro-F1: {mean_val_f1:.2f}% | Per-domain: {[round(x, 2) for x in val_f1_list]}")

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
                print(f"Early stopping triggered after {patience} non-improving epochs at epoch {epoch}.")
                break

    os.makedirs(checkpoint_save_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_save_dir, "dan_model.pth")
    torch.save(best_weights, ckpt_path)
    print(f"Best DAN checkpoint saved to: {ckpt_path} (Validation Macro-F1: {best_val_macro_f1:.2f}%)")
    return ckpt_path

if __name__ == "__main__":
    train_dan()
