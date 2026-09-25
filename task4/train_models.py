import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from task4.dataset_protocol import set_seed, SEED, get_cifar10_splits
from task4.models.cifar_resnet import CIFARResNet18
from task4.methods.proser import PROSERModel, PROSERLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_accuracy(model, loader, device, num_known=10):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = model(imgs)
            preds = logits[:, :num_known].argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return (correct / total) * 100.0

def train_standard_classifier(name="vanilla", use_randaugment=False):
    set_seed(SEED)
    splits = get_cifar10_splits("./data", use_randaugment=use_randaugment)
    train_loader = DataLoader(splits["train"], batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(splits["val"], batch_size=128, shuffle=False, num_workers=2)

    model = CIFARResNet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    best_val_acc = -1.0
    best_state = None

    print(f"\n--- Training {name.upper()} (100 Epochs, lr=0.1, Cosine Decay) ---")
    for epoch in range(1, 101):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits, _ = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        scheduler.step()
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 20 == 0 or epoch == 100:
            avg_loss = total_loss / len(splits["train"])
            print(f"Epoch {epoch:03d} | Train Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    ckpt_path = f"task4/results/checkpoints/{name}_resnet18.pth"
    torch.save(best_state, ckpt_path)
    print(f"Saved {name} model checkpoint: {ckpt_path}")
    return ckpt_path

def train_proser():
    set_seed(SEED)
    vanilla_ckpt = "task4/results/checkpoints/vanilla_resnet18.pth"
    assert os.path.exists(vanilla_ckpt), "Vanilla checkpoint required before training PROSER."

    base_model = CIFARResNet18(num_classes=10).to(DEVICE)
    base_model.load_state_dict(torch.load(vanilla_ckpt, map_location=DEVICE))
    proser_model = PROSERModel(base_model, num_known=10, num_dummies=5).to(DEVICE)

    splits = get_cifar10_splits("./data", use_randaugment=False)
    train_loader = DataLoader(splits["train"], batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(splits["val"], batch_size=128, shuffle=False, num_workers=2)

    criterion = PROSERLoss(num_known=10, num_dummies=5, beta=1.0, gamma=0.1)
    optimizer = torch.optim.SGD(proser_model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    best_val_acc = -1.0
    best_state = None

    print("\n--- Fine-tuning PROSER (50 Epochs, lr=1e-3, 5 Dummy Classifiers, Manifold Mixup at Layer2) ---")
    for epoch in range(1, 51):
        proser_model.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            b = imgs.size(0)
            if b < 4:
                continue
            b_half = b // 2

            # First half: Classifier-placeholder loss
            x_cp, y_cp = imgs[:b_half], labels[:b_half]
            logits_cp, _ = proser_model(x_cp)

            # Second half: Data-placeholder loss via manifold mixup after layer2
            x_dp, y_dp = imgs[b_half:], labels[b_half:]
            n_dp = x_dp.size(0)
            perm = torch.randperm(n_dp, device=DEVICE)
            diff_mask = y_dp != y_dp[perm]

            logits_dp = None
            if diff_mask.sum() > 0:
                h_pre = proser_model.forward_pre_layer2(x_dp)
                h1 = h_pre[diff_mask]
                h2 = h_pre[perm][diff_mask]

                # Beta(2, 2) sampling
                lam = np.random.beta(2.0, 2.0)
                h_mixed = lam * h1 + (1.0 - lam) * h2
                logits_dp, _ = proser_model.forward_post_layer2(h_mixed)

            optimizer.zero_grad()
            loss, _ = criterion(logits_cp, y_cp, logits_dp)
            loss.backward()
            optimizer.step()

        scheduler.step()
        val_acc = evaluate_accuracy(proser_model, val_loader, DEVICE, num_known=10)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(proser_model.state_dict())

        if epoch % 10 == 0 or epoch == 50:
            print(f"PROSER Epoch {epoch:02d} | Val CSA: {val_acc:.2f}% (Best: {best_val_acc:.2f}%)")

    ckpt_path = "task4/results/checkpoints/proser_resnet18.pth"
    torch.save(best_state, ckpt_path)
    print(f"Saved PROSER model checkpoint: {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vanilla", "gcsc", "proser"], required=True)
    args = parser.parse_args()

    if args.model == "vanilla":
        train_standard_classifier("vanilla", use_randaugment=False)
    elif args.model == "gcsc":
        train_standard_classifier("gcsc", use_randaugment=True)
    elif args.model == "proser":
        train_proser()
