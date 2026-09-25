import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from shared.pacs_protocol import SEED, SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES
from shared.pacs import build_or_load_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "task2/results/checkpoints"
RESULTS_DIR = "task2/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

METHODS = {
    "Source-Only": os.path.join(CHECKPOINT_DIR, "source_only_erm.pth"),
    "DAN": os.path.join(CHECKPOINT_DIR, "dan_model.pth"),
    "DANN": os.path.join(CHECKPOINT_DIR, "dann_model.pth"),
    "CDAN": os.path.join(CHECKPOINT_DIR, "cdan_model.pth")
}

@torch.no_grad()
def extract_features_and_preds(backbone, classifier, loader):
    backbone.eval()
    classifier.eval()
    features, logits_list, targets = [], [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(DEVICE)
        feats = backbone(imgs)
        logits = classifier(feats)
        features.append(feats.cpu())
        logits_list.append(logits.cpu())
        targets.append(labels.cpu())
    features = torch.cat(features, dim=0).numpy()
    logits_list = torch.cat(logits_list, dim=0).numpy()
    targets = torch.cat(targets, dim=0).numpy()
    preds = np.argmax(logits_list, axis=1)
    return features, preds, targets

def compute_domain_separability(source_feats, target_feats):
    # Subsample target features to exactly match source count
    n_source = len(source_feats)
    rng = np.random.RandomState(SEED)
    if len(target_feats) > n_source:
        chosen_indices = rng.choice(len(target_feats), size=n_source, replace=False)
        target_subset = target_feats[chosen_indices]
    else:
        chosen_indices = rng.choice(n_source, size=len(target_feats), replace=False)
        source_feats = source_feats[chosen_indices]
        target_subset = target_feats

    x = np.concatenate([source_feats, target_subset], axis=0)
    y = np.concatenate([np.zeros(len(source_feats)), np.ones(len(target_subset))], axis=0)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=SEED, stratify=y
    )

    clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)
    clf.fit(x_train, y_train)
    preds = clf.predict(x_test)
    return accuracy_score(y_test, preds) * 100.0

def run_diagnostic():
    print(f"Running Unified Diagnostic across methods on {DEVICE}...")
    datasets = build_or_load_splits("./pacs_data")

    # 1. Prepare data loaders
    source_val_loaders = {
        d: DataLoader(datasets["source_val"][d], batch_size=64, shuffle=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }
    target_loader = DataLoader(datasets["target_eval"], batch_size=64, shuffle=False, num_workers=2)

    summary_records = []
    per_class_records = {}
    confusion_matrices = {}

    baseline_target_acc = None

    for name, ckpt_path in METHODS.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: Checkpoint not found at {ckpt_path}")
            continue

        backbone = ResNet18Backbone().to(DEVICE)
        classifier = LinearClassifierHead(in_features=backbone.out_features).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone"])
        classifier.load_state_dict(ckpt["classifier"])

        # Evaluate Source Validation
        source_f1s = []
        source_feats_list = []
        for d in SOURCE_DOMAINS:
            f_val, p_val, y_val = extract_features_and_preds(backbone, classifier, source_val_loaders[d])
            source_f1s.append(f1_score(y_val, p_val, average="macro") * 100.0)
            source_feats_list.append(f_val)

        mean_src_val_f1 = float(np.mean(source_f1s))
        pooled_source_feats = np.concatenate(source_feats_list, axis=0)

        # Evaluate Target
        target_feats, target_preds, target_labels = extract_features_and_preds(backbone, classifier, target_loader)
        t_acc = accuracy_score(target_labels, target_preds) * 100.0
        t_f1 = f1_score(target_labels, target_preds, average="macro") * 100.0

        if name == "Source-Only":
            baseline_target_acc = t_acc
            delta_target_acc = 0.0
        else:
            delta_target_acc = t_acc - baseline_target_acc

        # Compute Domain Separability (C=1 Logistic Regression)
        dom_sep = compute_domain_separability(pooled_source_feats, target_feats)

        # Record Per-Class Accuracies
        cm = confusion_matrix(target_labels, target_preds, labels=list(range(len(CLASSES))))
        confusion_matrices[name] = cm
        class_accuracies = cm.diagonal() / cm.sum(axis=1) * 100.0
        per_class_records[name] = class_accuracies

        summary_records.append({
            "Method": name,
            "Source Val Macro-F1 (%)": round(mean_src_val_f1, 2),
            "Target Acc (%)": round(t_acc, 2),
            "Target Macro-F1 (%)": round(t_f1, 2),
            "Δ Target Acc (%)": round(delta_target_acc, 2),
            "Domain Separability (%)": round(dom_sep, 2)
        })

    # Summary Benchmark Table
    df_summary = pd.DataFrame(summary_records)
    summary_path = os.path.join(RESULTS_DIR, "adaptation_summary_benchmark.csv")
    df_summary.to_csv(summary_path, index=False)
    print("\n" + "="*80)
    print("TABLE 1: OVERALL ADAPTATION PERFORMANCE & DOMAIN SEPARABILITY")
    print("="*80)
    print(df_summary.to_string(index=False))

    # Per-Class Accuracy Table
    df_per_class = pd.DataFrame(per_class_records, index=CLASSES)
    df_per_class.to_csv(os.path.join(RESULTS_DIR, "target_per_class_accuracy.csv"))
    print("\n" + "="*80)
    print("TABLE 2: PER-CLASS TARGET ACCURACY (%)")
    print("="*80)
    print(df_per_class.round(2).to_string())

    # Detailed Class Gain/Degradation vs. Source-Only
    if "Source-Only" in per_class_records:
        src_cls = per_class_records["Source-Only"]
        print("\n" + "="*80)
        print("CLASS SHIFT ANALYSIS RELATIVE TO SOURCE-ONLY")
        print("="*80)
        for name in METHODS.keys():
            if name == "Source-Only" or name not in per_class_records:
                continue
            diffs = per_class_records[name] - src_cls
            best_cls = CLASSES[np.argmax(diffs)]
            worst_cls = CLASSES[np.argmin(diffs)]
            print(f"\n[{name}]")
            print(f"  Largest Gain:        {best_cls} (+{diffs[np.argmax(diffs)]:.2f}%)")
            print(f"  Largest Degradation: {worst_cls} ({diffs[np.argmin(diffs)]:.2f}%)")

            # Confusion analysis for worst-performing class
            worst_idx = np.argmin(diffs)
            cm_method = confusion_matrices[name]
            errors = cm_method[worst_idx].copy()
            errors[worst_idx] = 0
            dominant_err_idx = np.argmax(errors)
            print(f"  Dominant Confusion for '{worst_cls}': Misclassified as '{CLASSES[dominant_err_idx]}' ({errors[dominant_err_idx]} samples)")

if __name__ == "__main__":
    run_diagnostic()
