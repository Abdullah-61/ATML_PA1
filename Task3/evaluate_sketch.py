import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from shared.pacs_protocol import SOURCE_DOMAINS, TARGET_DOMAIN, CLASSES
from shared.pacs import build_or_load_splits
from task3.models.backbone import ResNet18Backbone
from task3.models.classifier_head import LinearClassifierHead
from task3.evaluation.domain_metrics import (
    evaluate_domain,
    extract_features,
    compute_source_domain_separability,
    measure_sharpness_proxy
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = "task3/results"

MODELS = {
    "ERM (Baseline)": "task2/results/checkpoints/source_only_erm.pth",
    "DAN-DG": "task3/results/checkpoints/dan_dg_model.pth",
    "SAM": "task3/results/checkpoints/sam_model.pth"
}

def run_task3_final_evaluation():
    datasets = build_or_load_splits("./pacs_data")
    source_val_loaders = {
        d: DataLoader(datasets["source_val"][d], batch_size=64, shuffle=False, num_workers=2)
        for d in SOURCE_DOMAINS
    }
    target_loader = DataLoader(datasets["target_eval"], batch_size=64, shuffle=False, num_workers=2)

    summary_rows = []
    per_class_accuracies = {}
    confusion_dict = {}
    erm_target_acc = None

    for name, ckpt_path in MODELS.items():
        if not os.path.exists(ckpt_path):
            print(f"Skipping {name}: checkpoint {ckpt_path} not found.")
            continue

        backbone = ResNet18Backbone().to(DEVICE)
        classifier = LinearClassifierHead(in_features=backbone.out_features).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        backbone.load_state_dict(ckpt["backbone"])
        classifier.load_state_dict(ckpt["classifier"])

        # 1. Source evaluations
        src_accs, src_f1s = {}, {}
        src_feats = {}
        for d in SOURCE_DOMAINS:
            acc, f1 = evaluate_domain(backbone, classifier, source_val_loaders[d], DEVICE)
            src_accs[d] = acc
            src_f1s[d] = f1
            src_feats[d] = extract_features(backbone, source_val_loaders[d], DEVICE)

        mean_src_acc = np.mean(list(src_accs.values()))
        worst_src_acc = min(src_accs.values())
        mean_src_f1 = np.mean(list(src_f1s.values()))
        worst_src_f1 = min(src_f1s.values())

        # 2. Source-side diagnostics
        sep_score = compute_source_domain_separability(src_feats)
        sharpness = measure_sharpness_proxy(backbone, classifier, datasets, DEVICE, rho=0.05)

        # 3. Final locked Sketch evaluation
        backbone.eval()
        classifier.eval()
        t_preds, t_targets = [], []
        with torch.no_grad():
            for imgs, labels, _ in target_loader:
                imgs = imgs.to(DEVICE)
                preds = classifier(backbone(imgs)).argmax(dim=1).cpu()
                t_preds.extend(preds.numpy().tolist())
                t_targets.extend(labels.numpy().tolist())

        t_acc = accuracy_score(t_targets, t_preds) * 100.0
        t_f1 = f1_score(t_targets, t_preds, average="macro", zero_division=0) * 100.0

        if name == "ERM (Baseline)":
            erm_target_acc = t_acc
            delta_target = 0.0
        else:
            delta_target = t_acc - erm_target_acc

        cm = confusion_matrix(t_targets, t_preds, labels=list(range(len(CLASSES))))
        confusion_dict[name] = cm
        per_class_accuracies[name] = cm.diagonal() / cm.sum(axis=1) * 100.0

        summary_rows.append({
            "Method": name,
            "Photo Acc": round(src_accs["photo"], 2),
            "Art Acc": round(src_accs["art_painting"], 2),
            "Cartoon Acc": round(src_accs["cartoon"], 2),
            "Mean Source Acc": round(mean_src_acc, 2),
            "Worst Source Acc": round(worst_src_acc, 2),
            "Mean Source F1": round(mean_src_f1, 2),
            "Worst Source F1": round(worst_src_f1, 2),
            "Source Separability (%)": round(sep_score, 2),
            "Loss Sharpness (Proxy)": round(sharpness, 4),
            "Sketch Acc (%)": round(t_acc, 2),
            "Sketch Macro-F1 (%)": round(t_f1, 2),
            "Δ Sketch Acc vs ERM": round(delta_target, 2)
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "task3_dg_benchmark_summary.csv"), index=False)
    print("\n" + "="*100)
    print("TASK 3 DOMAIN GENERALIZATION BENCHMARK SUMMARY")
    print("="*100)
    print(df_summary.to_string(index=False))

    df_classes = pd.DataFrame(per_class_accuracies, index=CLASSES)
    df_classes.to_csv(os.path.join(RESULTS_DIR, "task3_sketch_per_class.csv"))
    print("\n" + "="*80)
    print("PER-CLASS SKETCH ACCURACY (%)")
    print("="*80)
    print(df_classes.round(2).to_string())

    if "ERM (Baseline)" in per_class_accuracies:
        erm_cls = per_class_accuracies["ERM (Baseline)"]
        print("\n" + "="*80)
        print("PER-CLASS SHIFTS & DOMINANT CONFUSIONS RELATIVE TO ERM")
        print("="*80)
        for name in ["DAN-DG", "SAM"]:
            if name not in per_class_accuracies:
                continue
            diffs = per_class_accuracies[name] - erm_cls
            best_idx = np.argmax(diffs)
            worst_idx = np.argmin(diffs)
            print(f"\n[{name}]")
            print(f"  Largest Gain:        {CLASSES[best_idx]} (+{diffs[best_idx]:.2f}%)")
            print(f"  Largest Degradation: {CLASSES[worst_idx]} ({diffs[worst_idx]:.2f}%)")

            cm = confusion_dict[name]
            errs = cm[worst_idx].copy()
            errs[worst_idx] = 0
            dom_err = np.argmax(errs)
            print(f"  Dominant Confusion for '{CLASSES[worst_idx]}': Misclassified as '{CLASSES[dom_err]}' ({errs[dom_err]} samples)")

if __name__ == "__main__":
    run_task3_final_evaluation()
