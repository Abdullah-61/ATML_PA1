import os
import sys
# Ensure parent directory is in sys.path when executed directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from torch.utils.data import DataLoader
from shared.pacs_protocol import SOURCE_DOMAINS, TARGET_DOMAIN
from shared.pacs import build_or_load_splits
from task2.models.backbone import ResNet18Backbone
from task2.models.classifier_head import LinearClassifierHead
from task2.evaluation.metrics import evaluate_dataset

def evaluate_source_only_erm(checkpoint_path="task2/results/checkpoints/source_only_erm.pth", data_root="./pacs_data"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    datasets = build_or_load_splits(data_root)

    backbone = ResNet18Backbone().to(device)
    classifier = LinearClassifierHead(in_features=backbone.out_features).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    backbone.load_state_dict(ckpt["backbone"])
    classifier.load_state_dict(ckpt["classifier"])

    records = []
    # 1. Source validation evaluations
    for d in SOURCE_DOMAINS:
        loader = DataLoader(datasets["source_val"][d], batch_size=64, shuffle=False, num_workers=2)
        acc, f1 = evaluate_dataset(backbone, classifier, loader, device)
        records.append({"Domain": f"{d.capitalize()} (Val)", "Split": "Source Val", "Accuracy (%)": f"{acc:.2f}", "Macro-F1 (%)": f"{f1:.2f}"})

    # 2. Final locked evaluation on unseen Sketch target domain
    target_loader = DataLoader(datasets["target_eval"], batch_size=64, shuffle=False, num_workers=2)
    t_acc, t_f1 = evaluate_dataset(backbone, classifier, target_loader, device)
    records.append({"Domain": f"{TARGET_DOMAIN.capitalize()} (Target)", "Split": "Target Final", "Accuracy (%)": f"{t_acc:.2f}", "Macro-F1 (%)": f"{t_f1:.2f}"})

    df = pd.DataFrame(records)
    csv_path = "task2/results/step1_source_only_erm_summary.csv"
    df.to_csv(csv_path, index=False)
    print("\n--- Source-Only ERM Final Evaluation Report ---")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    evaluate_source_only_erm()
