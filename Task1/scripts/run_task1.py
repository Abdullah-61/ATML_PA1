import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
import open_clip

from task1.configs.config import SEED, DEVICE, STL10_CLASSES, RESULTS_DIR
from task1.data.make_subset import get_datasets
from task1.data.transforms import apply_grayscale, apply_hue_rotation_180
from task1.models.backbones import load_backbones
from task1.analysis.evaluate_bias import extract_all_backbones, train_linear_head
from task1.analysis.feature_similarity import compute_cosine_stability

CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")

def main():
    # 1. Determinism
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Data & Backbones
    print("Partitioning STL-10 dataset...")
    train_subset, val_subset, eval_subset = get_datasets()

    print("Loading pretrained backbones...")
    backbones, norms = load_backbones(DEVICE)

    # 3. Features & Linear Head Training (Trained Once & Reused)
    probe_dims = {"resnet50": 2048, "vit_b16": 768, "clip": 512}
    trained_heads = {}
    models = ["resnet50", "vit_b16", "clip"]

    # Check if all probe weights already exist to skip redundant feature extraction & training
    all_checkpoints_exist = all(
        os.path.exists(os.path.join(CHECKPOINT_DIR, f"{m}_probe.pth")) for m in models
    )

    if all_checkpoints_exist:
        print("\n--> Found existing linear probe checkpoints. Reusing without re-training.")
        for m in models:
            head = torch.nn.Linear(probe_dims[m], 10).to(DEVICE)
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{m}_probe.pth")
            head.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            head.eval()
            trained_heads[m] = head
        print("Extracting features for clean 500-sample eval set...")
        feats_eval = extract_all_backbones(eval_subset, backbones, norms, DEVICE)
    else:
        print("\n--> Training probes for the first time (AdamW, lr=1e-3, early stopping=5)...")
        feats_train = extract_all_backbones(train_subset, backbones, norms, DEVICE)
        feats_val = extract_all_backbones(val_subset, backbones, norms, DEVICE)
        feats_eval = extract_all_backbones(eval_subset, backbones, norms, DEVICE)

        for m in models:
            print(f"Training probe for {m}...")
            head = train_linear_head(
                probe_dims[m], feats_train[m][0], feats_train[m][1],
                feats_val[m][0], feats_val[m][1], DEVICE, SEED
            )
            ckpt_path = os.path.join(CHECKPOINT_DIR, f"{m}_probe.pth")
            torch.save(head.state_dict(), ckpt_path)
            print(f"Saved {m} probe weights to {ckpt_path}")
            trained_heads[m] = head

    # 4. Zero-Shot CLIP Setup
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    prompts = [f"a photo of a {c}." for c in STL10_CLASSES]
    text_tokens = tokenizer(prompts).to(DEVICE)
    with torch.no_grad():
        text_features = backbones["clip"].encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    logit_scale = backbones["clip"].logit_scale.exp()

    # 5. Step 1: Clean Baseline Evaluation
    print("\n--- Evaluating Step 1: Clean Baseline ---")
    clean_metrics = {}
    baseline_records = []

    for m in models:
        f_eval, y_eval = feats_eval[m]
        with torch.no_grad():
            logits = trained_heads[m](f_eval.to(DEVICE))
            probs = torch.softmax(logits, dim=-1).cpu()
            max_confs, preds = probs.max(dim=-1)

        acc = float(accuracy_score(y_eval, preds) * 100.0)
        f1 = float(f1_score(y_eval, preds, average="macro") * 100.0)
        conf = float(max_confs.mean().item() * 100.0)

        key = f"{m.upper()} Linear Probe"
        clean_metrics[key] = {"acc": acc, "preds": preds}
        baseline_records.append({"Model": key, "Top-1 Accuracy (%)": f"{acc:.2f}", "Macro-F1 (%)": f"{f1:.2f}", "Mean Max Confidence (%)": f"{conf:.2f}"})

    # Zero-Shot CLIP Baseline
    f_clip, y_clip = feats_eval["clip"]
    with torch.no_grad():
        sims = (f_clip.to(DEVICE) @ text_features.T) * logit_scale
        zs_probs = torch.softmax(sims, dim=-1).cpu()
        zs_confs, zs_preds = zs_probs.max(dim=-1)

    acc_zs = float(accuracy_score(y_clip, zs_preds) * 100.0)
    f1_zs = float(f1_score(y_clip, zs_preds, average="macro") * 100.0)
    conf_zs = float(zs_confs.mean().item() * 100.0)
    clean_metrics["OpenCLIP ViT-B/32 Zero-Shot"] = {"acc": acc_zs, "preds": zs_preds}
    baseline_records.append({"Model": "OpenCLIP ViT-B/32 Zero-Shot", "Top-1 Accuracy (%)": f"{acc_zs:.2f}", "Macro-F1 (%)": f"{f1_zs:.2f}", "Mean Max Confidence (%)": f"{conf_zs:.2f}"})

    df_clean = pd.DataFrame(baseline_records)
    print(df_clean.to_string(index=False))
    df_clean.to_csv(os.path.join(RESULTS_DIR, "step1_clean_baseline.csv"), index=False)

    # Save clean predictions and clean representations for downstream interventions & t-SNE
    torch.save(clean_metrics, os.path.join(RESULTS_DIR, "clean_metrics.pt"))
    torch.save(feats_eval, os.path.join(RESULTS_DIR, "clean_eval_features.pt"))

    # 6. Step 2: Color Bias Evaluations (Reusing the Trained Heads & Clean Baselines)
    print("\n--- Evaluating Step 2: Color Interventions ---")
    color_records = []
    eval_targets = feats_eval["resnet50"][1]

    for int_name, int_fn in [("Grayscale", apply_grayscale), ("Hue Rotation (+180°)", apply_hue_rotation_180)]:
        print(f"Applying {int_name}...")
        trans_imgs = torch.stack([int_fn(eval_subset[i][0]) for i in range(len(eval_subset))], dim=0)

        loader = torch.utils.data.DataLoader(list(zip(trans_imgs, eval_targets)), batch_size=128, shuffle=False)
        trans_feats = {"resnet50": [], "vit_b16": [], "clip": []}
        with torch.no_grad():
            for imgs, _ in loader:
                imgs = imgs.to(DEVICE)
                trans_feats["resnet50"].append(backbones["resnet50"](norms["resnet50"](imgs)).cpu())
                trans_feats["vit_b16"].append(backbones["vit_b16"](norms["vit_b16"](imgs)).cpu())
                fc = backbones["clip"].encode_image(norms["clip"](imgs))
                trans_feats["clip"].append((fc / fc.norm(dim=-1, keepdim=True)).cpu())

        trans_feats = {k: torch.cat(v, dim=0) for k, v in trans_feats.items()}

        # Evaluate Probes
        for m in models:
            key = f"{m.upper()} Linear Probe"
            logits = trained_heads[m](trans_feats[m].to(DEVICE))
            preds = logits.argmax(dim=-1).cpu()

            acc = accuracy_score(eval_targets, preds) * 100.0
            consistency = (preds == clean_metrics[key]["preds"]).float().mean().item() * 100.0
            i_t = compute_cosine_stability(feats_eval[m][0], trans_feats[m])
            clean_acc = clean_metrics[key]["acc"]

            color_records.append({
                "Intervention": int_name, "Model": key, "Clean Acc (%)": clean_acc,
                "Transformed Acc (%)": acc, "Δ Acc Abs (%)": acc - clean_acc,
                "Consistency (%)": consistency, "Stability I_T": i_t
            })

        # Evaluate Zero-Shot CLIP
        sims = (trans_feats["clip"].to(DEVICE) @ text_features.T) * logit_scale
        zs_preds = sims.argmax(dim=-1).cpu()
        acc_zs = accuracy_score(eval_targets, zs_preds) * 100.0
        cons_zs = (zs_preds == clean_metrics["OpenCLIP ViT-B/32 Zero-Shot"]["preds"]).float().mean().item() * 100.0
        i_t_clip = compute_cosine_stability(feats_eval["clip"][0], trans_feats["clip"])
        clean_acc_zs = clean_metrics["OpenCLIP ViT-B/32 Zero-Shot"]["acc"]

        color_records.append({
            "Intervention": int_name, "Model": "OpenCLIP ViT-B/32 Zero-Shot",
            "Clean Acc (%)": clean_acc_zs, "Transformed Acc (%)": acc_zs,
            "Δ Acc Abs (%)": acc_zs - clean_acc_zs, "Consistency (%)": cons_zs,
            "Stability I_T": i_t_clip
        })

    df_color = pd.DataFrame(color_records)
    print(df_color.to_string(index=False))
    df_color.to_csv(os.path.join(RESULTS_DIR, "step2_color_bias.csv"), index=False)
    print("\nPipeline execution complete. All probe weights and clean baseline anchors are saved and will be reused.")

if __name__ == "__main__":
    main()
