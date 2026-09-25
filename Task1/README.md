# Representation Analysis & Robustness Benchmark (Task 1)

## Suggested Repository Structure Implementation
- `configs/config.py`: Seeds, device configuration, paths, and class labels.
- `data/make_subset.py`: STL-10 dataset loading, 80/20 train/val split, and class-balanced 500-sample test evaluation set.
- `data/transforms.py`: Grayscale, Hue Rotation (+180°), Translation, and Patch Permutation.
- `data/make_cue_conflicts.py`: Stylized cue-conflict image synthesis.
- `models/backbones.py`: ResNet-50, ViT-B/16, and OpenCLIP ViT-B/32 frozen extractors.
- `analysis/evaluate_bias.py`: Linear probe training engine and multi-backbone feature extraction.
- `analysis/feature_similarity.py`: Representation cosine stability (I_T) metric computation.
- `analysis/representation.py`: Dimensionality reduction (t-SNE/UMAP) across clean and intervened representations.
- `scripts/run_task1.py`: Orchestration driver script executing Step 1 (Clean Baseline) and Step 2 (Color Bias).
- `results/`: Persisted outputs, evaluation indices (`eval_500_indices.npy`), and metrics CSVs.
