import torch

SEED = 6304
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_ROOT = "./data_storage"
RESULTS_DIR = "task1/results"

STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
