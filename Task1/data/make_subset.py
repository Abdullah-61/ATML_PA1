import os
import numpy as np
import torchvision
from torchvision import transforms
from torch.utils.data import Subset
from sklearn.model_selection import StratifiedShuffleSplit
from task1.configs.config import SEED, DATA_ROOT, RESULTS_DIR

# Common 224x224 RGB base transform prior to model normalization
common_base_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])

def get_datasets():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_ROOT, exist_ok=True)

    stl10_train = torchvision.datasets.STL10(
        root=DATA_ROOT, split="train", download=True, transform=common_base_transform
    )
    stl10_test = torchvision.datasets.STL10(
        root=DATA_ROOT, split="test", download=True, transform=common_base_transform
    )

    # 80/20 stratified train/val split using seed 6304
    train_labels = np.array(stl10_train.labels)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, val_idx = next(sss_val.split(np.zeros(len(train_labels)), train_labels))

    train_subset = Subset(stl10_train, train_idx)
    val_subset = Subset(stl10_train, val_idx)

    # Stratified class-balanced 500-sample test evaluation subset
    test_labels = np.array(stl10_test.labels)
    sss_test = StratifiedShuffleSplit(n_splits=1, train_size=500, random_state=SEED)
    eval_500_idx, _ = next(sss_test.split(np.zeros(len(test_labels)), test_labels))
    eval_subset = Subset(stl10_test, eval_500_idx)

    # Persist the 500 selected test image identifiers
    np.save(os.path.join(RESULTS_DIR, "eval_500_indices.npy"), eval_500_idx)

    return train_subset, val_subset, eval_subset
