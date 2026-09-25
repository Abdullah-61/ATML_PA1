import random
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 6304

NEAR_UNKNOWN_CLASSES = [
    "bus", "pickup_truck", "motorcycle", "tractor",
    "wolf", "fox", "leopard", "camel"
]
FAR_UNKNOWN_CLASSES = [
    "bottle", "bowl", "chair", "clock",
    "keyboard", "mushroom", "sunflower", "wardrobe"
]

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

def get_cifar_transforms(use_randaugment=False):
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    if use_randaugment:
        train_ops.append(transforms.RandAugment(num_ops=2, magnitude=9))

    train_ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])

    test_ops = [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ]
    return transforms.Compose(train_ops), transforms.Compose(test_ops)

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return img, label, idx

def get_cifar10_splits(data_root="./data", use_randaugment=False):
    train_tf, eval_tf = get_cifar_transforms(use_randaugment=use_randaugment)
    full_train = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(root=data_root, train=True, download=True, transform=eval_tf)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=eval_tf)

    targets = np.array(full_train.targets)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    train_idx, val_idx = next(sss.split(np.zeros(len(targets)), targets))

    train_split = Subset(full_train, train_idx)
    val_split = Subset(full_train_eval, val_idx)
    train_unaugmented_split = Subset(full_train_eval, train_idx)

    return {
        "train": train_split,
        "val": val_split,
        "train_unaugmented": train_unaugmented_split,
        "test": test_set
    }

def get_cifar100_unknowns(data_root="./data"):
    _, eval_tf = get_cifar_transforms(use_randaugment=False)
    cifar100_test = datasets.CIFAR100(root=data_root, train=False, download=True, transform=eval_tf)

    fine_to_idx = {name: i for i, name in enumerate(cifar100_test.classes)}
    near_indices_set = {fine_to_idx[name] for name in NEAR_UNKNOWN_CLASSES}
    far_indices_set = {fine_to_idx[name] for name in FAR_UNKNOWN_CLASSES}

    near_samples, far_samples = [], []
    for idx, (img, target) in enumerate(cifar100_test):
        if target in near_indices_set:
            near_samples.append((idx, cifar100_test.classes[target]))
        elif target in far_indices_set:
            far_samples.append((idx, cifar100_test.classes[target]))

    assert len(near_samples) == 800, f"Expected 800 near samples, found {len(near_samples)}"
    assert len(far_samples) == 800, f"Expected 800 far samples, found {len(far_samples)}"

    near_indices = [idx for idx, _ in near_samples]
    far_indices = [idx for idx, _ in far_samples]

    return {
        "near": Subset(cifar100_test, near_indices),
        "far": Subset(cifar100_test, far_indices),
        "near_meta": near_samples,
        "far_meta": far_samples
    }
