"""
data_setup.py

Dataset and transform helpers for STL-10 downstream experiments.
Provides train/test transforms and a reproducible label-subset sampler.
"""

import numpy as np
from torch.utils.data import Subset
from torchvision import transforms

from src.data_handling.datasets import STL10Dataset

# ── STL-10 channel statistics (computed from the training split) ─────────────
STL10_MEAN = [0.4467, 0.4398, 0.4066]
STL10_STD = [0.2241, 0.2215, 0.2239]


def get_transforms():
    """Return (train_transform, test_transform) for downstream classification."""
    normalize = transforms.Normalize(mean=STL10_MEAN, std=STL10_STD)

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(112),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        normalize,
    ])

    return train_transforms, test_transforms


def get_limited_dataset(split="train", percent=10, seed=42):
    """
    Return a reproducible random subset of the labeled STL-10 split.

    Parameters
    ----------
    split : str – "train" or "test"
    percent : int – percentage of the full split to keep (1-100)
    seed : int – random seed for subset selection
    """
    train_transforms, _ = get_transforms()
    dataset = STL10Dataset(split=split, labeled=True, transform=train_transforms)

    n_total = len(dataset)
    n_selected = max(1, int(n_total * percent / 100))

    rng = np.random.RandomState(seed=seed)
    indices = rng.choice(n_total, n_selected, replace=False)
    return Subset(dataset, indices)


def get_test_dataset():
    """Return the full labeled STL-10 test set with eval transforms."""
    _, test_transforms = get_transforms()
    return STL10Dataset(split="test", labeled=True, transform=test_transforms)

