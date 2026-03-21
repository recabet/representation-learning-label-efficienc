from pathlib import Path
from PIL import Image
from typing import Union, Tuple

import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

from src.configs.global_config import GLOBAL_CONFIG


def read_X_bin(file_path: Path):
    """
    Reads STL-10 X binary file correctly.
    STL-10 is stored in column-major (Fortran) order.
    Returns images as (N, H, W, C)
    """
    with open(file_path, "rb") as f:
        data = np.fromfile(f, dtype=np.uint8)

    # Reshape with Fortran order
    data = data.reshape(-1, 3, 96, 96)

    # Convert (N, C, H, W) -> (N, H, W, C)
    data = np.transpose(data, (0, 2, 3, 1))

    data = np.array([np.rot90(img, k=-1) for img in data])

    return data


def read_y_bin(file_path: Path):
    """
    Reads STL-10 y binary file and returns labels as numpy array (0-based)
    """
    with open(file_path, "rb") as f:
        labels = np.fromfile(f, dtype=np.uint8)
    labels = labels - 1
    return labels


class STL10Dataset(Dataset):
    """
    PyTorch Dataset for STL-10 binary files
    Supports labeled and unlabeled modes
    Can apply transforms (including SimCLR dual views)
    """

    def __init__(self,
                 split: str = "train",
                 transform: transforms = None,
                 labeled: bool = True):

        self.root: Path = Path(GLOBAL_CONFIG.RAW_DATA_DIR) / "stl10_binary"
        self.transform: transforms = transform
        self.labeled: bool = labeled

        if split == "train":
            self.data = read_X_bin(self.root / "train_X.bin")
            if labeled:
                self.labels = read_y_bin(self.root / "train_y.bin")
            else:
                self.labels = None
        elif split == "test":
            self.data = read_X_bin(self.root / "test_X.bin")
            if labeled:
                self.labels = read_y_bin(self.root / "test_y.bin")
            else:
                self.labels = None
        elif split == "unlabeled":
            self.data = read_X_bin(self.root / "unlabeled_X.bin")
            self.labels = None
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Union[Image.Image, Tuple[Image.Image, int]]:

        img = Image.fromarray(self.data[idx])
        if self.transform:
            img = self.transform(img)
        if self.labeled and self.labels is not None:
            label = int(self.labels[idx])
            return img, label
        else:
            return img
