"""
reproducibility.py

Utilities for reproducible experiments:
- Global seed setting (Python, NumPy, PyTorch, CUDA)
- Environment logging
"""

import os
import random

import numpy as np
import torch

from src.configs.global_config import GLOBAL_CONFIG


def set_seed(seed: int = GLOBAL_CONFIG.SEED, deterministic: bool = True):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_experiment(cfg):
    """Ensure output directories exist."""
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)


def log_environment():
    """Print runtime environment info."""
    print(f"PyTorch version : {torch.__version__}")
    print(f"Device          : {GLOBAL_CONFIG.DEVICE}")
    if torch.cuda.is_available():
        print(f"CUDA version    : {torch.version.cuda}")
        print(f"GPU             : {torch.cuda.get_device_name(0)}")

