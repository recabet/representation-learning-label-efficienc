import random
import os

from src.configs.global_config import GLOBAL_CONFIG

import torch
import numpy as np


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_experiment(cfg):
    os.makedirs(cfg.SAVE_DIR, exist_ok=True)


def log_environment():
    print("Torch version:", torch.__version__)
    print("Device:", GLOBAL_CONFIG.DEVICE)
    if GLOBAL_CONFIG.DEVICE:
        print("CUDA version:", torch.version.cuda)
        print("GPU:", torch.cuda.get_device_name(0))
