import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch




@dataclass
class GLOBAL_CONFIG:
    SEED: int = 42
    DETERMINISTIC: bool = True
    BENCHMARK: bool = False

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    USE_AMP: bool = True
    COMPILE_MODEL: bool = False

    LOG_INTERVAL: int = 50
    SAVE_DIR: str = "checkpoints"
    NUM_WORKERS: int = 4

    PROJECT_NAME: str = "representation-learning-label-efficiency"

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"


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