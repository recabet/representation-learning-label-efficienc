from dataclasses import dataclass
from pathlib import Path

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


