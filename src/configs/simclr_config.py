from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms

@dataclass
class SIMCLR_CONFIG:
    # -----------------------------
    # Device & Training Settings
    # -----------------------------
    EPOCHS: int = 100
    BATCH_SIZE: int = 128
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 1e-3
    CHECKPOINT_DIR: Path = Path(__file__).resolve().parents[2] / "checkpoints/simclr_pretrain"

    # -----------------------------
    # Model Settings
    # -----------------------------
    BASE_MODEL: str = "resnet18"  # resnet18 or resnet50
    OUT_DIM: int = 128
    PRETRAINED: bool = False  # Use ImageNet weights

    # -----------------------------
    # Dataset / Images
    # -----------------------------
    IMAGE_SIZE: int = 96  # STL-10 images
    UNLABELED_SPLIT: str = "unlabeled"

    # -----------------------------
    # SimCLR Augmentations
    # -----------------------------
    SIMCLR_AUGMENTATIONS = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
    ])

    # -----------------------------
    # NT-Xent Loss
    # -----------------------------
    TEMPERATURE: float = 0.5