from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms

@dataclass
class SIMCLR_CONFIG:
    # -----------------------------
    # Device & Training Settings
    # -----------------------------
    EPOCHS: int = 500
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 4
    LEARNING_RATE: float = 0.3
    CHECKPOINT_DIR: Path = Path(__file__).resolve().parents[2] / "checkpoints/simclr_pretrain_v2"

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
    transforms.RandomResizedCrop(
        96,
        scale=(0.2, 1.0),
    ),
    transforms.RandomHorizontalFlip(),

    transforms.RandomApply([
        transforms.ColorJitter(
            brightness=0.8,
            contrast=0.8,
            saturation=0.8,
            hue=0.2
        )
    ], p=0.8),

    transforms.RandomGrayscale(p=0.2),

    transforms.RandomApply(
        [transforms.GaussianBlur(kernel_size=9)],
        p=0.5
    ),

    transforms.ToTensor(),
])

    # -----------------------------
    # NT-Xent Loss
    # -----------------------------
    TEMPERATURE: float = 0.2