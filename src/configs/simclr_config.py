from dataclasses import dataclass
from pathlib import Path
from torchvision import transforms

@dataclass
class SIMCLR_CONFIG:

    # -----------------------------
    # Training Settings
    # -----------------------------
    EPOCHS: int = 800                 # ⬆ longer training helps a lot
    BATCH_SIZE: int = 256
    NUM_WORKERS: int = 8              # ⬆ faster data loading
    LEARNING_RATE: float = 0.5        # ⬆ for SGD (not Adam)
    WEIGHT_DECAY: float = 1e-4
    WARMUP_EPOCHS: int = 10

    CHECKPOINT_DIR: Path = (
        Path(__file__).resolve().parents[2]
        / "checkpoints/simclr_pretrain_v3"
    )

    # -----------------------------
    # Model Settings
    # -----------------------------
    BASE_MODEL: str = "resnet18"
    OUT_DIM: int = 128
    PRETRAINED: bool = False

    # -----------------------------
    # Dataset
    # -----------------------------
    IMAGE_SIZE: int = 96
    UNLABELED_SPLIT: str = "unlabeled"

    # -----------------------------
    # SimCLR Augmentations (IMPROVED)
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
            [transforms.GaussianBlur(kernel_size=7)],  # ⬇ slightly smaller
            p=0.5
        ),

        transforms.ToTensor(),

        # ✅ IMPORTANT — normalization
        transforms.Normalize(
            mean=[0.4467, 0.4398, 0.4066],
            std=[0.2241, 0.2215, 0.2239]
        ),
    ])

    # -----------------------------
    # NT-Xent
    # -----------------------------
    TEMPERATURE: float = 0.5