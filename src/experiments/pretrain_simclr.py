#!/usr/bin/env python
"""
pretrain_simclr.py

Pretrain SimCLR on unlabeled STL-10 dataset.
Saves checkpoints every 10 epochs.
"""

from src.data_handling.datasets import STL10Dataset
from src.models.simclr import SimCLR
from src.training.train import fit_simclr
from src.losses.nt_xent import NTXentLoss
from src.configs.simclr_config import SIMCLR_CONFIG
from src.configs.global_config import GLOBAL_CONFIG

import torch
from torch.utils.data import DataLoader


# =========================================================
# SimCLR Dataset (FIXED)
# =========================================================

class SimCLRDataset(torch.utils.data.Dataset):
    """
    Wrap STL10Dataset to return two DIFFERENT augmented views for SimCLR.
    Each call applies the stochastic transform independently twice.
    """

    def __init__(self, split="unlabeled"):
        self.dataset = STL10Dataset(
            split=split,
            transform=None,
            labeled=False
        )
        self.transform = SIMCLR_CONFIG.SIMCLR_AUGMENTATIONS

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]

        # ✅ Apply stochastic transform twice
        x1 = self.transform(img)
        x2 = self.transform(img)

        return x1, x2


# =========================================================
# DataLoader
# =========================================================

train_dataset = SimCLRDataset(split=SIMCLR_CONFIG.UNLABELED_SPLIT)

train_loader = DataLoader(
    train_dataset,
    batch_size=SIMCLR_CONFIG.BATCH_SIZE,
    shuffle=True,
    num_workers=SIMCLR_CONFIG.NUM_WORKERS,
    drop_last=True,
    pin_memory=True,
)


# =========================================================
# Model
# =========================================================

model = SimCLR(
    base_model=SIMCLR_CONFIG.BASE_MODEL,
    out_dim=SIMCLR_CONFIG.OUT_DIM,
    pretrained=SIMCLR_CONFIG.PRETRAINED
).to(GLOBAL_CONFIG.DEVICE)


# =========================================================
# Loss
# =========================================================

criterion = NTXentLoss(
    temperature=SIMCLR_CONFIG.TEMPERATURE
)


# =========================================================
# Optimizer (FIXED → SGD instead of Adam)
# =========================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=SIMCLR_CONFIG.LEARNING_RATE,
    momentum=0.9,
    weight_decay=1e-4,
)


# =========================================================
# Scheduler (Warmup + Cosine)
# =========================================================

warmup_epochs = 10

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=SIMCLR_CONFIG.EPOCHS - warmup_epochs,
    eta_min=1e-6
)

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer,
    start_factor=0.01,
    total_iters=warmup_epochs
)

lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_epochs]
)


# =========================================================
# Main
# =========================================================

if __name__ == "__main__":

    import socket
    import threading
    from src.analysis.loss_dashboard import create_dashboard

    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    DASH_PORT = 7860

    if _port_free(DASH_PORT):
        dashboard = create_dashboard(
            checkpoints_root=GLOBAL_CONFIG.SAVE_DIR,
            refresh_interval=10,
        )

        dash_thread = threading.Thread(
            target=lambda: dashboard.launch(
                server_port=DASH_PORT,
                share=True,
                quiet=True
            ),
            daemon=True,
        )
        dash_thread.start()

        print(f"📉 Loss dashboard launched at http://localhost:{DASH_PORT}")

    else:
        print(f"📉 Dashboard already running at http://localhost:{DASH_PORT}")

    fit_simclr(
        model,
        train_loader,
        criterion,
        optimizer,
        GLOBAL_CONFIG.DEVICE,
        epochs=SIMCLR_CONFIG.EPOCHS,
        checkpoint_dir=str(SIMCLR_CONFIG.CHECKPOINT_DIR),
        scheduler=lr_scheduler,
    )