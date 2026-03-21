#!/usr/bin/env python
"""
pretrain_simclr.py

Pretrain SimCLR on unlabeled STL-10 dataset.

Saves checkpoints every 10 epochs.

Usage:
  # ResNet18 (default config):
  python -m src.experiments.pretrain_simclr

  # ResNet50: change BASE_MODEL in simclr_config.py to "resnet50", then:
  python -m src.experiments.pretrain_simclr
"""


from src.data_handling.datasets import STL10Dataset
from src.models.simclr import SimCLR
from src.training.train import fit_simclr
from src.losses.nt_xent import NTXentLoss
from src.configs.simclr_config import SIMCLR_CONFIG
from src.configs.global_config import GLOBAL_CONFIG
from src.experiments.cli_args import add_shared_training_args

import argparse
from pathlib import Path

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
        self.dataset = STL10Dataset(split=split, transform=None, labeled=False)
        self.transform = SIMCLR_CONFIG.SIMCLR_AUGMENTATIONS

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        x1 = self.transform(img)
        x2 = self.transform(img)
        return x1, x2


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pretrain SimCLR with optional config overrides"
    )
    add_shared_training_args(parser)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--warmup-epochs", type=int)
    parser.add_argument("--base-model", type=str)
    parser.add_argument("--out-dim", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--split", type=str)
    parser.add_argument("--checkpoint-dir", type=str)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=None)
    return parser


def _apply_cfg_overrides(cfg: SIMCLR_CONFIG, args: argparse.Namespace) -> SIMCLR_CONFIG:
    # Only apply fields that are passed from CLI.
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if args.learning_rate is not None:
        cfg.LEARNING_RATE = args.learning_rate
    if args.weight_decay is not None:
        cfg.WEIGHT_DECAY = args.weight_decay
    if args.warmup_epochs is not None:
        cfg.WARMUP_EPOCHS = args.warmup_epochs
    if args.base_model is not None:
        cfg.BASE_MODEL = args.base_model
    if args.out_dim is not None:
        cfg.OUT_DIM = args.out_dim
    if args.temperature is not None:
        cfg.TEMPERATURE = args.temperature
    if args.split is not None:
        cfg.UNLABELED_SPLIT = args.split
    if args.pretrained is not None:
        cfg.PRETRAINED = args.pretrained

    cfg.__post_init__()

    if args.checkpoint_dir is not None:
        cfg.CHECKPOINT_DIR = Path(args.checkpoint_dir)

    return cfg


# =========================================================
# DataLoader
# =========================================================

if __name__ == "__main__":
    import socket
    import threading

    args = _build_arg_parser().parse_args()

    # Instantiate config then override using any provided CLI args.
    cfg = _apply_cfg_overrides(SIMCLR_CONFIG(), args)

    train_dataset = SimCLRDataset(split=cfg.UNLABELED_SPLIT)

    train_loader = DataLoader(train_dataset,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=cfg.NUM_WORKERS,
                              drop_last=True)

    model = SimCLR(base_model=cfg.BASE_MODEL,
                   out_dim=cfg.OUT_DIM,
                   pretrained=cfg.PRETRAINED).to(GLOBAL_CONFIG.DEVICE)

    criterion = NTXentLoss(temperature=cfg.TEMPERATURE)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=cfg.LEARNING_RATE,
                                momentum=0.9,
                                weight_decay=cfg.WEIGHT_DECAY)

    warmup_epochs = cfg.WARMUP_EPOCHS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS - warmup_epochs, eta_min=1e-6
    )
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs
    )
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, scheduler], milestones=[warmup_epochs]
    )

    # ── Optional: live loss dashboard ──
    def _port_free(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return True
            except OSError:
                return False

    DASH_PORT = 7860
    try:
        from src.analysis.loss_dashboard import create_dashboard
    except ModuleNotFoundError:
        create_dashboard = None

    if create_dashboard and _port_free(DASH_PORT):
        dashboard = create_dashboard(
            checkpoints_root=GLOBAL_CONFIG.SAVE_DIR,
            refresh_interval=10,
        )
        dash_thread = threading.Thread(
            target=lambda: dashboard.launch(
                server_port=DASH_PORT, share=True, quiet=True
            ),
            daemon=True,
        )
        dash_thread.start()
        print(f"Loss dashboard launched at http://localhost:{DASH_PORT} (+ public Gradio link above)")
    elif create_dashboard:
        print(f"Dashboard already running at http://localhost:{DASH_PORT} - skipping.")
    else:
        print("Dashboard dependency not installed (gradio). Continuing without dashboard.")

    # ── Launch pretraining ──
    print(f"🚀 Pretraining SimCLR with {cfg.BASE_MODEL} for {cfg.EPOCHS} epochs")
    print(f"📁 Checkpoints → {cfg.CHECKPOINT_DIR}")

    fit_simclr(model,
               train_loader,
               criterion,
               optimizer,
               GLOBAL_CONFIG.DEVICE,
               epochs=cfg.EPOCHS,
               checkpoint_dir=str(cfg.CHECKPOINT_DIR),
               scheduler=lr_scheduler)
