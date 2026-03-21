#!/usr/bin/env python
"""
set_experiment.py

Orchestrates the three-way comparison experiment on STL-10:
  1. Scratch — train ResNet from random init
  2. ImageNet — fine-tune an ImageNet-pretrained ResNet
  3. SimCLR  — linear probe on a frozen SimCLR encoder

Results (metrics JSON, confusion matrices, bar charts) are saved per experiment.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models

from src.models.simclr import SimCLR
from src.models.linear_probe import LinearProbe
from src.models.probe_wrapper import ProbeWrapper
from src.training.train import fit_cls
from src.experiments.evaluation import evaluate_model
from src.experiments.data_setup import get_limited_dataset, get_test_dataset
from src.analysis.visualization import (
    plot_confusion_matrix,
    plot_metric_comparison,
    STL10_CLASSES,
)
from src.configs.global_config import GLOBAL_CONFIG

# ─── Helpers ────────────────────────────────────────────────────────────────

def _serializable(metrics: dict) -> dict:
    """Convert a metrics dict to a JSON-serializable form."""
    return {k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in metrics.items()}


def _build_resnet(base_model: str, weights=None):
    """Instantiate a torchvision ResNet and return (model, feat_dim)."""
    if base_model == "resnet18":
        model = models.resnet18(weights=weights)
        return model, 512
    elif base_model == "resnet50":
        model = models.resnet50(weights=weights)
        return model, 2048
    else:
        raise ValueError(f"Unknown base_model: {base_model}")


def _imagenet_weights(base_model: str):
    """Return the appropriate ImageNet weight enum."""
    if base_model == "resnet18":
        return models.ResNet18_Weights.IMAGENET1K_V1
    return models.ResNet50_Weights.IMAGENET1K_V1


# ─── Main experiment function ───────────────────────────────────────────────

def run_experiment(
    experiment_name: str,
    percent_labels: int = 10,
    simclr_path: str = "",
    batch_size: int = 64,
    epochs: int = 100,
    device: str = "cuda",
    base_model: str = "resnet18",
    out_dim: int = 128,
):
    """
    Run a full three-way comparison and save all outputs.

    Parameters
    ----------
    experiment_name : str   – used for folder names
    percent_labels  : int   – percentage of labeled training data to use
    simclr_path     : str   – path to a SimCLR checkpoint (.pth)
    batch_size      : int
    epochs          : int   – epochs for scratch / ImageNet classifiers
    device          : str   – "cuda" or "cpu"
    base_model      : str   – "resnet18" or "resnet50"
    out_dim         : int   – projection dimension used when the SimCLR
                              checkpoint was trained (must match)
    """
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {experiment_name}  |  {percent_labels}% labels  |  {base_model}")
    print(f"{'='*60}")

    # ── Data ─────────────────────────────────────────────────────────────────
    train_dataset = get_limited_dataset("train", percent_labels)
    val_dataset = get_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    _, feat_dim = _build_resnet(base_model)

    # ── 1. Scratch ───────────────────────────────────────────────────────────
    print(f"\n--- 1/3  Training {base_model} from Scratch ---")

    scratch_model, _ = _build_resnet(base_model, weights=None)
    scratch_model.fc = nn.Linear(feat_dim, 10)

    fit_cls(scratch_model, train_loader, device=device, epochs=epochs,
            lr=1e-3, checkpoint_dir=GLOBAL_CONFIG.SAVE_DIR / Path(f"scratch/{experiment_name}"))

    metrics_scratch = evaluate_model(scratch_model, val_loader, device)
    print("Scratch  :", {k: f"{v:.4f}" for k, v in metrics_scratch.items()
                         if k != "confusion_matrix"})

    # ── 2. ImageNet fine-tune ────────────────────────────────────────────────
    print(f"\n--- 2/3  Fine-tuning {base_model} (ImageNet) ---")

    imagenet_model, _ = _build_resnet(base_model,
                                      weights=_imagenet_weights(base_model))
    imagenet_model.fc = nn.Linear(feat_dim, 10)

    fit_cls(imagenet_model, train_loader, device=device, epochs=epochs,
            lr=1e-4, checkpoint_dir=GLOBAL_CONFIG.SAVE_DIR / Path(f"imagenet/{experiment_name}"))

    metrics_imagenet = evaluate_model(imagenet_model, val_loader, device)
    print("ImageNet :", {k: f"{v:.4f}" for k, v in metrics_imagenet.items()
                         if k != "confusion_matrix"})

    # ── 3. SimCLR linear probe ───────────────────────────────────────────────
    print(f"\n--- 3/3  Linear Probe on SimCLR Encoder ({base_model}) ---")

    simclr_model = SimCLR(base_model=base_model, out_dim=out_dim,
                          pretrained=False)

    # Load only encoder weights (projector shape may differ between versions)
    checkpoint = torch.load(simclr_path, map_location=device)
    encoder_state = {k: v for k, v in checkpoint.items()
                     if k.startswith("encoder.")}
    simclr_model.load_state_dict(encoder_state, strict=False)

    for param in simclr_model.encoder.parameters():
        param.requires_grad = False

    linear_probe = LinearProbe(feat_dim=feat_dim, num_classes=10)
    model_probe = ProbeWrapper(simclr_model.encoder, linear_probe)

    fit_cls(model_probe, train_loader, device=device, epochs=200,
            lr=0.1, checkpoint_dir= GLOBAL_CONFIG.SAVE_DIR / Path(f"linear_probe/{experiment_name}"),
            use_sgd=True)

    metrics_probe = evaluate_model(model_probe, val_loader, device)
    print("Probe    :", {k: f"{v:.4f}" for k, v in metrics_probe.items()
                         if k != "confusion_matrix"})

    # ── Save everything ──────────────────────────────────────────────────────
    save_dir = GLOBAL_CONFIG.PROJECT_ROOT/ Path("plots") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrix(metrics_scratch["confusion_matrix"],
                          STL10_CLASSES, save_dir / "cm_scratch.png")
    plot_confusion_matrix(metrics_imagenet["confusion_matrix"],
                          STL10_CLASSES, save_dir / "cm_imagenet.png")
    plot_confusion_matrix(metrics_probe["confusion_matrix"],
                          STL10_CLASSES, save_dir / "cm_probe.png")

    plot_metric_comparison(metrics_scratch, metrics_imagenet, metrics_probe,
                           save_dir=str(save_dir))

    results = {
        "experiment_name": experiment_name,
        "percent_labels": percent_labels,
        "base_model": base_model,
        "epochs": epochs,
        "scratch": _serializable(metrics_scratch),
        "imagenet": _serializable(metrics_imagenet),
        "simclr_probe": _serializable(metrics_probe),
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📁  Results saved to {save_dir.resolve()}")
    print("=" * 60)

