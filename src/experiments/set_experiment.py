#!/usr/bin/env python
"""
set_experiment.py

STL-10 Semi-Supervised Experiment:
- Scratch ResNet18
- ImageNet ResNet18
- SimCLR + Linear Probe
- Metric comparison plots
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

from src.data_handling.datasets import STL10Dataset
from src.models.simclr import SimCLR
from src.models.linear_probe import LinearProbe
from src.models.probe_wrapper import ProbeWrapper
from src.training.train import fit_cls


# =========================================================
# Data & Transforms
# =========================================================

def get_transforms():
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(96),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor()
    ])

    return train_transforms, test_transforms


def get_limited_dataset(split="train", percent=10):
    train_transforms, _ = get_transforms()
    dataset = STL10Dataset(split=split, labeled=True, transform=train_transforms)

    n_total = len(dataset)
    n_selected = max(1, int(n_total * percent / 100))

    indices = np.random.choice(n_total, n_selected, replace=False)
    subset = Subset(dataset, indices)

    return subset


def get_test_dataset():
    _, test_transforms = get_transforms()
    return STL10Dataset(split="test", labeled=True, transform=test_transforms)


# =========================================================
# Evaluation
# =========================================================

def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            preds = outputs.argmax(dim=1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

    return metrics


def plot_confusion_matrix(cm, classes, save_path=None):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path)

    plt.close()


# =========================================================
# Metric Comparison Plots
# =========================================================

def plot_metric_comparison(metrics_scratch,
                           metrics_imagenet,
                           metrics_probe,
                           save_dir="plots"):

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True,exist_ok=True)

    model_names = ["Scratch", "ImageNet", "SimCLR Probe"]

    metrics_dict = {
        "accuracy": [
            metrics_scratch["accuracy"],
            metrics_imagenet["accuracy"],
            metrics_probe["accuracy"]
        ],
        "f1_macro": [
            metrics_scratch["f1_macro"],
            metrics_imagenet["f1_macro"],
            metrics_probe["f1_macro"]
        ],
        "f1_weighted": [
            metrics_scratch["f1_weighted"],
            metrics_imagenet["f1_weighted"],
            metrics_probe["f1_weighted"]
        ],
        "precision_macro": [
            metrics_scratch["precision_macro"],
            metrics_imagenet["precision_macro"],
            metrics_probe["precision_macro"]
        ],
        "recall_macro": [
            metrics_scratch["recall_macro"],
            metrics_imagenet["recall_macro"],
            metrics_probe["recall_macro"]
        ],
    }

    # Individual metric plots
    for metric_name, values in metrics_dict.items():
        plt.figure(figsize=(6, 5))
        sns.barplot(x=model_names, y=values)
        plt.title(f"{metric_name.upper()} Comparison")
        plt.ylim(0, 1)
        plt.ylabel(metric_name.upper())
        plt.tight_layout()
        plt.savefig(save_dir / f"{metric_name}_comparison.png")
        plt.close()

    # Combined plot
    plt.figure(figsize=(10, 6))

    x = np.arange(len(model_names))
    width = 0.15

    for i, (metric_name, values) in enumerate(metrics_dict.items()):
        plt.bar(x + i * width, values, width, label=metric_name)

    plt.xticks(x + width * 2, model_names)
    plt.ylim(0, 1)
    plt.ylabel("Score")
    plt.title("All Metrics Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "all_metrics_comparison.png")
    plt.close()


# =========================================================
# Experiment
# =========================================================

def run_experiment(experiment_name,
                   percent_labels=10,
                   simclr_path="",
                   batch_size=64,
                   epochs=100,
                   device="cuda"):

    print(f"\n===== RUNNING EXPERIMENT: {percent_labels}% labels =====")

    train_dataset = get_limited_dataset("train", percent_labels)
    val_dataset = get_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    # -----------------------------------------------------
    # 1️⃣ Scratch
    # -----------------------------------------------------

    print("\n--- Training ResNet18 from Scratch ---")

    scratch_model = models.resnet18(weights=None)
    scratch_model.fc = nn.Linear(scratch_model.fc.in_features, 10)

    fit_cls(scratch_model, train_loader,
            device=device,
            epochs=epochs,
            lr=1e-3, checkpoint_dir="checkpoints/scratch/"+experiment_name)

    metrics_scratch = evaluate_model(scratch_model, val_loader, device)
    print("Scratch Metrics:", metrics_scratch)

    # -----------------------------------------------------
    # 2️⃣ ImageNet
    # -----------------------------------------------------

    print("\n--- Fine-tuning ResNet18 (ImageNet) ---")

    imagenet_model = models.resnet18(
        weights=models.ResNet18_Weights.IMAGENET1K_V1
    )
    imagenet_model.fc = nn.Linear(imagenet_model.fc.in_features, 10)

    fit_cls(imagenet_model, train_loader,
            device=device, epochs=epochs,
            lr=1e-4, checkpoint_dir="checkpoints/imagenet/"+experiment_name)

    metrics_imagenet = evaluate_model(imagenet_model, val_loader, device)
    print("ImageNet Metrics:", metrics_imagenet)

    # -----------------------------------------------------
    # 3️⃣ SimCLR Linear Probe
    # -----------------------------------------------------

    print("\n--- Linear Probe on SimCLR Encoder ---")

    simclr_model = SimCLR(base_model="resnet18",
                          out_dim=128,
                          pretrained=True)

    simclr_model.load_state_dict(
        torch.load(simclr_path, map_location=device)
    )

    for param in simclr_model.encoder.parameters():
        param.requires_grad = False

    linear_probe = LinearProbe(feat_dim=512, num_classes=10)
    model_probe = ProbeWrapper(simclr_model.encoder, linear_probe)

    fit_cls(model_probe, train_loader,
            device=device, epochs=epochs,
            lr=1e-3,
            checkpoint_dir="checkpoints/linear_probe/"+experiment_name)

    metrics_probe = evaluate_model(model_probe, val_loader, device)
    print("Linear Probe Metrics:", metrics_probe)

    # -----------------------------------------------------
    # Confusion Matrices
    # -----------------------------------------------------

    classes = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

    plot_confusion_matrix(metrics_scratch["confusion_matrix"], classes, Path("cm_scratch.png"))
    plot_confusion_matrix(metrics_imagenet["confusion_matrix"], classes, Path("cm_imagenet.png"))
    plot_confusion_matrix(metrics_probe["confusion_matrix"], classes, Path("cm_probe.png"))



    plot_metric_comparison(metrics_scratch,
                           metrics_imagenet,
                           metrics_probe,
                           save_dir="plots/"+experiment_name)

    print("\n===== EXPERIMENT COMPLETE =====")