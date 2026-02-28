#!/usr/bin/env python
"""
set_experiment.py

Sets up STL-10 experiments with limited labels for semi-supervised learning (SimCLR) and supervised baselines.
Includes:
- Data loading with percentage of labeled data
- Model initialization (scratch / ImageNet / SimCLR)
- Training loop
- Evaluation with multiple metrics
"""

from pathlib import Path


from src.data_handling.datasets import STL10Dataset
from src.models.simclr import SimCLR
from src.models.linear_probe import LinearProbe

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models


# -----------------------------
# Data & Transforms
# -----------------------------

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
    """
    Returns a subset of STL-10 labeled data corresponding to the given percentage
    """
    train_transforms, test_transforms = get_transforms()
    dataset = STL10Dataset(split=split, labeled=True, transform=train_transforms)

    n_total = len(dataset)
    n_selected = max(1, int(n_total * percent / 100))
    indices = np.random.choice(n_total, n_selected, replace=False)
    subset = Subset(dataset, indices)

    return subset


def get_test_dataset():
    _, test_transforms = get_transforms()
    test_dataset = STL10Dataset(split="test", labeled=True, transform=test_transforms)
    return test_dataset


# -----------------------------
# Training Loop
# -----------------------------

def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cuda"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Loss: {avg_loss:.4f}")

    print("Training complete.")
    return model


# -----------------------------
# Evaluation Metrics
# -----------------------------

def evaluate_model(model, data_loader, device="cuda"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.close()


# -----------------------------
# Experiment Setup
# -----------------------------

def run_experiment(percent_labels=10,
                   batch_size=64,
                   epochs=20,
                   device="cuda"):
    """
    Run experiment with a given percentage of labeled data
    """

    print(f"\n===== RUNNING EXPERIMENT: {percent_labels}% of labels =====")

    train_dataset = get_limited_dataset(split="train", percent=percent_labels)
    val_dataset = get_test_dataset()

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    print("\n--- Training ResNet18 from scratch ---")
    scratch_model = models.resnet18(weights=None)
    scratch_model.fc = nn.Linear(scratch_model.fc.in_features, 10)
    scratch_model = scratch_model.to(device)
    train_model(scratch_model, train_loader, val_loader, epochs=epochs, device=device)
    metrics_scratch = evaluate_model(scratch_model, val_loader, device=device)
    print("Metrics Scratch:", metrics_scratch)

    print("\n--- Fine-tuning ResNet18 pretrained on ImageNet ---")
    imagenet_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    imagenet_model.fc = nn.Linear(imagenet_model.fc.in_features, 10)
    imagenet_model = imagenet_model.to(device)
    train_model(imagenet_model, train_loader, val_loader, epochs=epochs, device=device)
    metrics_imagenet = evaluate_model(imagenet_model, val_loader, device=device)
    print("Metrics ImageNet:", metrics_imagenet)

    print("\n--- Linear Probe on SimCLR pretrained encoder ---")
    simclr_model = SimCLR(base_model="resnet18", out_dim=128, pretrained=False)
    # Load your pretrained SimCLR weights here if available:
    # simclr_model.load_state_dict(torch.load("simclr_encoder.pth"))

    for param in simclr_model.encoder.parameters():
        param.requires_grad = False

    linear_probe = LinearProbe(feat_dim=512, num_classes=10).to(device)

    class ProbeWrapper(nn.Module):
        def __init__(self, encoder, probe):
            super().__init__()
            self.encoder = encoder
            self.probe = probe

        def forward(self, x):
            h, _ = self.encoder(x)
            return self.probe(h)

    model_probe = ProbeWrapper(simclr_model.encoder, linear_probe).to(device)
    train_model(model_probe, train_loader, val_loader, epochs=epochs, device=device)
    metrics_probe = evaluate_model(model_probe, val_loader, device=device)
    print("Metrics Linear Probe:", metrics_probe)

    classes = ["airplane", "bird", "car", "cat", "deer", "dog", "horse", "monkey", "ship", "truck"]
    plot_confusion_matrix(metrics_scratch["confusion_matrix"], classes, save_path=Path(f"./cm_scratch.png"))
    plot_confusion_matrix(metrics_imagenet["confusion_matrix"], classes, save_path=Path(f"./cm_imagenet.png"))
    plot_confusion_matrix(metrics_probe["confusion_matrix"], classes, save_path=Path(f"./cm_probe.png"))

    print("\n===== EXPERIMENT COMPLETE =====")


if __name__ == "__main__":
    run_experiment(percent_labels=10,
                   batch_size=64,
                   epochs=10,
                   device="cuda")