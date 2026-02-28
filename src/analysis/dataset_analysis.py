#!/usr/bin/env python
"""
dataset_analysis.py

Comprehensive exploratory data analysis (EDA) for STL-10 binary dataset.

Performs:
- Dataset size inspection
- Shape verification
- Label distribution
- Pixel statistics (mean/std)
- Brightness distribution
- Sample visualization
- Channel histograms
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from src.data_handling.datasets import STL10Dataset
from src.configs.global_config import GLOBAL_CONFIG


# -------------------------------------------------
# Utility Functions
# -------------------------------------------------

def compute_channel_stats(data: np.ndarray):
    """
    Compute per-channel mean and std (normalized to [0,1])
    """
    data = data.astype(np.float32) / 255.0
    mean = data.mean(axis=(0, 1, 2))
    std = data.std(axis=(0, 1, 2))
    return mean, std


def compute_brightness(data: np.ndarray):
    """
    Compute per-image brightness distribution
    """
    brightness = data.mean(axis=(1, 2, 3))
    return brightness


def plot_sample_images(data: np.ndarray, save_dir: Path, n=16):
    """
    Plot grid of sample images
    """
    plt.figure(figsize=(8, 8))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(data[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_dir / "sample_images.png")
    plt.close()


def plot_channel_histograms(data: np.ndarray, save_dir: Path):
    """
    Plot RGB channel histograms
    """
    plt.figure(figsize=(8, 5))

    colors = ["r", "g", "b"]
    for i, c in enumerate(colors):
        plt.hist(
            data[..., i].flatten(),
            bins=50,
            alpha=0.5,
            label=f"{c.upper()} channel"
        )

    plt.legend()
    plt.title("RGB Channel Histogram")
    plt.savefig(save_dir / "channel_histograms.png")
    plt.close()


def plot_brightness_distribution(brightness, save_dir: Path):
    plt.figure()
    plt.hist(brightness, bins=50)
    plt.title("Brightness Distribution")
    plt.savefig(save_dir / "brightness_distribution.png")
    plt.close()



def plot_class_distribution(dataset, save_dir: Path):
    """
    Plot bar chart of class distribution
    """
    STL10_CLASSES = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

    counts = Counter(dataset.labels)
    counts_list = [counts[i] for i in range(10)]

    plt.figure(figsize=(8, 5))
    plt.bar(STL10_CLASSES, counts_list, color='skyblue')
    plt.xticks(rotation=45)
    plt.ylabel("Number of Images")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(save_dir / "class_distribution.png")
    plt.close()


# -------------------------------------------------
# Main Analysis
# -------------------------------------------------

def analyze_split(split: str, labeled: bool = True):

    print(f"\n========== Analyzing {split.upper()} split ==========")

    dataset = STL10Dataset(split=split, labeled=labeled)
    data = dataset.data

    print(f"Number of samples: {len(dataset)}")
    print(f"Image shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Pixel range: [{data.min()}, {data.max()}]")

    if labeled and dataset.labels is not None:
        label_counts = Counter(dataset.labels)
        print("\nLabel distribution:")
        for k in sorted(label_counts.keys()):
            print(f"Class {k}: {label_counts[k]}")

    # Compute stats
    mean, std = compute_channel_stats(data)
    print("\nChannel mean (normalized):", mean)
    print("Channel std  (normalized):", std)

    brightness = compute_brightness(data)
    print("\nBrightness:")
    print("Mean:", brightness.mean())
    print("Std :", brightness.std())

    save_dir = GLOBAL_CONFIG.PROCESSED_DATA_DIR / "dataset_analysis"
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- FIXED LOGIC HERE ----
    if labeled and dataset.labels is not None:
        plot_sample_images_with_labels(dataset, save_dir)
        plot_class_distribution(dataset, save_dir)
        label_counts = Counter(dataset.labels)
        print("\nLabel distribution:")
        for k in sorted(label_counts.keys()):
            print(f"Class {k}: {label_counts[k]}")
    else:
        plot_sample_images(data, save_dir)

    plot_channel_histograms(data, save_dir)
    plot_brightness_distribution(brightness, save_dir)

    print(f"\nPlots saved to: {save_dir}")


def plot_sample_images_with_labels(dataset, save_dir: Path, n: int = 16):
    """
    Plot grid of sample images with labels as titles
    """

    STL10_CLASSES = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]

    indices = np.random.choice(len(dataset), n, replace=False)

    plt.figure(figsize=(10, 10))

    for i, idx in enumerate(indices):
        img = dataset.data[idx]
        label = dataset.labels[idx]

        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.title(STL10_CLASSES[label], fontsize=9)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_dir / "sample_images_with_labels.png")
    plt.close()

if __name__ == "__main__":

    np.random.seed(GLOBAL_CONFIG.SEED)

    analyze_split("train", labeled=True)
    analyze_split("test", labeled=True)
    # analyze_split("unlabeled", labeled=False)

    print("\nDataset analysis complete.")