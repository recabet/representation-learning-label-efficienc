"""
visualization.py

Plotting utilities for experiment results — confusion matrices and
per-metric bar-chart comparisons.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ── STL-10 class names (0-indexed) ──────────────────────────────────────────
STL10_CLASSES = [
    "airplane", "bird", "car", "cat", "deer",
    "dog", "horse", "monkey", "ship", "truck",
]


def plot_confusion_matrix(cm, classes=None, save_path=None):
    """
    Render a confusion-matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray  – (num_classes, num_classes)
    classes : list[str] or None – tick labels (defaults to STL10_CLASSES)
    save_path : Path or None – if given, saves figure to disk
    """
    if classes is None:
        classes = STL10_CLASSES

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_metric_comparison(metrics_scratch, metrics_imagenet, metrics_probe,
                           save_dir="plots"):
    """
    Generate per-metric bar charts and a combined chart comparing
    Scratch / ImageNet / SimCLR Probe performance.

    Parameters
    ----------
    metrics_scratch, metrics_imagenet, metrics_probe : dict
        Each must contain the keys used below (accuracy, f1_macro, …).
    save_dir : str or Path – directory where PNGs are saved.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = ["Scratch", "ImageNet", "SimCLR Probe"]

    metrics_dict = {
        "accuracy": [metrics_scratch["accuracy"],
                     metrics_imagenet["accuracy"],
                     metrics_probe["accuracy"]],
        "f1_macro": [metrics_scratch["f1_macro"],
                     metrics_imagenet["f1_macro"],
                     metrics_probe["f1_macro"]],
        "f1_weighted": [metrics_scratch["f1_weighted"],
                        metrics_imagenet["f1_weighted"],
                        metrics_probe["f1_weighted"]],
        "precision_macro": [metrics_scratch["precision_macro"],
                            metrics_imagenet["precision_macro"],
                            metrics_probe["precision_macro"]],
        "recall_macro": [metrics_scratch["recall_macro"],
                         metrics_imagenet["recall_macro"],
                         metrics_probe["recall_macro"]],
    }

    # ── Individual metric plots ──────────────────────────────────────────────
    for metric_name, values in metrics_dict.items():
        plt.figure(figsize=(6, 5))
        sns.barplot(x=model_names, y=values)
        plt.title(f"{metric_name.upper()} Comparison")
        plt.ylim(0, 1)
        plt.ylabel(metric_name.upper())
        plt.tight_layout()
        plt.savefig(save_dir / f"{metric_name}_comparison.png")
        plt.close()

    # ── Combined plot ────────────────────────────────────────────────────────
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

