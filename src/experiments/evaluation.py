"""
evaluation.py

Model evaluation utilities — computes accuracy, F1, precision, recall,
and confusion matrix on a held-out test set.
"""

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_model(model, data_loader, device="cuda"):
    """
    Run inference on *data_loader* and return a dict of classification metrics.

    Returns
    -------
    dict with keys: accuracy, f1_macro, f1_weighted,
                    precision_macro, recall_macro, confusion_matrix
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

