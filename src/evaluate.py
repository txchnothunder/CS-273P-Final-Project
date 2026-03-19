"""
evaluate.py

Evaluation utilities:
  - evaluate        : run model over a DataLoader, return predictions + targets
  - compute_metrics : F1, precision, recall, accuracy for both tasks
  - print_report    : pretty-print a classification report
  - plot_confusion_matrix : visualize confusion matrix for failure type
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    roc_auc_score,
)

FAILURE_TYPE_NAMES = ["No Failure", "TWF", "HDF", "PWF", "OSF", "RNF"]


# ------------------------------------------------------------------
# Inference pass
# ------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    """
    Run model over an entire DataLoader without gradient computation.

    Returns:
        avg_loss        : mean total loss over all batches
        binary_preds    : np.ndarray (N,) of predicted binary labels {0, 1}
        binary_targets  : np.ndarray (N,) of ground truth binary labels
        type_preds      : np.ndarray (N,) of predicted type class indices 0-5
        type_targets    : np.ndarray (N,) of ground truth type class indices
    """
    model.eval()

    total_loss = 0.0
    all_binary_preds, all_binary_targets = [], []
    all_type_preds,   all_type_targets   = [], []

    for X, b_true, t_true in loader:
        X, b_true, t_true = X.to(device), b_true.to(device), t_true.to(device)

        bin_logits, type_logits = model(X)
        loss, _, _ = loss_fn(bin_logits, type_logits, b_true, t_true)
        total_loss += loss.item()

        bin_preds  = (torch.sigmoid(bin_logits) >= 0.5).long()
        type_preds = type_logits.argmax(dim=1)

        all_binary_preds.append(bin_preds.cpu().numpy())
        all_binary_targets.append(b_true.cpu().long().numpy())
        all_type_preds.append(type_preds.cpu().numpy())
        all_type_targets.append(t_true.cpu().numpy())

    avg_loss       = total_loss / len(loader)
    binary_preds   = np.concatenate(all_binary_preds)
    binary_targets = np.concatenate(all_binary_targets)
    type_preds     = np.concatenate(all_type_preds)
    type_targets   = np.concatenate(all_type_targets)

    return avg_loss, binary_preds, binary_targets, type_preds, type_targets


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def compute_metrics(binary_preds, binary_targets, type_preds, type_targets):
    """
    Compute evaluation metrics for both tasks.

    Returns a dict with keys prefixed by "binary_" and "type_".
    """
    metrics = {}

    # --- Binary task ---
    metrics["binary_accuracy"]  = accuracy_score(binary_targets, binary_preds)
    metrics["binary_f1"]        = f1_score(binary_targets, binary_preds, zero_division=0)
    metrics["binary_precision"] = precision_score(binary_targets, binary_preds, zero_division=0)
    metrics["binary_recall"]    = recall_score(binary_targets, binary_preds, zero_division=0)

    # ROC-AUC only if both classes are present
    if len(np.unique(binary_targets)) == 2:
        metrics["binary_roc_auc"] = roc_auc_score(binary_targets, binary_preds)

    # --- Type task (macro + weighted F1 for imbalanced classes) ---
    metrics["type_accuracy"]       = accuracy_score(type_targets, type_preds)
    metrics["type_f1_macro"]       = f1_score(type_targets, type_preds, average="macro",    zero_division=0)
    metrics["type_f1_weighted"]    = f1_score(type_targets, type_preds, average="weighted", zero_division=0)
    metrics["type_precision_macro"]= precision_score(type_targets, type_preds, average="macro",    zero_division=0)
    metrics["type_recall_macro"]   = recall_score(type_targets, type_preds, average="macro",    zero_division=0)

    return metrics


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------

def print_report(binary_preds, binary_targets, type_preds, type_targets):
    """Print sklearn classification reports for both tasks."""
    print("\n--- Binary Failure Classification ---")
    print(classification_report(
        binary_targets, binary_preds,
        target_names=["No Failure", "Failure"],
        zero_division=0,
    ))

    print("--- Failure Type Classification (6-class) ---")
    print(classification_report(
        type_targets, type_preds,
        target_names=FAILURE_TYPE_NAMES,
        zero_division=0,
    ))


# ------------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------------

def plot_confusion_matrix(type_preds, type_targets, save_path: str = None):
    """
    Plot and optionally save the confusion matrix for the 6-class type task.
    """
    cm = confusion_matrix(type_targets, type_preds, labels=list(range(6)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2f"],
        ["Confusion Matrix (counts)", "Confusion Matrix (normalized)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, ax=ax,
            xticklabels=FAILURE_TYPE_NAMES,
            yticklabels=FAILURE_TYPE_NAMES,
            cmap="YlOrBr",
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
