"""
utils.py

Shared utilities:
  - set_seed        : global reproducibility
  - save_checkpoint : save model + optimizer state
  - load_checkpoint : restore model + optimizer state
  - plot_loss_curves: plot train/val loss over epochs
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------

def set_seed(seed: int = 42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def save_checkpoint(model, optimizer, epoch: int, metrics: dict, path: str):
    """
    Save model and optimizer state to disk.

    Args:
        model     : nn.Module
        optimizer : torch optimizer
        epoch     : current epoch number
        metrics   : dict of metric values to store alongside weights
        path      : file path ending in .pt
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(model, optimizer, path: str, device: torch.device):
    """
    Load model and optimizer state from disk.

    Returns:
        epoch   : epoch the checkpoint was saved at
        metrics : metrics dict stored with the checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["metrics"]


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_loss_curves(history: dict, save_path: str = None):
    """
    Plot train and validation loss curves.

    Args:
        history   : dict with keys "train_loss", "val_loss",
                    and optionally "train_binary_loss", "val_binary_loss",
                    "train_type_loss", "val_type_loss"
        save_path : if provided, saves figure to this path
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    has_subtasks = "train_binary_loss" in history

    fig, axes = plt.subplots(1, 3 if has_subtasks else 1, figsize=(15 if has_subtasks else 6, 4))
    if not has_subtasks:
        axes = [axes]

    # Total loss
    axes[0].plot(epochs, history["train_loss"], label="Train", color="#2A9D8F")
    axes[0].plot(epochs, history["val_loss"],   label="Val",   color="#E76F51")
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    if has_subtasks:
        # Binary loss
        axes[1].plot(epochs, history["train_binary_loss"], label="Train", color="#2A9D8F")
        axes[1].plot(epochs, history["val_binary_loss"],   label="Val",   color="#E76F51")
        axes[1].set_title("Binary Failure Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend()

        # Type loss
        axes[2].plot(epochs, history["train_type_loss"], label="Train", color="#2A9D8F")
        axes[2].plot(epochs, history["val_type_loss"],   label="Val",   color="#E76F51")
        axes[2].set_title("Failure Type Loss")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Loss curves saved to {save_path}")

    plt.show()


def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")