"""
test.py

Load the best saved checkpoint and evaluate on the held-out test set.
Run this after training is complete.

Usage:
    python src/test.py
    python src/test.py --checkpoint checkpoints/best_model.pt --results-dir results
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import torch

from dataset  import load_data, get_dataloaders
from model    import MTLModel, MTLLoss
from evaluate import evaluate, compute_metrics, print_report, plot_confusion_matrix
from utils    import set_seed, get_device


def test(
    data_path: str,
    checkpoint_path: str = os.path.join("checkpoints", "best_model.pt"),
    results_dir: str = "results",
    batch_size: int = 64,
    alpha: float = 0.5,
    seed: int = 42,
):
    """
    Evaluate the best saved model on the test set.

    Args:
        data_path       : path to ai4i2020.csv
        checkpoint_path : path to saved .pt checkpoint
        results_dir     : where to save confusion_matrix.png
        batch_size      : DataLoader batch size
        alpha           : must match the value used during training
        seed            : must match the value used during training (for same split)

    Returns:
        metrics : dict of test metrics
    """
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # --- Data (same seed = same split as training) ---
    train_ds, val_ds, test_ds, _, w_bin, w_type = load_data(data_path, random_state=seed)
    _, _, test_loader = get_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    print(f"Test set size: {len(test_ds)}")

    # --- Load model from checkpoint ---
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Run `python src/train.py` first.")
        sys.exit(1)

    model = MTLModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    saved_epoch = checkpoint["epoch"]
    print(f"Loaded checkpoint from epoch {saved_epoch}")

    # --- Loss (needed for evaluate()) ---
    loss_fn = MTLLoss(
        alpha=alpha,
        binary_pos_weight=w_bin.to(device),
        type_class_weights=w_type.to(device),
    )

    # --- Evaluate ---
    test_loss, bin_preds, bin_targets, type_preds, type_targets = evaluate(
        model, test_loader, loss_fn, device
    )
    metrics = compute_metrics(bin_preds, bin_targets, type_preds, type_targets)

    # --- Report ---
    print(f"\nTest Loss: {test_loss:.4f}")
    print_report(bin_preds, bin_targets, type_preds, type_targets)

    print("Test metrics summary:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # --- Save confusion matrix ---
    os.makedirs(results_dir, exist_ok=True)
    plot_confusion_matrix(
        type_preds, type_targets,
        save_path=os.path.join(results_dir, "confusion_matrix.png"),
    )

    return metrics


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate best MTL checkpoint on test set")
    parser.add_argument("--data",       type=str, default=os.path.join("data", "ai4i2020.csv"))
    parser.add_argument("--checkpoint", type=str, default=os.path.join("checkpoints", "best_model.pt"))
    parser.add_argument("--results-dir",type=str, default="results")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha",      type=float, default=0.5)
    parser.add_argument("--seed",       type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    test(
        data_path       = args.data,
        checkpoint_path = args.checkpoint,
        results_dir     = args.results_dir,
        batch_size      = args.batch_size,
        alpha           = args.alpha,
        seed            = args.seed,
    )
