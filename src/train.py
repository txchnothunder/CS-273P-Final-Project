"""
train.py

Training loop for the MTL predictive maintenance model.
Saves the best checkpoint to checkpoints/best_model.pt and loss curves to results/.

Usage:
    python src/train.py                      # defaults
    python src/train.py --epochs 50 --lr 1e-3 --alpha 0.5 --batch-size 64
    python src/train.py --no-early-stopping

After training, evaluate on the test set with:
    python src/test.py
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.optim as optim

from dataset  import load_data, get_dataloaders
from model    import MTLModel, MTLLoss, count_parameters
from evaluate import evaluate, compute_metrics
from utils    import set_seed, save_checkpoint, plot_loss_curves, get_device


# ------------------------------------------------------------------
# One epoch
# ------------------------------------------------------------------

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    """Run one full pass over the training DataLoader."""
    model.train()
    total, bin_total, type_total = 0.0, 0.0, 0.0

    for X, b_true, t_true in loader:
        X, b_true, t_true = X.to(device), b_true.to(device), t_true.to(device)

        optimizer.zero_grad()
        bin_logits, type_logits = model(X)
        loss, bl, tl = loss_fn(bin_logits, type_logits, b_true, t_true)
        loss.backward()
        optimizer.step()

        total      += loss.item()
        bin_total  += bl.item()
        type_total += tl.item()

    n = len(loader)
    return total / n, bin_total / n, type_total / n


# ------------------------------------------------------------------
# Full training pipeline
# ------------------------------------------------------------------

def train(
    data_path: str,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    alpha: float = 0.5,
    dropout: float = 0.3,
    patience: int = 10,
    seed: int = 42,
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
):
    """
    Train the MTL model. Saves best checkpoint and loss curve plots.

    Args:
        data_path     : path to ai4i2020.csv
        epochs        : maximum training epochs
        lr            : Adam learning rate
        batch_size    : samples per batch
        alpha         : weight on binary loss (1-alpha goes to type loss)
        dropout       : dropout rate in model
        patience      : early stopping patience (set to epochs to disable)
        seed          : random seed
        checkpoint_dir: where to save best_model.pt
        results_dir   : where to save loss_curves.png

    Returns:
        history : dict of per-epoch loss and metric lists
    """
    set_seed(seed)
    device = get_device()
    print(f"Device: {device}")

    # --- Data ---
    train_ds, val_ds, test_ds, _, w_bin, w_type = load_data(data_path, random_state=seed)
    train_loader, val_loader, _ = get_dataloaders(train_ds, val_ds, test_ds, batch_size=batch_size)
    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    # --- Model ---
    model = MTLModel(dropout=dropout, alpha=alpha).to(device)
    print(f"Parameters: {count_parameters(model):,}")

    # --- Loss + Optimizer ---
    loss_fn   = MTLLoss(alpha=alpha, binary_pos_weight=w_bin.to(device), type_class_weights=w_type.to(device))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # --- History ---
    history = {
        "train_loss": [], "val_loss": [],
        "train_binary_loss": [], "val_binary_loss": [],
        "train_type_loss":   [], "val_type_loss":   [],
        "val_binary_f1": [],
        "val_type_f1_macro": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")

    for epoch in range(1, epochs + 1):
        # Train
        tr_loss, tr_bl, tr_tl = train_one_epoch(model, train_loader, loss_fn, optimizer, device)

        # Validate
        val_loss, bin_preds, bin_targets, type_preds, type_targets = evaluate(
            model, val_loader, loss_fn, device
        )
        val_metrics = compute_metrics(bin_preds, bin_targets, type_preds, type_targets)

        # Sub-losses on val
        model.eval()
        val_bl_total, val_tl_total = 0.0, 0.0
        with torch.no_grad():
            for X, b_true, t_true in val_loader:
                X, b_true, t_true = X.to(device), b_true.to(device), t_true.to(device)
                bl_logits, tl_logits = model(X)
                _, bl, tl = loss_fn(bl_logits, tl_logits, b_true, t_true)
                val_bl_total += bl.item()
                val_tl_total += tl.item()
        val_bl = val_bl_total / len(val_loader)
        val_tl = val_tl_total / len(val_loader)

        scheduler.step(val_loss)

        # Log
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_binary_loss"].append(tr_bl)
        history["val_binary_loss"].append(val_bl)
        history["train_type_loss"].append(tr_tl)
        history["val_type_loss"].append(val_tl)
        history["val_binary_f1"].append(val_metrics["binary_f1"])
        history["val_type_f1_macro"].append(val_metrics["type_f1_macro"])

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train Loss: {tr_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Binary F1: {val_metrics['binary_f1']:.4f} | "
            f"Val Type F1 (macro): {val_metrics['type_f1_macro']:.4f}"
        )

        # Checkpoint + early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            save_checkpoint(model, optimizer, epoch, val_metrics, best_checkpoint_path)
            print(f"  -> Saved best model (val_loss={val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # --- Save loss curves ---
    os.makedirs(results_dir, exist_ok=True)
    plot_loss_curves(history, save_path=os.path.join(results_dir, "loss_curves.png"))
    print(f"\nTraining complete. Best checkpoint: {best_checkpoint_path}")
    print("Run `python src/test.py` to evaluate on the test set.")

    return history


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train MTL predictive maintenance model")
    parser.add_argument("--data",           type=str,   default=os.path.join("data", "ai4i2020.csv"))
    parser.add_argument("--epochs",         type=int,   default=50)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--batch-size",     type=int,   default=64)
    parser.add_argument("--alpha",          type=float, default=0.5)
    parser.add_argument("--dropout",        type=float, default=0.3)
    parser.add_argument("--patience",       type=int,   default=10)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--checkpoint-dir", type=str,   default="checkpoints")
    parser.add_argument("--results-dir",    type=str,   default="results")
    parser.add_argument("--no-early-stopping", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_early_stopping:
        args.patience = args.epochs

    train(
        data_path      = args.data,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        alpha          = args.alpha,
        dropout        = args.dropout,
        patience       = args.patience,
        seed           = args.seed,
        checkpoint_dir = args.checkpoint_dir,
        results_dir    = args.results_dir,
    )
