"""
smoke_test.py

Run this from the project root to verify the full pipeline works:
    python smoke_test.py

Expected output: all checks print OK and final "ALL CHECKS PASSED".
Covers: dataset, model, evaluate, utils, train
"""

import sys
import os
sys.path.insert(0, "src")

print("=" * 55)
print("CS273P Final Project — Smoke Test")
print("=" * 55)

# ------------------------------------------------------------------
print("\n[1] Imports...")
# ------------------------------------------------------------------
try:
    import torch
    from dataset  import load_data, get_dataloaders
    from model    import MTLModel, MTLLoss, count_parameters
    from evaluate import evaluate, compute_metrics, print_report
    from utils    import set_seed, save_checkpoint, load_checkpoint, get_device
    print("    OK")
except ImportError as e:
    print(f"    FAILED: {e}")
    print("    Make sure you ran: pip install -r requirements.txt")
    sys.exit(1)

# ------------------------------------------------------------------
print("\n[2] utils.py — seed + device...")
# ------------------------------------------------------------------
set_seed(42)
device = get_device()
print(f"    Seed set. Device: {device}")

# ------------------------------------------------------------------
print("\n[3] dataset.py — data pipeline...")
# ------------------------------------------------------------------
DATA_PATH = os.path.join("data", "ai4i2020.csv")
if not os.path.exists(DATA_PATH):
    print(f"    FAILED: {DATA_PATH} not found.")
    print("    Place ai4i2020.csv inside the data/ folder.")
    sys.exit(1)

train_ds, val_ds, test_ds, scaler, w_bin, w_type = load_data(DATA_PATH)
assert len(train_ds) + len(val_ds) + len(test_ds) == 10000
print(f"    Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
print(f"    Binary pos_weight  : {w_bin.item():.2f}")
print(f"    Type class weights : {[round(w, 3) for w in w_type.tolist()]}")

x0, b0, t0 = train_ds[0]
assert x0.shape == (6,)
assert b0.dtype == torch.float32
assert t0.dtype == torch.int64
print("    Sample shapes + dtypes OK")

# ------------------------------------------------------------------
print("\n[4] dataset.py — DataLoaders...")
# ------------------------------------------------------------------
train_loader, val_loader, test_loader = get_dataloaders(
    train_ds, val_ds, test_ds, batch_size=64
)
X_batch, b_batch, t_batch = next(iter(train_loader))
assert X_batch.shape[1] == 6
print(f"    Batch — X: {tuple(X_batch.shape)}, binary: {tuple(b_batch.shape)}, type: {tuple(t_batch.shape)}")

# ------------------------------------------------------------------
print("\n[5] model.py — forward pass...")
# ------------------------------------------------------------------
set_seed(42)
model = MTLModel().to(device)
print(f"    Trainable parameters: {count_parameters(model):,}")

model.eval()
with torch.no_grad():
    X_dev = X_batch.to(device)
    bin_logits, type_logits = model(X_dev)

assert bin_logits.shape  == (X_batch.shape[0],)
assert type_logits.shape == (X_batch.shape[0], 6)
print(f"    binary_logits: {tuple(bin_logits.shape)}  type_logits: {tuple(type_logits.shape)}")

# ------------------------------------------------------------------
print("\n[6] model.py — MTLLoss...")
# ------------------------------------------------------------------
loss_fn = MTLLoss(alpha=0.5, binary_pos_weight=w_bin.to(device), type_class_weights=w_type.to(device))
total, bl, tl = loss_fn(bin_logits, type_logits, b_batch.to(device), t_batch.to(device))
assert total.item() > 0
assert not torch.isnan(total)
print(f"    total: {total.item():.4f}  binary: {bl.item():.4f}  type: {tl.item():.4f}")

# ------------------------------------------------------------------
print("\n[7] model.py — backward pass + gradients...")
# ------------------------------------------------------------------
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer.zero_grad()
bin_logits, type_logits = model(X_dev)
total, _, _ = loss_fn(bin_logits, type_logits, b_batch.to(device), t_batch.to(device))
total.backward()
optimizer.step()

grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
assert all(g == g for g in grad_norms), "NaN gradients detected"
print(f"    Gradients OK — mean grad norm: {sum(grad_norms)/len(grad_norms):.6f}")

# ------------------------------------------------------------------
print("\n[8] evaluate.py — evaluate() over val loader...")
# ------------------------------------------------------------------
val_loss, bin_preds, bin_targets, type_preds, type_targets = evaluate(
    model, val_loader, loss_fn, device
)
assert len(bin_preds) == len(val_ds)
assert len(type_preds) == len(val_ds)
print(f"    Val loss: {val_loss:.4f}")
print(f"    Predictions — binary: {bin_preds.shape}  type: {type_preds.shape}")

# ------------------------------------------------------------------
print("\n[9] evaluate.py — compute_metrics()...")
# ------------------------------------------------------------------
metrics = compute_metrics(bin_preds, bin_targets, type_preds, type_targets)
required_keys = [
    "binary_accuracy", "binary_f1", "binary_precision", "binary_recall",
    "type_accuracy", "type_f1_macro", "type_f1_weighted",
]
for key in required_keys:
    assert key in metrics, f"Missing metric: {key}"
    assert 0.0 <= metrics[key] <= 1.0, f"Metric out of range: {key}={metrics[key]}"
print(f"    binary_f1: {metrics['binary_f1']:.4f}  type_f1_macro: {metrics['type_f1_macro']:.4f}")
print("    All required metric keys present and in [0, 1]")

# ------------------------------------------------------------------
print("\n[10] utils.py — save + load checkpoint...")
# ------------------------------------------------------------------
ckpt_path = os.path.join("checkpoints", "_smoke_test_ckpt.pt")
os.makedirs("checkpoints", exist_ok=True)
save_checkpoint(model, optimizer, epoch=1, metrics=metrics, path=ckpt_path)
assert os.path.exists(ckpt_path), "Checkpoint file not created"

model2 = MTLModel().to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
epoch_loaded, metrics_loaded = load_checkpoint(model2, optimizer2, ckpt_path, device)
assert epoch_loaded == 1
assert "binary_f1" in metrics_loaded

for p1, p2 in zip(model.parameters(), model2.parameters()):
    assert torch.allclose(p1, p2), "Loaded weights don't match saved weights"

os.remove(ckpt_path)
print("    Checkpoint saved + loaded + weights verified OK")

# ------------------------------------------------------------------
print("\n[11] train.py — 3-epoch mini training run...")
# ------------------------------------------------------------------
from train import train

_, test_metrics, history = train(
    data_path="data/ai4i2020.csv",
    epochs=3,
    lr=1e-3,
    batch_size=64,
    alpha=0.5,
    patience=3,
    seed=42,
    checkpoint_dir="checkpoints",
    results_dir="results",
)

assert "binary_f1"     in test_metrics
assert "type_f1_macro" in test_metrics
assert len(history["train_loss"]) == 3
assert os.path.exists(os.path.join("checkpoints", "best_model.pt"))
print("    3-epoch run completed, checkpoint exists, history length OK")

# ------------------------------------------------------------------
print("\n" + "=" * 55)
print("ALL CHECKS PASSED")
print("=" * 55)
