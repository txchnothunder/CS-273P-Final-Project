"""
dataset.py

Dataset and DataLoader for the AI4I 2020 Predictive Maintenance dataset.

Targets:
  - binary_label : 0/1  (Machine failure)
  - type_label   : 0–5  (0 = No Failure, 1 = TWF, 2 = HDF, 3 = PWF, 4 = OSF, 5 = RNF)
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

FAILURE_COLS = ["TWF", "HDF", "PWF", "OSF", "RNF"]
FAILURE_TO_IDX = {col: idx + 1 for idx, col in enumerate(FAILURE_COLS)}  # 1-indexed; 0 = no failure

CONTINUOUS_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]

TYPE_MAP = {"L": 0, "M": 1, "H": 2}


# ------------------------------------------------------------------
# Label construction
# ------------------------------------------------------------------

def build_labels(df: pd.DataFrame):
    """
    Returns:
        binary : np.ndarray of shape (N,) — 1 if machine failed, else 0
        failure_type : np.ndarray of shape (N,) — class index 0–5
    """
    binary = df["Machine failure"].values.astype(np.int64)

    failure_type = np.zeros(len(df), dtype=np.int64)  # default: no failure
    for col, idx in FAILURE_TO_IDX.items():
        mask = df[col].values == 1
        failure_type[mask] = idx

    return binary, failure_type


# ------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------

class MaintenanceDataset(Dataset):
    """
    PyTorch Dataset for the AI4I 2020 Predictive Maintenance data.

    Args:
        features : np.ndarray of shape (N, D) — scaled + encoded features
        binary   : np.ndarray of shape (N,)   — binary failure label
        failure_type : np.ndarray of shape (N,) — 6-class failure type label
    """

    def __init__(self, features: np.ndarray, binary: np.ndarray, failure_type: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.binary = torch.tensor(binary, dtype=torch.float32)       # BCEWithLogitsLoss expects float
        self.failure_type = torch.tensor(failure_type, dtype=torch.long)  # CrossEntropyLoss expects long

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.binary[idx], self.failure_type[idx]


# ------------------------------------------------------------------
# Data loading pipeline
# ------------------------------------------------------------------

def load_data(
    csv_path: str,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
):
    """
    Loads and preprocesses the AI4I 2020 dataset.

    Splits are stratified on binary failure label to preserve the ~3.4% failure rate.

    Args:
        csv_path     : path to ai4i2020.csv
        test_size    : fraction of data held out for test
        val_size     : fraction of data held out for validation (taken from train)
        random_state : seed for reproducibility

    Returns:
        train_dataset, val_dataset, test_dataset : MaintenanceDataset instances
        scaler : fitted StandardScaler (save this if you want to inverse-transform later)
        class_weights_binary : tensor for weighted BCE loss
        class_weights_type   : tensor for weighted CrossEntropy loss
    """
    df = pd.read_csv(csv_path)

    # --- Encode categorical type feature ---
    df["Type_enc"] = df["Type"].map(TYPE_MAP).astype(np.float32)

    # --- Build feature matrix ---
    feature_cols = CONTINUOUS_COLS + ["Type_enc"]
    X = df[feature_cols].values.astype(np.float32)

    # --- Build labels ---
    binary, failure_type = build_labels(df)

    # --- Stratified train / val+test split ---
    X_train, X_tmp, y_bin_train, y_bin_tmp, y_type_train, y_type_tmp = train_test_split(
        X, binary, failure_type,
        test_size=(test_size + val_size),
        stratify=binary,
        random_state=random_state,
    )

    # Split tmp into val and test
    relative_test = test_size / (test_size + val_size)
    X_val, X_test, y_bin_val, y_bin_test, y_type_val, y_type_test = train_test_split(
        X_tmp, y_bin_tmp, y_type_tmp,
        test_size=relative_test,
        stratify=y_bin_tmp,
        random_state=random_state,
    )

    # --- Scale continuous features only (columns 0..4), leave Type_enc as-is ---
    scaler = StandardScaler()
    X_train[:, :5] = scaler.fit_transform(X_train[:, :5])
    X_val[:, :5]   = scaler.transform(X_val[:, :5])
    X_test[:, :5]  = scaler.transform(X_test[:, :5])

    # --- Compute class weights for imbalanced learning ---
    # Binary: weight for positive class = N_neg / N_pos
    n_pos = y_bin_train.sum()
    n_neg = len(y_bin_train) - n_pos
    class_weights_binary = torch.tensor(n_neg / n_pos, dtype=torch.float32)

    # 6-class: inverse frequency weights
    class_weights_type = _compute_class_weights(y_type_train, num_classes=6)

    # --- Build datasets ---
    train_dataset = MaintenanceDataset(X_train, y_bin_train, y_type_train)
    val_dataset   = MaintenanceDataset(X_val,   y_bin_val,   y_type_val)
    test_dataset  = MaintenanceDataset(X_test,  y_bin_test,  y_type_test)

    return train_dataset, val_dataset, test_dataset, scaler, class_weights_binary, class_weights_type


def _compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.where(counts == 0, 1, counts)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------
# DataLoader factory
# ------------------------------------------------------------------

def get_dataloaders(
    train_dataset: MaintenanceDataset,
    val_dataset: MaintenanceDataset,
    test_dataset: MaintenanceDataset,
    batch_size: int = 64,
    num_workers: int = 0,
):
    """
    Wraps datasets in DataLoaders.

    Args:
        batch_size  : samples per batch
        num_workers : set to 0 for Windows compatibility; increase on Linux/Mac

    Returns:
        train_loader, val_loader, test_loader
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
