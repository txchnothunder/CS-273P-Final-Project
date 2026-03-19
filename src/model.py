"""
model.py

Multi-Task Learning architecture for predictive maintenance.

Architecture:
    Shared backbone  : MLP with residual connections + BatchNorm + Dropout
    Head 1 (binary)  : single logit for binary failure prediction
    Head 2 (type)    : 6-class logits for failure type classification
                       (0 = No Failure, 1 = TWF, 2 = HDF, 3 = PWF, 4 = OSF, 5 = RNF)

Loss:
    total_loss = alpha * BCE_loss + (1 - alpha) * CE_loss
"""

import torch
import torch.nn as nn


# ------------------------------------------------------------------
# Residual Block
# ------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """
    Single residual block: Linear -> BN -> ReLU -> Linear -> BN, plus skip connection.

    If input and output dims differ, a linear projection is applied to the skip.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.block(x) + self.skip(x))


# ------------------------------------------------------------------
# Task Head
# ------------------------------------------------------------------

class TaskHead(nn.Module):
    """
    Small MLP head for a single task.

    Args:
        in_dim    : dimension coming from shared backbone
        hidden_dim: intermediate hidden size
        out_dim   : number of output logits
        dropout   : dropout rate
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ------------------------------------------------------------------
# Full MTL Model
# ------------------------------------------------------------------

class MTLModel(nn.Module):
    """
    Multi-Task Learning model with shared backbone and two task heads.

    Args:
        input_dim      : number of input features (default 6: 5 continuous + 1 type encoding)
        backbone_dims  : list of hidden sizes for backbone residual blocks
        head_hidden_dim: hidden size inside each task head
        dropout        : dropout rate applied throughout
        alpha          : weight on binary loss in combined loss (1 - alpha for type loss)
    """

    def __init__(
        self,
        input_dim: int = 6,
        backbone_dims: list = [64, 128, 64],
        head_hidden_dim: int = 32,
        dropout: float = 0.3,
        alpha: float = 0.5,
    ):
        super().__init__()
        self.alpha = alpha

        # --- Shared backbone ---
        layers = []
        in_dim = input_dim
        for out_dim in backbone_dims:
            layers.append(ResidualBlock(in_dim, out_dim, dropout=dropout))
            in_dim = out_dim
        self.backbone = nn.Sequential(*layers)

        # --- Task heads ---
        self.binary_head = TaskHead(in_dim, head_hidden_dim, out_dim=1,  dropout=dropout)
        self.type_head   = TaskHead(in_dim, head_hidden_dim, out_dim=6,  dropout=dropout)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x : (batch_size, input_dim)

        Returns:
            binary_logits : (batch_size,)   — squeeze to 1D for BCEWithLogitsLoss
            type_logits   : (batch_size, 6) — raw logits for CrossEntropyLoss
        """
        shared = self.backbone(x)
        binary_logits = self.binary_head(shared).squeeze(1)
        type_logits   = self.type_head(shared)
        return binary_logits, type_logits


# ------------------------------------------------------------------
# Combined Loss
# ------------------------------------------------------------------

class MTLLoss(nn.Module):
    """
    Combined loss for both tasks.

    Args:
        alpha               : weight on binary BCE loss (1 - alpha on type CE loss)
        binary_pos_weight   : scalar tensor for class imbalance in binary task
        type_class_weights  : (6,) tensor for class imbalance in type task
    """

    def __init__(
        self,
        alpha: float = 0.5,
        binary_pos_weight: torch.Tensor = None,
        type_class_weights: torch.Tensor = None,
    ):
        super().__init__()
        self.alpha = alpha
        self.binary_loss_fn = nn.BCEWithLogitsLoss(pos_weight=binary_pos_weight)
        self.type_loss_fn   = nn.CrossEntropyLoss(weight=type_class_weights)

    def forward(
        self,
        binary_logits: torch.Tensor,
        type_logits: torch.Tensor,
        binary_targets: torch.Tensor,
        type_targets: torch.Tensor,
    ):
        """
        Returns:
            total_loss   : combined scalar loss
            binary_loss  : scalar BCE loss (for logging)
            type_loss    : scalar CE loss  (for logging)
        """
        binary_loss = self.binary_loss_fn(binary_logits, binary_targets)
        type_loss   = self.type_loss_fn(type_logits, type_targets)
        total_loss  = self.alpha * binary_loss + (1 - self.alpha) * type_loss
        return total_loss, binary_loss, type_loss


# ------------------------------------------------------------------
# Model summary utility
# ------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
