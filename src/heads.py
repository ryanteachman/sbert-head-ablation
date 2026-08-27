"""Phase 4 — classifier heads.

Linear probe (PRIMARY) and a single fixed MLP (SECONDARY, robustness only).
Both take a standardized feature vector and emit ``C`` logits for softmax
cross-entropy. See PROTOCOL.md section 8.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearHead(nn.Module):
    """PRIMARY. Single linear layer — the SBERT paper's classification objective."""

    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """SECONDARY. One hidden layer, fixed width — never tuned or swept.

    ``d_in -> hidden -> ReLU -> Dropout -> n_classes``. Hidden width is fixed
    regardless of ``d_in`` so the MLP is not itself a capacity confound.
    """

    def __init__(self, d_in: int, n_classes: int, hidden: int = 256,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_head(kind: str, d_in: int, n_classes: int, cfg: dict) -> nn.Module:
    heads = cfg["heads"]
    if kind == "linear":
        return LinearHead(d_in, n_classes)
    if kind == "mlp":
        m = heads["mlp"]
        return MLPHead(d_in, n_classes, hidden=m["hidden"], dropout=m["dropout"])
    raise KeyError(kind)


def weight_decay_for(kind: str, cfg: dict) -> float:
    if kind == "linear":
        return float(cfg["train"]["weight_decay_linear"])
    return float(cfg["heads"]["mlp"]["weight_decay"])
