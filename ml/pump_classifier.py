"""PyTorch pump / no-pump classifier."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class PumpClassifier(nn.Module):
    def __init__(
        self,
        window_size: int,
        num_features: int,
        hidden_sizes: Sequence[int] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        input_dim = window_size * num_features
        for hidden in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            input_dim = hidden
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window, feats = x.shape
        flattened = x.view(batch_size, window * feats)
        logits = self.network(flattened)
        return self.sigmoid(logits).squeeze(-1)
