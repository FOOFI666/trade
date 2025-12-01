"""Neural network inference utilities for entry prediction."""

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

import config

logger = logging.getLogger("nn_entry_model")


class EntryNN(nn.Module):
    def __init__(self, input_features: int, hidden_sizes=config.NN_HIDDEN_SIZES):
        super().__init__()
        layers = []
        in_dim = input_features * config.NN_WINDOW_SIZE
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, window, feats = x.shape
        x = x.view(batch_size, window * feats)
        return self.net(x).squeeze(-1)


_nn_model: Optional[EntryNN] = None
_nn_device = torch.device("cpu")


def load_entry_model() -> EntryNN:
    global _nn_model
    if _nn_model is None:
        model = EntryNN(input_features=len(config.NN_FEATURE_COLUMNS))
        state = torch.load(config.NN_MODEL_PATH, map_location=_nn_device)
        model.load_state_dict(state)
        model.to(_nn_device)
        model.eval()
        _nn_model = model
        logger.info(
            "Loaded NN entry model from %s with layers %s",
            config.NN_MODEL_PATH,
            config.NN_HIDDEN_SIZES,
        )
    return _nn_model


def predict_entry_proba(window_features: np.ndarray) -> float:
    """
    window_features: np.ndarray shape [window_size, num_features]
    возвращает float в диапазоне [0, 1]
    """
    model = load_entry_model()
    x = torch.from_numpy(window_features).float().unsqueeze(0).to(_nn_device)  # [1, W, F]
    with torch.no_grad():
        prob = model(x).item()
    return prob
