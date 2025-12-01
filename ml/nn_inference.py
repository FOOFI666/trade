"""Runtime inference helper for pump classifier."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

import config
from ml.pump_classifier import PumpClassifier

_model: Optional[PumpClassifier] = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_model() -> PumpClassifier:
    global _model
    if _model is None:
        model = PumpClassifier(
            window_size=config.NN_WINDOW_SIZE,
            num_features=len(config.NN_FEATURE_COLUMNS),
            hidden_sizes=config.NN_HIDDEN_SIZES,
        )
        state = torch.load(Path(config.NN_MODEL_PATH), map_location=_device)
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model = model
    return _model


def predict_pump_proba(window_df) -> float:
    """Return pump probability for the provided window dataframe."""
    if window_df is None or len(window_df) < config.NN_WINDOW_SIZE:
        return 0.0

    model = _load_model()
    features = []
    for col in config.NN_FEATURE_COLUMNS:
        if col in window_df.columns:
            features.append(window_df[col].fillna(0.0).values)
        else:
            features.append(np.zeros(len(window_df)))

    window_features = np.stack(features, axis=1).astype(np.float32)
    x = torch.from_numpy(window_features).unsqueeze(0).to(_device)

    with torch.no_grad():
        prob = model(x).item()
    return float(prob)
