"""Utilities to construct pump/no-pump datasets from OHLCV data."""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
import pandas as pd


def _determine_label(
    future_slice: pd.DataFrame,
    entry_price: float,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> int:
    tp_price = entry_price * (1 + take_profit_pct)
    sl_price = entry_price * (1 - stop_loss_pct)

    for _, row in future_slice.iterrows():
        if row.get("high", entry_price) >= tp_price:
            return 1
        if row.get("low", entry_price) <= sl_price:
            return 0

    if future_slice.get("high", pd.Series(dtype=float)).max() >= tp_price:
        return 1
    return 0


def build_pump_dataset(
    df: pd.DataFrame,
    window_size: int,
    future_horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    feature_columns: Sequence[str],
    only_pre_pump: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Построение датасета для классификации пампа.

    Возвращает массив признаков формы [N, window_size, num_features]
    и вектор меток [N], где метка = 1, если TP достигнут раньше SL в горизонте future_horizon.
    """

    if df is None or df.empty:
        return np.empty((0, window_size, len(feature_columns))), np.empty((0,))

    feature_cols_present = [col for col in feature_columns if col in df.columns]
    missing_cols = [col for col in feature_columns if col not in df.columns]

    work_df = df.reset_index(drop=True)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for end_idx in range(window_size, len(work_df) - future_horizon):
        window_df = work_df.iloc[end_idx - window_size : end_idx]
        label_row = window_df.iloc[-1]

        if only_pre_pump and not bool(label_row.get("is_pre_pump", False)):
            continue

        future_slice = work_df.iloc[end_idx : end_idx + future_horizon]
        entry_price = float(label_row.get("close", float("nan")))
        if pd.isna(entry_price):
            continue

        label = _determine_label(
            future_slice, entry_price, take_profit_pct, stop_loss_pct
        )

        window_features = window_df[feature_cols_present].copy()
        for col in missing_cols:
            window_features[col] = 0.0
        window_features = window_features[feature_columns].fillna(0.0).to_numpy()

        X_list.append(window_features)
        y_list.append(label)

    if not X_list:
        return np.empty((0, window_size, len(feature_columns))), np.empty((0,))

    X = np.stack(X_list, axis=0)
    y = np.asarray(y_list, dtype=np.float32)
    return X, y
