"""Feature correlation heatmap for pump dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import config


def _load_frame(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _build_samples(
    df: pd.DataFrame,
    window_size: int,
    future_horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    feature_columns: Sequence[str],
    only_pre_pump: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for end_idx in range(window_size, len(df) - future_horizon):
        window_df = df.iloc[end_idx - window_size : end_idx]
        label_row = window_df.iloc[-1]
        if only_pre_pump and not bool(label_row.get("is_pre_pump", False)):
            continue

        future_slice = df.iloc[end_idx : end_idx + future_horizon]
        entry_price = float(label_row.get("close", float("nan")))
        if pd.isna(entry_price):
            continue

        tp_price = entry_price * (1 + take_profit_pct)
        sl_price = entry_price * (1 - stop_loss_pct)
        label = 0
        for _, row in future_slice.iterrows():
            if row.get("high", entry_price) >= tp_price:
                label = 1
                break
            if row.get("low", entry_price) <= sl_price:
                label = 0
                break
        future_max_up = future_slice.get("high", pd.Series(dtype=float)).max()
        future_max_up_move = (future_max_up - entry_price) / entry_price if pd.notna(future_max_up) else 0.0

        feature_values = {col: label_row.get(col, 0.0) for col in feature_columns}
        feature_values.update({
            "y": label,
            "future_max_up_move": future_max_up_move,
        })
        rows.append(feature_values)

    return pd.DataFrame(rows)


def plot_correlations(dataset: pd.DataFrame, output_path: str):
    corr_targets = dataset.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_targets, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature correlations with targets")
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Saved heatmap to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot feature correlation heatmap")
    parser.add_argument("data_path", type=str, help="Path to raw candles or enriched dataset")
    parser.add_argument("--window_size", type=int, default=config.NN_WINDOW_SIZE)
    parser.add_argument("--future_horizon", type=int, default=60)
    parser.add_argument("--take_profit_pct", type=float, default=0.02)
    parser.add_argument("--stop_loss_pct", type=float, default=0.01)
    parser.add_argument("--feature_columns", type=str, nargs="*", default=config.NN_FEATURE_COLUMNS)
    parser.add_argument("--only_pre_pump", action="store_true", default=False)
    parser.add_argument("--output", type=str, default="feature_correlations.png")
    args = parser.parse_args()

    df = _load_frame(args.data_path)
    samples = _build_samples(
        df,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        feature_columns=args.feature_columns,
        only_pre_pump=args.only_pre_pump,
    )

    if samples.empty:
        raise ValueError("No samples generated for correlation analysis")

    plot_correlations(samples, args.output)


if __name__ == "__main__":
    main()
