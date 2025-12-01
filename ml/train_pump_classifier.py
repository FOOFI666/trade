"""Training script for the pump/no-pump classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import config
from ml.dataset_builder import build_pump_dataset
from ml.pump_classifier import PumpClassifier


def _load_frame(path: str) -> pd.DataFrame:
    file_path = Path(path)
    if file_path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(file_path)
    return pd.read_csv(file_path)


def _make_dataloaders(
    X: np.ndarray, y: np.ndarray, batch_size: int, val_split: float
) -> tuple[DataLoader, DataLoader]:
    split_idx = int(len(X) * (1 - val_split))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def _evaluate(model: PumpClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.BCELoss()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * len(xb)
    return total_loss / max(len(loader.dataset), 1)


def train(
    data_path: str,
    window_size: int,
    future_horizon: int,
    take_profit_pct: float,
    stop_loss_pct: float,
    feature_columns: Sequence[str],
    only_pre_pump: bool,
    batch_size: int,
    epochs: int,
    lr: float,
    val_split: float,
    device: torch.device,
):
    df = _load_frame(data_path)
    X, y = build_pump_dataset(
        df,
        window_size=window_size,
        future_horizon=future_horizon,
        take_profit_pct=take_profit_pct,
        stop_loss_pct=stop_loss_pct,
        feature_columns=feature_columns,
        only_pre_pump=only_pre_pump,
    )

    if len(X) == 0:
        raise ValueError("Dataset is empty after applying filters. Check data and parameters.")

    train_loader, val_loader = _make_dataloaders(X, y, batch_size=batch_size, val_split=val_split)

    model = PumpClassifier(
        window_size=window_size,
        num_features=len(feature_columns),
        hidden_sizes=config.NN_HIDDEN_SIZES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(xb)

        train_loss = running_loss / len(train_loader.dataset)
        val_loss = _evaluate(model, val_loader, device)
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    Path(config.NN_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.NN_MODEL_PATH)
    print(f"Model saved to {config.NN_MODEL_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Train pump/no-pump classifier")
    parser.add_argument("data_path", type=str, help="Path to CSV or Parquet with features")
    parser.add_argument("--window_size", type=int, default=config.NN_WINDOW_SIZE)
    parser.add_argument("--future_horizon", type=int, default=60)
    parser.add_argument("--take_profit_pct", type=float, default=0.02)
    parser.add_argument("--stop_loss_pct", type=float, default=0.01)
    parser.add_argument("--feature_columns", type=str, nargs="*", default=config.NN_FEATURE_COLUMNS)
    parser.add_argument("--only_pre_pump", action="store_true", default=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        data_path=args.data_path,
        window_size=args.window_size,
        future_horizon=args.future_horizon,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        feature_columns=args.feature_columns,
        only_pre_pump=args.only_pre_pump,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        val_split=args.val_split,
        device=device,
    )


if __name__ == "__main__":
    main()
