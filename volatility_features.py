"""Volatility-related feature engineering."""
import pandas as pd
from utils import rolling_percentile, sma


def add_true_range(df: pd.DataFrame) -> pd.DataFrame:
    """Add True Range column to the DataFrame."""
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = tr
    return df


def add_atr(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Add Average True Range column based on a rolling window."""
    if "true_range" not in df.columns:
        add_true_range(df)
    df[col_name] = df["true_range"].rolling(window=window, min_periods=window).mean()
    return df


def add_bollinger_bandwidth(df: pd.DataFrame, window: int, k: float, col_name: str) -> pd.DataFrame:
    """Compute Bollinger Bandwidth and append it as a column."""
    ma = sma(df["close"], window)
    std = df["close"].rolling(window=window, min_periods=window).std()
    upper = ma + k * std
    lower = ma - k * std
    bandwidth = (upper - lower) / ma
    df[col_name] = bandwidth
    return df


def add_bbw_percentile(df: pd.DataFrame, bbw_col: str, window: int, col_name: str) -> pd.DataFrame:
    """Add rolling percentile of Bollinger Bandwidth."""
    df[col_name] = rolling_percentile(df[bbw_col], window=window)
    return df
