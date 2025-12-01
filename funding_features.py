"""Funding rate and open interest feature engineering."""
import pandas as pd
from utils import sma


def add_funding_normalized(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Add funding moving average, std, and z-score columns."""
    df["funding_ma"] = sma(df["funding_rate"], window)
    df["funding_std"] = df["funding_rate"].rolling(window=window, min_periods=window).std()
    df["funding_zscore"] = (df["funding_rate"] - df["funding_ma"]) / df["funding_std"]
    return df


def add_open_interest_change(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Add relative change in open interest over a rolling window."""
    shifted = df["open_interest"].shift(window)
    df[col_name] = (df["open_interest"] - shifted) / shifted
    return df
