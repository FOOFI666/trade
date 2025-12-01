"""Volume and candle shape feature engineering."""
import pandas as pd
from utils import safe_divide, sma


def add_basic_candle_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add body, candle_range, and body_ratio columns."""
    df["body"] = (df["close"] - df["open"]).abs()
    df["candle_range"] = df["high"] - df["low"]
    df["body_ratio"] = safe_divide(df["body"], df["candle_range"])
    return df


def add_volume_relative(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Add rolling mean volume and relative volume ratio."""
    ma_col = f"vol_ma_{window}"
    df[ma_col] = sma(df["volume"], window)
    df[col_name] = safe_divide(df["volume"], df[ma_col])
    return df


def add_taker_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add taker buy metrics when the necessary columns are present."""
    if {"taker_buy_base", "volume"}.issubset(df.columns):
        df["buy_ratio"] = safe_divide(df["taker_buy_base"], df["volume"])
        df["taker_delta"] = df["taker_buy_base"] - (df["volume"] - df["taker_buy_base"])
    return df
