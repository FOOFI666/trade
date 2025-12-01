"""Data loading helpers for candle, BTC benchmark, and derivatives data."""
from pathlib import Path
from typing import Dict, Iterable, Union
import pandas as pd


CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "taker_buy_base",
    "taker_buy_quote",
]


FUNDING_COLUMNS = ["funding_rate", "open_interest"]


def _read_csv(path: Union[str, Path]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def load_symbol_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load OHLCV-like data for a single symbol from CSV."""
    return _read_csv(path)


def load_multiple_symbols(paths: Dict[str, Union[str, Path]]) -> Dict[str, pd.DataFrame]:
    """Load multiple symbols keyed by symbol name."""
    return {symbol: load_symbol_data(p) for symbol, p in paths.items()}


def load_btc_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load BTC benchmark data used for relative strength calculations."""
    return _read_csv(path)


def merge_funding_data(df: pd.DataFrame, funding_path: Union[str, Path]) -> pd.DataFrame:
    """Merge funding/open interest data into the main DataFrame based on timestamp."""
    funding_df = _read_csv(funding_path)
    merged = pd.merge(df, funding_df[FUNDING_COLUMNS + ["timestamp"]], on="timestamp", how="left")
    return merged


def ensure_columns(df: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    """Ensure required columns exist, filling missing ones with NaN."""
    for col in required:
        if col not in df.columns:
            df[col] = pd.NA
    return df
