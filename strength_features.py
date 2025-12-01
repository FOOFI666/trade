"""Relative strength features against BTC benchmark."""

import pandas as pd


def add_return(df: pd.DataFrame, window: int, col_name: str) -> pd.DataFrame:
    """Add percentage return over the specified window."""
    shifted = df["close"].shift(window)
    df[col_name] = (df["close"] - shifted) / shifted
    return df


def add_relative_strength(
    df_asset: pd.DataFrame, df_btc: pd.DataFrame, window: int, col_name: str
) -> pd.DataFrame:
    """Compute asset's relative strength versus BTC over the given window."""
    merged = pd.merge(
        df_asset,
        df_btc[["timestamp", "close"]].rename(columns={"close": "close_btc"}),
        on="timestamp",
        how="inner",
        suffixes=("", "_btc"),
    )
    merged["return_asset"] = (merged["close"] - merged["close"].shift(window)) / merged[
        "close"
    ].shift(window)
    merged["return_btc"] = (
        merged["close_btc"] - merged["close_btc"].shift(window)
    ) / merged["close_btc"].shift(window)
    merged[col_name] = merged["return_asset"] - merged["return_btc"]
    df_asset = merged.drop(columns=["close_btc", "return_asset", "return_btc"])
    return df_asset
