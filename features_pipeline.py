"""Shared feature engineering pipeline for scanner and backtests."""

from __future__ import annotations

import pandas as pd

import config
from funding_features import add_funding_zscore, add_open_interest_change
from strength_features import add_relative_strength
from volatility_features import (
    add_atr,
    add_bbw_percentile,
    add_bollinger_bandwidth,
    add_true_range,
)
from volume_features import add_basic_candle_metrics, add_taker_metrics, add_volume_relative


def update_features(df: pd.DataFrame, btc_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Apply feature computations used by both realtime scanner and offline backtests."""

    if df is None or df.empty:
        return df

    df = df.copy()

    add_true_range(df)
    add_atr(df, config.ATR_WINDOW, "atr_60")
    add_atr(df, config.VOL_RATIO_30_WINDOW, "atr_30")

    add_bollinger_bandwidth(df, config.BBW_WINDOW, 2.0, "bbw")
    add_bbw_percentile(df, "bbw", config.BBW_PERCENTILE_WINDOW, "bbw_percentile")

    add_basic_candle_metrics(df)
    add_volume_relative(df, config.VOL_RATIO_60_WINDOW, "vol_ma_60", "vol_ratio_60")
    add_volume_relative(df, config.VOL_RATIO_30_WINDOW, "vol_ma_30", "vol_ratio_30")
    add_taker_metrics(df)

    if btc_df is not None and not btc_df.empty:
        df = add_relative_strength(df, btc_df, config.REL_STRENGTH_WINDOW, "rel_strength_180")

    if {"funding_rate"}.issubset(df.columns):
        add_funding_zscore(df, config.BODY_RATIO_MEAN_WINDOW)
    if "open_interest" in df.columns:
        add_open_interest_change(df, config.VOL_RATIO_60_WINDOW, "oi_change_60")

    return df
