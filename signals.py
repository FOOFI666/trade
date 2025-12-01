"""Signal computation for pre-pump detection and long entries."""
import pandas as pd
import config


def compute_pre_pump_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute composite pre-pump score and boolean flag."""
    score = pd.Series(0, index=df.index, dtype=float)

    score += (df.get("bbw_percentile", pd.Series(dtype=float)) < config.BBW_PERCENTILE_THRESHOLD).fillna(
        False
    )
    score += (
        (df.get("vol_ratio_60", pd.Series(dtype=float)) >= config.VOL_RATIO_60_MIN)
        & (df.get("vol_ratio_60", pd.Series(dtype=float)) <= config.VOL_RATIO_60_MAX)
    ).fillna(False)

    body_ratio_mean = df.get("body_ratio", pd.Series(dtype=float)).rolling(30, min_periods=1).mean()
    score += (body_ratio_mean < config.BODY_RATIO_ACCUM_MAX).fillna(False)
    score += (df.get("rel_strength_180", pd.Series(dtype=float)) > config.REL_STRENGTH_MIN).fillna(
        False
    )

    if "buy_ratio" in df.columns:
        buy_ratio_mean = df["buy_ratio"].rolling(30, min_periods=1).mean()
        score += (buy_ratio_mean > config.BUY_RATIO_MIN).fillna(False)

    if {"funding_rate", "oi_change_60"}.issubset(df.columns):
        funding_ok = df["funding_rate"].abs() <= config.FUNDING_NEAR_ZERO
        oi_ok = df["oi_change_60"] > config.OI_CHANGE_60_MIN
        score += (funding_ok & oi_ok).fillna(False)

    df["pre_pump_score"] = score
    df["is_pre_pump"] = df["pre_pump_score"] >= config.PRE_PUMP_SCORE_THRESHOLD
    return df


def compute_long_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute long entry signals based on pre-pump flag and breakout logic."""
    pre_pump_recent = df["is_pre_pump"].rolling(config.BASE_WINDOW_MINUTES, min_periods=1).max() > 0
    prev_high = df["high"].rolling(config.BASE_WINDOW_MINUTES, min_periods=1).max().shift(1)

    conditions = [
        pre_pump_recent,
        df.get("vol_ratio_30", pd.Series(dtype=float)) >= config.VOL_RATIO_30_ENTRY,
        df.get("body_ratio", pd.Series(dtype=float)) >= config.BODY_RATIO_ENTRY,
        (df["high"] > prev_high) | (df["close"] > prev_high),
    ]
    df["long_entry"] = pd.concat(conditions, axis=1).all(axis=1)
    return df
