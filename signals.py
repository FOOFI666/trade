"""Signal computation for pre-pump detection and long entries."""

import pandas as pd

import config


def compute_pre_pump_score(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисление суммарного pre-pump score и флага is_pre_pump."""
    score = pd.Series(0, index=df.index, dtype=float)

    bbw = df.get("bbw_percentile", pd.Series(index=df.index, dtype=float))
    score += (bbw < config.BBW_PERCENTILE_THRESHOLD).fillna(False)

    vol_60 = df.get("vol_ratio_60", pd.Series(index=df.index, dtype=float))
    vol_in_range = (vol_60 >= config.VOL_RATIO_60_MIN) & (vol_60 <= config.VOL_RATIO_60_MAX)
    score += vol_in_range.fillna(False)

    body_ratio_mean = df.get("body_ratio", pd.Series(index=df.index, dtype=float)).rolling(
        config.BODY_RATIO_MEAN_WINDOW, min_periods=1
    ).mean()
    score += (body_ratio_mean < config.BODY_RATIO_ACCUM_MAX).fillna(False)

    rel_strength = df.get("rel_strength_180", pd.Series(index=df.index, dtype=float))
    score += (rel_strength > config.REL_STRENGTH_MIN).fillna(False)

    if "buy_ratio" in df.columns:
        buy_ratio_mean = df["buy_ratio"].rolling(config.BODY_RATIO_MEAN_WINDOW, min_periods=1).mean()
        score += (buy_ratio_mean > config.BUY_RATIO_MIN).fillna(False)

    if {"funding_rate", "oi_change_60"}.issubset(df.columns):
        funding_ok = df["funding_rate"].abs() <= config.FUNDING_NEAR_ZERO
        oi_ok = df["oi_change_60"] >= config.OI_CHANGE_60_MIN
        score += (funding_ok & oi_ok).fillna(False)

    df["pre_pump_score"] = score
    df["is_pre_pump"] = df["pre_pump_score"] >= config.PRE_PUMP_SCORE_THRESHOLD
    return df


def compute_long_entry_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Расчёт сигналов на вход в лонг после состояния pre-pump."""
    df["base_high"] = (
        df["high"].rolling(config.BASE_RANGE_WINDOW, min_periods=1).max().shift(1)
    )

    df["quote_volume_60m"] = df.get("quote_volume", pd.Series(index=df.index)).rolling(
        60, min_periods=1
    ).sum()

    liquidity_ok = pd.Series(True, index=df.index)
    if "quote_volume" in df.columns:
        liquidity_ok = (
            (df["quote_volume"] >= config.MIN_QUOTE_VOLUME_1M)
            & (df["quote_volume_60m"] >= config.MIN_QUOTE_VOLUME_60M)
        )

    atr_condition = pd.Series(True, index=df.index)
    if {"atr_30", "candle_range"}.issubset(df.columns):
        atr_condition = df["candle_range"] >= config.ATR_MULTIPLIER_ENTRY * df["atr_30"]

    conditions = [
        df.get("is_pre_pump", pd.Series(index=df.index, dtype=bool)).fillna(False),
        df.get("vol_ratio_30", pd.Series(index=df.index, dtype=float)) >= config.VOL_RATIO_30_ENTRY,
        df.get("body_ratio", pd.Series(index=df.index, dtype=float)) >= config.BODY_RATIO_ENTRY,
        (df["close"] > df["base_high"]) | (df["high"] > df["base_high"]),
        atr_condition,
        liquidity_ok,
    ]

    df["rule_long_entry"] = pd.concat(conditions, axis=1).all(axis=1)

    if config.CONFIRMATION_BARS > 0:
        df["rule_long_entry_confirmed"] = (
            df["rule_long_entry"]
            .rolling(config.CONFIRMATION_BARS, min_periods=1)
            .max()
            .astype(bool)
        )
    else:
        df["rule_long_entry_confirmed"] = df["rule_long_entry"]

    df["rule_long_entry"] = df["rule_long_entry_confirmed"]
    df["long_entry"] = df["rule_long_entry"]
    return df
