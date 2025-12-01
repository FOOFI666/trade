"""Signal computation for pre-pump detection and long entries."""

import pandas as pd

import config


def compute_pre_pump_score(df: pd.DataFrame) -> pd.DataFrame:
    """Вычисление суммарного pre-pump score и флага is_pre_pump."""
    df["cond_bbw_low"] = (
        df.get("bbw_percentile", pd.Series(index=df.index, dtype=float))
        <= config.PRE_BBW_MAX_PERCENTILE
    ).fillna(False)

    vol_60 = df.get("vol_ratio_60", pd.Series(index=df.index, dtype=float))
    df["cond_vol_60_in_range"] = (
        (vol_60 >= config.PRE_VOL60_MIN) & (vol_60 <= config.PRE_VOL60_MAX)
    ).fillna(False)

    body_ratio = df.get("body_ratio", pd.Series(index=df.index, dtype=float))
    body_ratio_mean = body_ratio.rolling(config.PRE_BODYR_WINDOW, min_periods=1).mean()
    df["cond_body_ratio_low"] = (body_ratio_mean <= config.PRE_BODYR_MAX).fillna(False)

    rel_strength = df.get("rel_strength_180", pd.Series(index=df.index, dtype=float))
    df["cond_rel_strength_pos"] = (rel_strength >= config.PRE_REL_STRENGTH_MIN).fillna(False)

    buy_ratio_mean = df.get("buy_ratio", pd.Series(index=df.index, dtype=float)).rolling(
        config.PRE_BUYR_WINDOW, min_periods=1
    ).mean()
    df["cond_buy_ratio_high"] = (buy_ratio_mean >= config.PRE_BUYR_MIN).fillna(False)

    funding_ok = df.get("funding_rate", pd.Series(index=df.index, dtype=float)).abs() <= config.PRE_FUNDING_ABS_MAX
    oi_ok = df.get("oi_change_60", pd.Series(index=df.index, dtype=float)) >= config.PRE_OI_CHANGE_MIN
    df["cond_funding_oi_ok"] = (funding_ok & oi_ok).fillna(False)

    cond_cols = [
        "cond_bbw_low",
        "cond_vol_60_in_range",
        "cond_body_ratio_low",
        "cond_rel_strength_pos",
        "cond_buy_ratio_high",
        "cond_funding_oi_ok",
    ]
    df["pre_pump_score"] = df[cond_cols].astype(int).sum(axis=1)
    df["is_pre_pump"] = df["pre_pump_score"] >= config.PRE_PUMP_SCORE_MIN
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
