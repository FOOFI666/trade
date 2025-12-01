"""Main runtime loop for Binance futures pre-pump scanner."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict

import pandas as pd

import config
import data_stream
from binance_client import get_futures_symbols, start_kline_stream
from funding_features import add_funding_zscore, add_open_interest_change
from signals import compute_long_entry_signals, compute_pre_pump_score
from strength_features import add_relative_strength
from volatility_features import add_atr, add_bbw_percentile, add_bollinger_bandwidth, add_true_range
from volume_features import add_basic_candle_metrics, add_taker_metrics, add_volume_relative


logger = logging.getLogger("scanner")
logger.setLevel(logging.INFO)

if config.LOG_SIGNALS_TO_CONSOLE:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

if config.LOG_SIGNALS_TO_FILE:
    file_handler = logging.FileHandler(config.SIGNALS_LOG_PATH)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)


# Placeholder for Telegram integration
def _maybe_send_telegram(message: str):
    if config.ENABLE_TELEGRAM_NOTIFICATIONS:
        logger.info("Telegram notifications enabled, but sender is not implemented.")


def _update_features_for_symbol(symbol: str) -> pd.DataFrame:
    df = data_stream.symbol_data[symbol]
    if df.empty:
        return df

    add_true_range(df)
    add_atr(df, config.ATR_WINDOW, "atr_60")
    add_atr(df, config.VOL_RATIO_30_WINDOW, "atr_30")

    add_bollinger_bandwidth(df, config.BBW_WINDOW, 2.0, "bbw")
    add_bbw_percentile(df, "bbw", config.BBW_PERCENTILE_WINDOW, "bbw_percentile")

    add_basic_candle_metrics(df)
    add_volume_relative(df, config.VOL_RATIO_60_WINDOW, "vol_ma_60", "vol_ratio_60")
    add_volume_relative(df, config.VOL_RATIO_30_WINDOW, "vol_ma_30", "vol_ratio_30")
    add_taker_metrics(df)

    if data_stream.btc_data is not None and not data_stream.btc_data.empty:
        df = add_relative_strength(
            df, data_stream.btc_data, config.REL_STRENGTH_WINDOW, "rel_strength_180"
        )

    if {"funding_rate"}.issubset(df.columns):
        add_funding_zscore(df, config.BODY_RATIO_MEAN_WINDOW)
    if "open_interest" in df.columns:
        add_open_interest_change(df, config.VOL_RATIO_60_WINDOW, "oi_change_60")

    data_stream.symbol_data[symbol] = df
    return df


def _log_signal(symbol: str, row: pd.Series):
    timestamp = datetime.fromtimestamp(row["timestamp"] / 1000.0)
    message = (
        f"LONG ENTRY {symbol} at {timestamp}: price={row['close']:.4f}, "
        f"score={row.get('pre_pump_score', 0):.0f}, vol_ratio_30={row.get('vol_ratio_30', float('nan')):.2f}"
    )
    if config.LOG_SIGNALS_TO_CONSOLE:
        logger.info(message)
    if config.LOG_SIGNALS_TO_FILE:
        logger.info(message)
    _maybe_send_telegram(message)


def main():
    symbols = [s for s in get_futures_symbols() if s.endswith("USDT")]
    symbols = [s for s in symbols if s not in config.SYMBOLS_BLACKLIST]

    data_stream.init_symbol_data(symbols)

    def on_kline(symbol: str, kline: Dict):
        df = data_stream.update_symbol_bar(symbol, kline)
        df = _update_features_for_symbol(symbol)
        df = compute_pre_pump_score(df)
        df = compute_long_entry_signals(df)

        if not df.empty and bool(df.iloc[-1]["long_entry"]):
            _log_signal(symbol, df.iloc[-1])

    start_kline_stream(symbols, config.KLINE_INTERVAL, on_kline)


if __name__ == "__main__":
    main()
