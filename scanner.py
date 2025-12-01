"""Main runtime loop for Binance futures pre-pump scanner."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Dict

import pandas as pd
import numpy as np

import config
import data_stream
from binance_client import get_futures_symbols, start_kline_stream
from features_pipeline import update_features
from nn_entry_model import predict_entry_proba
from signals import compute_long_entry_signals, compute_pre_pump_score


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

_last_signal_time: Dict[str, datetime] = {}


# Placeholder for Telegram integration
def _maybe_send_telegram(message: str):
    if config.ENABLE_TELEGRAM_NOTIFICATIONS:
        logger.info("Telegram notifications enabled, but sender is not implemented.")


def _can_emit_signal(symbol: str, now: datetime) -> bool:
    cooldown = timedelta(minutes=config.SIGNAL_COOLDOWN_MINUTES)
    last = _last_signal_time.get(symbol)
    if last is None or now - last >= cooldown:
        _last_signal_time[symbol] = now
        return True
    return False


def _update_features_for_symbol(symbol: str) -> pd.DataFrame:
    df = data_stream.symbol_data[symbol]
    if df.empty:
        return df

    df = update_features(df, data_stream.btc_data)

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


def _log_debug_conditions(symbol: str, row: pd.Series):
    if not getattr(config, "DEBUG_LOG_CONDITIONS", False):
        return

    cond_names = [
        "cond_bbw_low",
        "cond_vol_60_in_range",
        "cond_body_ratio_low",
        "cond_rel_strength_pos",
        "cond_buy_ratio_high",
        "cond_funding_oi_ok",
    ]

    satisfied = [name for name in cond_names if bool(row.get(name, False))]
    if not satisfied:
        return

    timestamp = datetime.fromtimestamp(row["timestamp"] / 1000.0)
    msg = (
        f"DEBUG CONDITIONS {symbol} at {timestamp}: "
        f"score={row.get('pre_pump_score', 0):.0f}, "
        f"flags={','.join(satisfied)}"
    )
    logger.info(msg)


def _log_nn_signal(symbol: str, row: pd.Series, prob: float):
    timestamp = datetime.fromtimestamp(row["timestamp"] / 1000.0)
    message = (
        f"NN LONG ENTRY {symbol} at {timestamp}: "
        f"price={row['close']:.4f}, "
        f"nn_prob={prob:.3f}, "
        f"score={row.get('pre_pump_score', 0):.0f}, "
        f"vol_ratio_30={row.get('vol_ratio_30', float('nan')):.2f}"
    )
    if config.LOG_NN_SIGNALS_TO_CONSOLE:
        logger.info(message)
    if config.LOG_NN_SIGNALS_TO_FILE:
        logger.info(message)
    _maybe_send_telegram("[NN] " + message)


def main():
    symbols = [s for s in get_futures_symbols() if s.endswith("USDT")]
    symbols = [s for s in symbols if s not in config.SYMBOLS_BLACKLIST]

    data_stream.init_symbol_data(symbols)

    def on_kline(symbol: str, kline: Dict):
        df = data_stream.update_symbol_bar(symbol, kline)
        df = _update_features_for_symbol(symbol)
        df = compute_pre_pump_score(df)
        df = compute_long_entry_signals(df)

        if not df.empty:
            _log_debug_conditions(symbol, df.iloc[-1])

        if config.ENABLE_NN_ENTRY and not df.empty:
            row = df.iloc[-1]
            if bool(row.get("is_pre_pump", False)) and len(df) >= config.NN_WINDOW_SIZE:
                window_df = df.iloc[-config.NN_WINDOW_SIZE :]
                features = []
                for col in config.NN_FEATURE_COLUMNS:
                    if col in window_df.columns:
                        features.append(window_df[col].values)
                    else:
                        features.append(np.zeros(config.NN_WINDOW_SIZE))
                window_features = np.stack(features, axis=1)

                prob = predict_entry_proba(window_features)
                df.loc[df.index[-1], "nn_entry_proba"] = prob

                if prob >= config.NN_ENTRY_THRESHOLD:
                    now = datetime.fromtimestamp(row["timestamp"] / 1000.0)
                    if _can_emit_signal(symbol + "_NN", now):
                        _log_nn_signal(symbol, row, prob)

        if not df.empty and bool(df.iloc[-1]["long_entry"]):
            now = datetime.fromtimestamp(df.iloc[-1]["timestamp"] / 1000.0)
            if _can_emit_signal(symbol, now):
                _log_signal(symbol, df.iloc[-1])

    start_kline_stream(symbols, config.KLINE_INTERVAL, on_kline)

    # Keep the main thread alive while the stream is running
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user")


if __name__ == "__main__":
    main()
