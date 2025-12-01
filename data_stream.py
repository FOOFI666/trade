"""Data stream management for Binance futures klines."""

from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

import config
from binance_client import get_historical_klines

symbol_data: Dict[str, pd.DataFrame] = {}
btc_data: Optional[pd.DataFrame] = None


def init_symbol_data(symbols: list[str]):
    """Load historical data for all symbols and initialize caches."""
    global btc_data

    for symbol in symbols:
        df = get_historical_klines(symbol, config.KLINE_INTERVAL, config.HISTORY_MINUTES)
        symbol_data[symbol] = df

    if config.BTC_SYMBOL in symbol_data:
        btc_data = symbol_data[config.BTC_SYMBOL].copy()
    else:
        btc_data = get_historical_klines(
            config.BTC_SYMBOL, config.KLINE_INTERVAL, config.HISTORY_MINUTES
        )


def _normalize_kline_payload(kline: dict) -> dict:
    payload = kline.get("k", kline)
    timestamp = int(payload.get("T", payload.get("t")))
    return {
        "timestamp": timestamp,
        "open": float(payload.get("o")),
        "high": float(payload.get("h")),
        "low": float(payload.get("l")),
        "close": float(payload.get("c")),
        "volume": float(payload.get("v", 0)),
        "quote_volume": float(payload.get("q", 0)),
        "taker_buy_base": float(payload.get("V", 0)),
        "taker_buy_quote": float(payload.get("Q", 0)),
    }


def update_symbol_bar(symbol: str, kline: dict) -> pd.DataFrame:
    """Append a newly closed kline into the symbol's DataFrame."""
    normalized = _normalize_kline_payload(kline)
    df = symbol_data.get(symbol, pd.DataFrame())
    df = pd.concat([df, pd.DataFrame([normalized])], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").tail(config.HISTORY_MINUTES)
    df.reset_index(drop=True, inplace=True)

    symbol_data[symbol] = df
    if symbol == config.BTC_SYMBOL:
        global btc_data
        btc_data = df.copy()

    return df
