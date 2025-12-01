"""Minimal REST/WebSocket client wrapper for Binance Futures."""

from __future__ import annotations

import json
import threading
from typing import Callable, Dict, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
import importlib.util

import config

BASE_URL = "https://fapi.binance.com"
STREAM_URL = "wss://fstream.binance.com/stream"


def _create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=config.REQUEST_RETRIES,
        backoff_factor=config.REQUEST_BACKOFF_FACTOR,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("HEAD", "GET", "OPTIONS"),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


SESSION = _create_session()


def _get(url: str, *, params: Optional[dict] = None) -> requests.Response:
    try:
        response = SESSION.get(url, params=params, timeout=config.REQUEST_TIMEOUT)
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as exc:  # pragma: no cover - network safety
        raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc


def get_futures_symbols() -> List[str]:
    """Получить список USDT-маржинальных фьючерсов Binance (perpetual, активные)."""
    url = f"{BASE_URL}/fapi/v1/exchangeInfo"
    response = _get(url)
    data = response.json()
    symbols = []
    for symbol_info in data.get("symbols", []):
        if (
            symbol_info.get("quoteAsset") == "USDT"
            and symbol_info.get("contractType") == "PERPETUAL"
            and symbol_info.get("status") == "TRADING"
        ):
            symbols.append(symbol_info["symbol"])
    return symbols


def get_historical_klines(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Загрузить последние 1m свечи для символа."""
    url = f"{BASE_URL}/fapi/v1/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    response = _get(url, params=params)
    klines = response.json()
    columns = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base",
        "taker_buy_quote",
        "ignore",
    ]
    df = pd.DataFrame(klines, columns=columns)
    df = df.assign(
        timestamp=df["close_time"].astype(int),
        open=df["open"].astype(float),
        high=df["high"].astype(float),
        low=df["low"].astype(float),
        close=df["close"].astype(float),
        volume=df["volume"].astype(float),
        quote_volume=df["quote_asset_volume"].astype(float),
        taker_buy_base=df["taker_buy_base"].astype(float),
        taker_buy_quote=df["taker_buy_quote"].astype(float),
    )[
        [
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
    ]
    return df


def get_funding_and_oi(symbol: str) -> Dict:
    """Получение последних значений funding_rate и open_interest."""
    funding_url = f"{BASE_URL}/fapi/v1/fundingRate"
    oi_url = f"{BASE_URL}/futures/data/openInterestHist"

    funding_resp = _get(funding_url, params={"symbol": symbol, "limit": 1})
    funding_data = funding_resp.json()
    funding_rate = float(funding_data[0]["fundingRate"]) if funding_data else None

    oi_resp = _get(oi_url, params={"symbol": symbol, "period": "5m", "limit": 1})
    oi_data = oi_resp.json()
    open_interest = float(oi_data[0]["sumOpenInterest"]) if oi_data else None

    return {"funding_rate": funding_rate, "open_interest": open_interest}


def start_kline_stream(symbols: List[str], interval: str, on_message: Callable[[str, dict], None]):
    """Запуск WebSocket-потока для списка символов."""
    if importlib.util.find_spec("websocket") is None:
        raise ImportError(
            "The 'websocket-client' package is required for streaming. Install it with "
            "'pip install websocket-client' or 'pip install -r requirements.txt'."
        )

    import websocket  # type: ignore

    streams = "/".join([f"{symbol.lower()}@kline_{interval}" for symbol in symbols])
    url = f"{STREAM_URL}?streams={streams}"

    def _handle_message(ws, message):  # pragma: no cover - network callback
        payload = json.loads(message)
        data = payload.get("data", payload)
        kline = data.get("k", data)
        if kline.get("x"):
            on_message(kline.get("s"), data.get("k", data))

    def _run():  # pragma: no cover - network thread
        ws = websocket.WebSocketApp(url, on_message=_handle_message)
        ws.run_forever()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return thread
