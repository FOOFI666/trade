"""Helpers for loading kline history for specific date ranges with immutable cache files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import config
from binance_client import get_historical_klines


def get_range_klines_path(symbol: str, interval: str, start_time_ms: int, end_time_ms: int) -> Path:
    """
    Строит путь к файлу диапазона для конкретного символа, таймфрейма и временного окна.

    Формат имени: SYMBOL_INTERVAL_YYYYMMDD_HHMMSS__YYYYMMDD_HHMMSS.parquet
    """

    start_str = pd.to_datetime(start_time_ms, unit="ms").strftime("%Y%m%d_%H%M%S")
    end_str = pd.to_datetime(end_time_ms, unit="ms").strftime("%Y%m%d_%H%M%S")

    ext = "parquet" if config.RANGE_KLINES_FORMAT == "parquet" else "csv"
    dir_path = Path(config.RANGE_KLINES_DIR)
    dir_path.mkdir(parents=True, exist_ok=True)

    return dir_path / f"{symbol}_{interval}_{start_str}__{end_str}.{ext}"


def load_range_klines(
    symbol: str,
    interval: str,
    start_time_ms: int,
    end_time_ms: int,
) -> pd.DataFrame:
    """
    1. Строит путь файла по диапазону.
    2. Если файл существует -> читает его и возвращает DataFrame.
    3. Если файла нет -> скачивает данные из Binance в этом диапазоне,
       сохраняет ИМЕННО этот диапазон в новый файл, возвращает DataFrame.
    4. Ничего не удаляет и не перезаписывает, только создаёт новые файлы.
    """

    path = get_range_klines_path(symbol, interval, start_time_ms, end_time_ms)

    if path.exists():
        if config.RANGE_KLINES_FORMAT == "parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)
        return df

    df = get_historical_klines(
        symbol,
        interval,
        start_time=start_time_ms,
        end_time=end_time_ms,
    )

    if df.empty:
        if config.RANGE_KLINES_FORMAT == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return df

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    if config.RANGE_KLINES_FORMAT == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)

    return df
