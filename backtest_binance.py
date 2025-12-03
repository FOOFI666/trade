"""Offline backtest of long-entry signals using Binance historical data."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import pandas as pd

import config
from binance_client import get_futures_symbols, load_klines_from_cache
from data_range_cache import load_range_klines
from features_pipeline import update_features
from signals import compute_long_entry_signals, compute_pre_pump_score


def _parse_date_ms(value: str) -> int:
    return int(pd.to_datetime(value).tz_localize(None).timestamp() * 1000)


def _resolve_symbols(cli_symbols: Iterable[str]) -> List[str]:
    symbols = [s for s in get_futures_symbols() if s.endswith("USDT")]

    if config.BACKTEST_SYMBOLS:
        allowed = set(config.BACKTEST_SYMBOLS)
        symbols = [s for s in symbols if s in allowed]

    if cli_symbols:
        allowed = set(cli_symbols)
        symbols = [s for s in symbols if s in allowed]

    return symbols


def _find_available_range(symbols: Iterable[str], interval: str) -> tuple[int, int] | None:
    starts: list[int] = []
    ends: list[int] = []

    for sym in symbols:
        cached = load_klines_from_cache(sym, interval)
        if cached is None or cached.empty:
            continue

        cached = cached.sort_values("timestamp").drop_duplicates(subset=["timestamp"])
        starts.append(int(cached["timestamp"].iloc[0]))
        ends.append(int(cached["timestamp"].iloc[-1]))

    if not starts or not ends:
        return None

    return min(starts), max(ends)


def _gather_signal_rows(symbol: str, df: pd.DataFrame) -> list[dict]:
    signal_rows = []
    mask = df.get("long_entry", pd.Series(False, index=df.index)).astype(bool)
    for _, row in df[mask].iterrows():
        signal_rows.append(
            {
                "symbol": symbol,
                "timestamp": int(row["timestamp"]),
                "datetime": pd.to_datetime(row["timestamp"], unit="ms"),
                "close": row.get("close"),
                "pre_pump_score": row.get("pre_pump_score", float("nan")),
                "vol_ratio_30": row.get("vol_ratio_30", float("nan")),
                "bbw_percentile": row.get("bbw_percentile", float("nan")),
                "rel_strength_180": row.get("rel_strength_180", float("nan")),
                "funding_rate": row.get("funding_rate", float("nan")),
                "oi_change_60": row.get("oi_change_60", float("nan")),
            }
        )
    return signal_rows


def main():
    parser = argparse.ArgumentParser(description="Offline backtest for Binance long-entry signals")
    parser.add_argument(
        "--start",
        type=str,
        help="Start datetime (ISO format, e.g. 2025-11-19 or 2025-11-19T00:00:00)",
    )
    parser.add_argument("--end", type=str, help="End datetime (ISO format)")
    parser.add_argument(
        "--days",
        type=int,
        default=config.BACKTEST_DAYS,
        help="If --start is not set: number of days before --end",
    )
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated list of symbols to backtest")
    parser.add_argument("--output", type=str, help="CSV output path")
    parser.add_argument(
        "--use-cache-range",
        action="store_true",
        help="Use full range from cached klines instead of specifying a start/end",
    )
    args = parser.parse_args()

    cli_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    symbols = _resolve_symbols(cli_symbols)

    cache_range = None
    if args.use_cache_range:
        cache_range = _find_available_range([config.BTC_SYMBOL, *symbols], config.BACKTEST_INTERVAL)
        if cache_range is None:
            raise ValueError("No cached klines found for the requested interval.")

    if cache_range:
        start_time_ms, end_time_ms = cache_range
    else:
        if args.start and args.end:
            start_time_ms = _parse_date_ms(args.start)
            end_time_ms = _parse_date_ms(args.end)
        else:
            end_time_ms = _parse_date_ms(args.end) if args.end else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            if args.start:
                start_time_ms = _parse_date_ms(args.start)
            else:
                start_dt = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc) - timedelta(days=args.days)
                start_time_ms = int(start_dt.timestamp() * 1000)

    start_dt = pd.to_datetime(start_time_ms, unit="ms")
    end_dt = pd.to_datetime(end_time_ms, unit="ms")
    print(f"Backtesting interval: {config.BACKTEST_INTERVAL} from {start_dt} to {end_dt}")
    print(f"Symbols to process: {len(symbols)}")

    if args.output:
        output_path = args.output
    else:
        start_str = start_dt.strftime("%Y%m%d_%H%M%S")
        end_str = end_dt.strftime("%Y%m%d_%H%M%S")
        output_path = f"backtest_signals_{start_str}__{end_str}.csv"

    btc_df = load_range_klines(config.BTC_SYMBOL, config.BACKTEST_INTERVAL, start_time_ms, end_time_ms)

    all_signals: list[dict] = []
    for symbol in symbols:
        print(f"Loading {symbol} history...")
        df = load_range_klines(symbol, config.BACKTEST_INTERVAL, start_time_ms, end_time_ms)
        if df.empty:
            continue

        df = update_features(df, btc_df)
        df = compute_pre_pump_score(df)
        df = compute_long_entry_signals(df)

        all_signals.extend(_gather_signal_rows(symbol, df))

    if not all_signals:
        print("No signals found for the selected period.")
        return

    signals_df = pd.DataFrame(all_signals)
    signals_df.sort_values(["timestamp", "symbol"], inplace=True)
    signals_df.to_csv(output_path, index=False)

    print(f"Saved {len(signals_df)} signals to {output_path}")
    counts = Counter(signals_df["symbol"])
    top_symbols = counts.most_common(5)
    print("Top symbols by signals:")
    for sym, cnt in top_symbols:
        print(f"  {sym}: {cnt}")


if __name__ == "__main__":
    main()
