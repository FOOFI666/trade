"""Offline backtest of long-entry signals using Binance historical data."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Iterable, List

import pandas as pd

import config
from binance_client import get_futures_symbols, get_historical_klines
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


def _fetch_history(symbol: str, interval: str, start_time: int, end_time: int) -> pd.DataFrame:
    df = get_historical_klines(
        symbol,
        interval,
        start_time=start_time,
        end_time=end_time,
    )
    if df.empty:
        return df

    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df.reset_index(drop=True, inplace=True)
    return df


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
    parser.add_argument("--days", type=int, default=config.BACKTEST_DAYS, help="Number of days to backtest")
    parser.add_argument("--symbols", type=str, default="", help="Comma-separated list of symbols to backtest")
    parser.add_argument("--start", type=str, help="Start datetime (ISO format)")
    parser.add_argument("--end", type=str, help="End datetime (ISO format)")
    parser.add_argument("--output", type=str, default=config.BACKTEST_OUTPUT_PATH, help="CSV output path")
    args = parser.parse_args()

    end_time_ms = _parse_date_ms(args.end) if args.end else int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    if args.start:
        start_time_ms = _parse_date_ms(args.start)
    else:
        start_dt = datetime.fromtimestamp(end_time_ms / 1000, tz=timezone.utc) - timedelta(days=args.days)
        start_time_ms = int(start_dt.timestamp() * 1000)

    cli_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    symbols = _resolve_symbols(cli_symbols)

    print(f"Backtesting interval: {config.BACKTEST_INTERVAL} from {pd.to_datetime(start_time_ms, unit='ms')} to {pd.to_datetime(end_time_ms, unit='ms')}")
    print(f"Symbols to process: {len(symbols)}")

    btc_df = _fetch_history(config.BTC_SYMBOL, config.BACKTEST_INTERVAL, start_time_ms, end_time_ms)

    all_signals: list[dict] = []
    for symbol in symbols:
        print(f"Downloading {symbol} history...")
        df = _fetch_history(symbol, config.BACKTEST_INTERVAL, start_time_ms, end_time_ms)
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
    signals_df.to_csv(args.output, index=False)

    print(f"Saved {len(signals_df)} signals to {args.output}")
    counts = Counter(signals_df["symbol"])
    top_symbols = counts.most_common(5)
    print("Top symbols by signals:")
    for sym, cnt in top_symbols:
        print(f"  {sym}: {cnt}")


if __name__ == "__main__":
    main()
