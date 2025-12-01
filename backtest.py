"""Backtesting logic for the long-only pre-pump strategy."""
from dataclasses import dataclass, field
from typing import Dict, List
import pandas as pd
import numpy as np
import config


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    stop_loss: float
    take_profit: float
    size: float
    fee_paid: float
    stop_distance: float
    max_favorable_price: float = field(default=0.0)


def _calculate_stop_levels(row: pd.Series) -> Dict[str, float]:
    atr_value = row.get("atr_30")
    if pd.notna(atr_value):
        stop_distance = atr_value * config.ATR_MULTIPLIER_ENTRY * config.STOP_LOSS_MULTIPLIER
    else:
        stop_distance = row["close"] * 0.005
    stop_loss = row["close"] - stop_distance
    take_profit = row["close"] + config.TAKE_PROFIT_MULTIPLIER * stop_distance
    return {"stop_loss": stop_loss, "take_profit": take_profit, "stop_distance": stop_distance}


def run_backtest(df: pd.DataFrame) -> Dict[str, object]:
    """Simulate trades over time using generated long_entry signals."""
    df = df.sort_values("timestamp").reset_index(drop=True)
    balance = config.INITIAL_BALANCE
    open_trades: List[Trade] = []
    closed_trades: List[Dict[str, object]] = []
    equity_curve: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        current_time = row["timestamp"]

        # Update open trades for exits
        updated_open_trades: List[Trade] = []
        for trade in open_trades:
            trade.max_favorable_price = max(trade.max_favorable_price, row["high"])
            exit_reason = None
            exit_price = None

            if row["low"] <= trade.stop_loss:
                exit_price = trade.stop_loss
                exit_reason = "stop_loss"
            elif row["high"] >= trade.take_profit:
                exit_price = trade.take_profit
                exit_reason = "take_profit"
            elif (current_time - trade.entry_time).total_seconds() / 60 >= config.MAX_HOLDING_MINUTES:
                exit_price = row["close"]
                exit_reason = "timeout"

            if exit_reason:
                gross_pnl = (exit_price - trade.entry_price) * trade.size
                exit_fee = exit_price * trade.size * config.FEE_RATE
                net_pnl = gross_pnl - trade.fee_paid - exit_fee
                balance += net_pnl
                closed_trades.append(
                    {
                        "entry_time": trade.entry_time,
                        "exit_time": current_time,
                        "entry_price": trade.entry_price,
                        "exit_price": exit_price,
                        "size": trade.size,
                        "exit_reason": exit_reason,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "stop_distance": trade.stop_distance,
                    }
                )
            else:
                updated_open_trades.append(trade)
        open_trades = updated_open_trades

        # Process new entries
        if row.get("long_entry") and len(open_trades) < config.MAX_OPEN_TRADES:
            levels = _calculate_stop_levels(row)
            stop_distance = levels["stop_distance"]
            if stop_distance <= 0:
                continue

            risk_amount = balance * config.RISK_PER_TRADE
            position_size = risk_amount / stop_distance

            entry_fee = row["close"] * position_size * config.FEE_RATE
            balance -= entry_fee

            open_trades.append(
                Trade(
                    entry_time=current_time,
                    entry_price=row["close"],
                    stop_loss=levels["stop_loss"],
                    take_profit=levels["take_profit"],
                    size=position_size,
                    fee_paid=entry_fee,
                    stop_distance=stop_distance,
                    max_favorable_price=row["high"],
                )
            )

        # Track equity including unrealized PnL
        unrealized = sum((row["close"] - t.entry_price) * t.size for t in open_trades)
        equity_curve.append({"timestamp": current_time, "equity": balance + unrealized})

    # Close remaining trades at final close
    if not df.empty and open_trades:
        final_price = df.iloc[-1]["close"]
        final_time = df.iloc[-1]["timestamp"]
        for trade in open_trades:
            gross_pnl = (final_price - trade.entry_price) * trade.size
            exit_fee = final_price * trade.size * config.FEE_RATE
            net_pnl = gross_pnl - trade.fee_paid - exit_fee
            balance += net_pnl
            closed_trades.append(
                {
                    "entry_time": trade.entry_time,
                    "exit_time": final_time,
                    "entry_price": trade.entry_price,
                    "exit_price": final_price,
                    "size": trade.size,
                    "exit_reason": "final_close",
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "stop_distance": trade.stop_distance,
                }
            )
        equity_curve.append({"timestamp": final_time, "equity": balance})

    equity_df = pd.DataFrame(equity_curve)
    closed_df = pd.DataFrame(closed_trades)

    total_pnl = closed_df["net_pnl"].sum() if not closed_df.empty else 0.0
    winrate = (closed_df["net_pnl"] > 0).mean() if not closed_df.empty else 0.0
    rr_values = []
    for trade in closed_trades:
        risk = trade["stop_distance"] * trade["size"] if trade.get("stop_distance") else np.nan
        if pd.notna(risk) and risk != 0:
            rr_values.append(trade["net_pnl"] / risk)
    avg_r = float(np.nanmean(rr_values)) if rr_values else 0.0

    if not equity_df.empty:
        cummax = equity_df["equity"].cummax()
        drawdowns = (equity_df["equity"] - cummax) / cummax
        max_dd = drawdowns.min()
    else:
        max_dd = 0.0

    return {
        "trades_df": closed_df,
        "equity_curve": equity_df,
        "metrics": {
            "total_pnl": total_pnl,
            "winrate": winrate,
            "avg_r": avg_r,
            "max_drawdown": max_dd,
            "final_balance": balance,
            "trade_count": len(closed_trades),
        },
    }
