"""Execute long-only strategy signals at the following candle's open.

ChartArt's lower-band buy stop is already activated when its signal closes
above that band. It therefore becomes a next-tick market order, rather than
a protective stop or a stop that must be touched again on the next candle.
SuperTrend's market entries and closing exits use the same next-open fills.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    quantity: int = 1
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    close_at_end: bool = True

    def __post_init__(self):
        if (
            isinstance(self.initial_capital, bool)
            or not isinstance(self.initial_capital, Real)
            or not isfinite(self.initial_capital)
            or self.initial_capital <= 0
        ):
            raise ValueError("initial_capital must be finite and greater than zero")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, Integral) or self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")
        for name in ("commission_bps", "slippage_bps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.slippage_bps >= 10_000:
            raise ValueError("slippage_bps must be less than 10000")
        if not isinstance(self.close_at_end, bool):
            raise ValueError("close_at_end must be a boolean")


@dataclass
class BacktestResult:
    metrics: dict
    trades: pd.DataFrame
    equity: pd.DataFrame
    orders: pd.DataFrame


TRADE_COLUMNS = [
    "entry_signal_time", "entry_time", "exit_signal_time", "exit_time",
    "entry_price", "exit_price", "quantity", "gross_profit", "fees",
    "net_profit", "return_pct", "exit_reason",
]
ORDER_COLUMNS = [
    "signal_time", "execution_time", "side", "status", "price", "quantity",
    "reason", "entry_stop",
]
EQUITY_COLUMNS = ["cash", "quantity", "close", "equity", "drawdown_pct"]


class LongOnlyBacktester:
    """Hold at most one fixed-size long; EXIT signals only close that long.

    Each ticker receives its own capital account. There is no leverage,
    pyramiding, shorting, dividend credit, or intrabar protective stop.
    """

    def __init__(self, config: BacktestConfig | None = None):
        self.config = config if config is not None else BacktestConfig()

    @staticmethod
    def _validate(signals: pd.DataFrame) -> None:
        required = {"open", "high", "low", "close", "buy_signal", "exit_signal"}
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"Missing backtest columns: {', '.join(sorted(missing))}")
        if signals.index.has_duplicates or not signals.index.is_monotonic_increasing or signals.index.hasnans:
            raise ValueError("Backtest index must be unique, chronological, and nonmissing")
        try:
            prices = signals[["open", "high", "low", "close"]].to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError("OHLC prices must be numeric") from exc
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise ValueError("OHLC prices must be finite and positive")
        if len(prices) and (
            (prices[:, 1] < np.max(prices[:, [0, 2, 3]], axis=1)).any()
            or (prices[:, 2] > np.min(prices[:, [0, 1, 3]], axis=1)).any()
        ):
            raise ValueError("Each candle's high and low must contain its open and close")
        for name in ("buy_signal", "exit_signal"):
            if not signals[name].map(lambda value: isinstance(value, (bool, np.bool_))).all():
                raise ValueError(f"{name} must contain boolean values without missing entries")
        if (signals["buy_signal"] & signals["exit_signal"]).any():
            raise ValueError("A candle cannot contain both buy and exit signals")

    def run(self, signals: pd.DataFrame) -> BacktestResult:
        self._validate(signals)
        config = self.config
        fee_rate = config.commission_bps / 10_000.0
        slip_rate = config.slippage_bps / 10_000.0
        cash = float(config.initial_capital)
        quantity = 0
        entry = None
        pending = None
        trades = []
        orders = []
        equity_rows = []
        total_fees = 0.0

        def close_position(timestamp, reference_price, reason, signal_time, order):
            nonlocal cash, quantity, entry, total_fees
            price = reference_price * (1.0 - slip_rate)
            notional = quantity * price
            exit_fee = notional * fee_rate
            cash += notional - exit_fee
            total_fees += exit_fee
            gross = quantity * (price - entry["price"])
            fees = entry["fee"] + exit_fee
            net = gross - fees
            trades.append({
                "entry_signal_time": entry["signal_time"],
                "entry_time": entry["time"],
                "exit_signal_time": signal_time,
                "exit_time": timestamp,
                "entry_price": entry["price"],
                "exit_price": price,
                "quantity": quantity,
                "gross_profit": gross,
                "fees": fees,
                "net_profit": net,
                "return_pct": net / (quantity * entry["price"] + entry["fee"]) * 100.0,
                "exit_reason": reason,
            })
            order.update(execution_time=timestamp, status="FILLED", price=price, quantity=quantity, reason=reason)
            quantity = 0
            entry = None

        for index, (timestamp, row) in enumerate(signals.iterrows()):
            # Only orders created on a previous close can fill at this open.
            if pending is not None:
                order = orders[pending]
                if order["side"] == "BUY":
                    price = float(row["open"]) * (1.0 + slip_rate)
                    notional = int(config.quantity) * price
                    fee = notional * fee_rate
                    if notional + fee > cash:
                        order.update(execution_time=timestamp, status="REJECTED", reason="INSUFFICIENT_CASH")
                    else:
                        cash -= notional + fee
                        quantity = int(config.quantity)
                        total_fees += fee
                        entry = {"signal_time": order["signal_time"], "time": timestamp, "price": price, "fee": fee}
                        order.update(execution_time=timestamp, status="FILLED", price=price, reason="BUY_SIGNAL")
                else:
                    close_position(timestamp, float(row["open"]), "EXIT_SIGNAL", order["signal_time"], order)
                pending = None

            side = "BUY" if row["buy_signal"] else "EXIT" if row["exit_signal"] else None
            if side is not None:
                stop = row.get("entry_stop", None) if side == "BUY" else None
                order = {
                    "signal_time": timestamp, "execution_time": None, "side": side,
                    "status": "PENDING", "price": None, "quantity": int(config.quantity),
                    "reason": "", "entry_stop": float(stop) if stop is not None and pd.notna(stop) else None,
                }
                if side == "BUY" and quantity:
                    order.update(status="SKIPPED", reason="POSITION_OPEN")
                elif side == "EXIT" and not quantity:
                    order.update(status="SKIPPED", reason="NO_POSITION")
                elif index == len(signals) - 1:
                    order.update(status="NO_NEXT_BAR", reason="NO_NEXT_BAR")
                else:
                    pending = len(orders)
                orders.append(order)

            # Liquidation is an explicit reporting convention, not a Pine signal.
            if index == len(signals) - 1 and quantity and config.close_at_end:
                order = {
                    "signal_time": None, "execution_time": None, "side": "EXIT",
                    "status": "PENDING", "price": None, "quantity": quantity,
                    "reason": "END_OF_DATA", "entry_stop": None,
                }
                close_position(timestamp, float(row["close"]), "END_OF_DATA", None, order)
                orders.append(order)
            close = float(row["close"])
            equity_rows.append({"cash": cash, "quantity": quantity, "close": close, "equity": cash + quantity * close})

        trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        orders_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
        equity_frame = pd.DataFrame(equity_rows, index=signals.index, columns=EQUITY_COLUMNS)
        if equity_frame.empty:
            final_equity = float(config.initial_capital)
            max_drawdown = 0.0
        else:
            peak = equity_frame["equity"].cummax().clip(lower=float(config.initial_capital))
            equity_frame["drawdown_pct"] = (peak - equity_frame["equity"]) / peak * 100.0
            max_drawdown = float(equity_frame["drawdown_pct"].max())
            final_equity = float(equity_frame["equity"].iloc[-1])
        total_trades = len(trades)
        winning_trades = sum(trade["net_profit"] > 0 for trade in trades)
        losing_trades = sum(trade["net_profit"] < 0 for trade in trades)
        breakeven_trades = sum(trade["net_profit"] == 0 for trade in trades)
        total_buys = sum(order["side"] == "BUY" and order["status"] == "FILLED" for order in orders)
        total_sells = sum(order["side"] == "EXIT" and order["status"] == "FILLED" for order in orders)
        closed_profit = sum(trade["net_profit"] for trade in trades)
        unrealized = quantity * (float(signals["close"].iloc[-1]) - entry["price"]) - entry["fee"] if entry else 0.0
        metrics = {
            "initial_capital": float(config.initial_capital),
            "final_equity": final_equity,
            "net_profit": final_equity - config.initial_capital,
            "total_return_pct": (final_equity / config.initial_capital - 1.0) * 100.0,
            "total_trades": total_trades,
            "total_buys": total_buys,
            "total_sells": total_sells,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "breakeven_trades": breakeven_trades,
            "win_rate_pct": winning_trades / total_trades * 100.0 if total_trades else None,
            "max_drawdown_pct": max_drawdown,
            "buy_signals": int(signals["buy_signal"].sum()),
            "exit_signals": int(signals["exit_signal"].sum()),
            "open_positions": int(bool(quantity)),
            "total_fees": total_fees,
            "closed_trade_profit": float(closed_profit),
            "unrealized_profit": float(unrealized),
        }
        return BacktestResult(metrics, trades_frame, equity_frame, orders_frame)
