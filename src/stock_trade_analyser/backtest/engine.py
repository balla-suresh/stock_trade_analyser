"""Execute long-only signals with optional position-sized cash exits.

Market entries and signal exits fill at the next candle's open. Optional
protective orders are created at the entry candle's close and become
eligible on the following candle, matching default Pine execution timing.
"""

from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .cash_exits import CashExitPolicy
from .position_sizing import capital_quantity, cash_rounding_tolerance


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100_000.0
    quantity: int = 1
    commission_bps: float = 0.0
    slippage_bps: float = 0.0
    close_at_end: bool = True
    position_sizing: str = "fixed"
    capital_allocation_pct: float = 100.0

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
        if self.position_sizing not in ("fixed", "capital"):
            raise ValueError("position_sizing must be 'fixed' or 'capital'")
        if (
            isinstance(self.capital_allocation_pct, bool)
            or not isinstance(self.capital_allocation_pct, Real)
            or not isfinite(self.capital_allocation_pct)
            or not 0 < self.capital_allocation_pct <= 100
        ):
            raise ValueError("capital_allocation_pct must be finite, greater than zero, and at most 100")


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
    """Hold at most one long; EXIT signals only close that long.

    Each ticker receives its own capital account. There is no leverage,
    pyramiding, shorting, or dividend credit. Strategies can optionally
    provide a cash-denominated take-profit and stop-loss policy.
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
        cash_exit_policy = CashExitPolicy.from_signals(signals)
        point_value = cash_exit_policy.point_value if cash_exit_policy else 1.0
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
            # A sell limit cannot fill worse than its take-profit price.
            price = reference_price if reason == "TAKE_PROFIT" else reference_price * (1.0 - slip_rate)
            notional = quantity * price * point_value
            exit_fee = notional * fee_rate
            cash += notional - exit_fee
            total_fees += exit_fee
            gross = quantity * (price - entry["price"]) * point_value
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
                "return_pct": net / (quantity * entry["price"] * point_value + entry["fee"]) * 100.0,
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
                    capital_sizing = config.position_sizing == "capital"
                    requested_quantity = (
                        capital_quantity(cash, config.capital_allocation_pct, price, point_value, fee_rate)
                        if capital_sizing else int(config.quantity)
                    )
                    notional = requested_quantity * price * point_value
                    fee = notional * fee_rate
                    cost = notional + fee
                    affordable = cost <= cash or (
                        capital_sizing and isfinite(cost)
                        and cost - cash <= cash_rounding_tolerance(cash, cost)
                    )
                    if not requested_quantity or not affordable:
                        order.update(execution_time=timestamp, status="REJECTED", reason="INSUFFICIENT_CASH")
                        if capital_sizing:
                            order["quantity"] = 0
                    else:
                        cash -= cost
                        if capital_sizing:
                            cash = max(0.0, cash)
                        quantity = requested_quantity
                        total_fees += fee
                        entry = {"signal_time": order["signal_time"], "time": timestamp, "price": price, "fee": fee}
                        if cash_exit_policy:
                            stop_loss_price, take_profit_price = cash_exit_policy.levels(price, quantity)
                            entry.update(
                                stop_loss_price=stop_loss_price,
                                take_profit_price=take_profit_price,
                                bracket_active_from=index + 1,
                                bracket_signal_time=timestamp,
                                stop_activated=False,
                            )
                        order.update(
                            execution_time=timestamp, status="FILLED", price=price,
                            quantity=quantity, reason="BUY_SIGNAL",
                        )
                else:
                    close_position(timestamp, float(row["open"]), "EXIT_SIGNAL", order["signal_time"], order)
                pending = None

            # The strategy first observes a new position at the entry bar's
            # close, so its bracket cannot use that bar's earlier high/low.
            if cash_exit_policy and quantity and index >= entry["bracket_active_from"]:
                if entry["stop_activated"]:
                    fill = (float(row["open"]), "STOP_LOSS")
                else:
                    fill = cash_exit_policy.fill(row, entry["stop_loss_price"], entry["take_profit_price"])
                if fill is not None:
                    reference_price, reason = fill
                    order = {
                        "signal_time": entry["bracket_signal_time"], "execution_time": None,
                        "side": "EXIT", "status": "PENDING", "price": None,
                        "quantity": quantity, "reason": reason, "entry_stop": None,
                    }
                    close_position(timestamp, reference_price, reason, order["signal_time"], order)
                    orders.append(order)

            side = "BUY" if row["buy_signal"] else "EXIT" if row["exit_signal"] else None
            if side is not None:
                stop = row.get("entry_stop", None) if side == "BUY" else None
                order = {
                    "signal_time": timestamp, "execution_time": None, "side": side,
                    "status": "PENDING", "price": None,
                    "quantity": quantity if side == "EXIT" else (
                        None if config.position_sizing == "capital" else int(config.quantity)
                    ),
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

            if cash_exit_policy and quantity and index + 1 == entry["bracket_active_from"]:
                # A stop already crossed when created becomes a market exit
                # at the next open, even if price recovers before that open.
                entry["stop_activated"] = float(row["close"]) <= entry["stop_loss_price"]

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
            equity_row = {
                "cash": cash, "quantity": quantity, "close": close,
                "equity": cash + quantity * close * point_value,
            }
            if cash_exit_policy:
                equity_row.update(
                    stop_loss_price=entry["stop_loss_price"] if entry else np.nan,
                    take_profit_price=entry["take_profit_price"] if entry else np.nan,
                    bracket_active=bool(entry and index >= entry["bracket_active_from"]),
                )
            equity_rows.append(equity_row)

        trades_frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
        orders_frame = pd.DataFrame(orders, columns=ORDER_COLUMNS)
        equity_columns = EQUITY_COLUMNS + ["stop_loss_price", "take_profit_price", "bracket_active"] if cash_exit_policy else EQUITY_COLUMNS
        equity_frame = pd.DataFrame(equity_rows, index=signals.index, columns=equity_columns)
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
        unrealized = quantity * (float(signals["close"].iloc[-1]) - entry["price"]) * point_value - entry["fee"] if entry else 0.0
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
