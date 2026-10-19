"""Position-wide cash exits for strategies that provide bracket parameters.

Signals supply constant cash amounts and instrument units. The executor owns
the entry price, quantity and order activation time; this module only converts
the amounts and resolves an already active bracket against a candle.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
import pandas as pd


PARAMETER_COLUMNS = ("take_profit_cash", "stop_loss_cash", "point_value", "tick_size")


@dataclass(frozen=True)
class CashExitPolicy:
    take_profit_cash: float
    stop_loss_cash: float
    point_value: float
    tick_size: float

    @classmethod
    def from_signals(cls, signals: pd.DataFrame) -> CashExitPolicy | None:
        """Read an optional complete, constant bracket configuration."""
        present = set(PARAMETER_COLUMNS).intersection(signals.columns)
        if not present:
            return None
        missing = set(PARAMETER_COLUMNS).difference(present)
        if missing:
            raise ValueError(f"Missing cash-exit columns: {', '.join(sorted(missing))}")
        if signals.empty:
            return None
        settings = {}
        for name in PARAMETER_COLUMNS:
            column = signals[name]
            if column.map(lambda value: isinstance(value, (bool, np.bool_))).any():
                raise ValueError(f"{name} must contain finite positive numbers, not booleans")
            try:
                values = column.to_numpy(dtype=float)
            except (ValueError, TypeError) as error:
                raise ValueError(f"{name} must contain finite positive numbers") from error
            if not np.isfinite(values).all() or (values <= 0).any():
                raise ValueError(f"{name} must contain finite positive numbers")
            if not (values == values[0]).all():
                raise ValueError(f"{name} must remain constant throughout a backtest")
            settings[name] = float(values[0])
        return cls(**settings)

    def _price_distances(self, quantity: int) -> tuple[float, float]:
        if quantity <= 0:
            raise ValueError("Cash exits require a positive filled quantity")
        units = self.point_value * quantity
        distances = (self.stop_loss_cash / units, self.take_profit_cash / units)
        if not all(isfinite(value) and value > 0 for value in distances):
            raise ValueError("Cash exit price distances must be finite and positive")
        return distances

    def levels(self, entry_price: float, quantity: int) -> tuple[float, float]:
        """Return stop and target around the actual average entry fill.

        The Pine formula divides by tick size to obtain float tick distances;
        multiplying those distances back by tick size cancels that factor.
        Preserve fractional distances rather than introducing integer ticks.
        """
        stop_distance, target_distance = self._price_distances(quantity)
        levels = (entry_price - stop_distance, entry_price + target_distance)
        if not all(isfinite(value) for value in levels):
            raise ValueError("Cash exit price levels must be finite")
        return levels

    def tick_distances(self, quantity: int) -> tuple[float, float]:
        """Return the source script's loss/profit distances in float ticks."""
        stop_distance, target_distance = self._price_distances(quantity)
        return stop_distance / self.tick_size, target_distance / self.tick_size

    def fill(
        self, row: pd.Series, stop_loss_price: float, take_profit_price: float,
    ) -> tuple[float, str] | None:
        """Resolve active orders using gap prices and the broker's OHLC path.

        A gap through a level fills at the open. Otherwise the first boundary
        reached on the assumed path wins. Equal open-to-high/low distances use
        the low first, an explicit convention for the undocumented tie case.
        Slippage, fees, activation and cancellation belong to the executor.
        """
        opening, high, low = (float(row[name]) for name in ("open", "high", "low"))
        if opening <= stop_loss_price:
            return opening, "STOP_LOSS"
        if opening >= take_profit_price:
            return opening, "TAKE_PROFIT"
        touches_stop = low <= stop_loss_price
        touches_target = high >= take_profit_price
        if touches_stop and touches_target:
            if high - opening < opening - low:
                return take_profit_price, "TAKE_PROFIT"
            return stop_loss_price, "STOP_LOSS"
        if touches_stop:
            return stop_loss_price, "STOP_LOSS"
        if touches_target:
            return take_profit_price, "TAKE_PROFIT"
        return None
