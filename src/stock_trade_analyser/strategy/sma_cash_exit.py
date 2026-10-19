# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Original Pine Script © adolgov. Python adaptation: long positions only.

"""SMA crossings with cash-denominated profit targets and stop losses."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from ..tools.data_utils import normalize_ohlcv
from .base import TradingStrategy


@dataclass(frozen=True)
class SmaCashExitStrategy(TradingStrategy):
    """Generate long-entry and close-long events from completed SMA crossings.

    A fast-SMA crossing above the slow SMA opens a long; the reverse crossing
    closes an existing long. Cash exits are passed to the backtest engine,
    which derives their price levels from the actual entry fill and quantity.
    No short positions or protective entry-stop orders are generated.
    """

    fast_period: int
    slow_period: int
    take_profit_cash: float
    stop_loss_cash: float
    point_value: float
    tick_size: float

    def __post_init__(self) -> None:
        for name in ("fast_period", "slow_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if self.fast_period >= self.slow_period:
            raise ValueError("fast_period must be less than slow_period.")
        for name in (
            "take_profit_cash", "stop_loss_cash", "point_value", "tick_size",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")

    @property
    def minimum_bars(self) -> int:
        """History needed for both current and previous complete SMA windows."""
        return max(self.fast_period, self.slow_period) + 1

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = normalize_ohlcv(data)
        frame["fast_sma"] = frame["close"].rolling(self.fast_period).mean()
        frame["slow_sma"] = frame["close"].rolling(self.slow_period).mean()
        frame["buy_signal"] = (frame["fast_sma"] > frame["slow_sma"]) & (
            frame["fast_sma"].shift(1) <= frame["slow_sma"].shift(1)
        )
        frame["exit_signal"] = (frame["fast_sma"] < frame["slow_sma"]) & (
            frame["fast_sma"].shift(1) >= frame["slow_sma"].shift(1)
        )
        frame["entry_stop"] = np.nan
        frame["signal"] = np.select(
            [frame["buy_signal"], frame["exit_signal"]],
            ["BUY", "EXIT"],
            default="HOLD",
        )
        for name in (
            "take_profit_cash", "stop_loss_cash", "point_value", "tick_size",
        ):
            frame[name] = getattr(self, name)
        return frame
