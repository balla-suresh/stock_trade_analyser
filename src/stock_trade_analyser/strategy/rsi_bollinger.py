"""Long-only adaptation of ChartArt's RSI + Bollinger Bands v1.1.

ChartArt's original Pine strategy (January 18, 2015) requires both crossings
on the same candle. Its short-entry condition is retained only as an instruction
to close an existing long, as requested for this adaptation.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .base import TradingStrategy
from ..tools.data_utils import normalize_ohlcv


def _wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    """Seed Wilder averages with the first ``period`` valid price changes."""
    result = np.full(len(close), np.nan, dtype=float)
    if len(close) <= period:
        return pd.Series(result, index=close.index, name="rsi")

    changes = np.diff(close.to_numpy(dtype=float))
    gains = np.maximum(changes, 0.0)
    losses = np.maximum(-changes, 0.0)
    average_gain = float(np.mean(gains[:period]))
    average_loss = float(np.mean(losses[:period]))
    for position in range(period, len(close)):
        if position > period:
            average_gain = (
                average_gain * (period - 1) + gains[position - 1]
            ) / period
            average_loss = (
                average_loss * (period - 1) + losses[position - 1]
            ) / period
        if average_loss == 0.0:
            # The source formula's 0/0 is undefined for a completely flat run.
            result[position] = 100.0 if average_gain > 0.0 else np.nan
        else:
            result[position] = 100.0 - 100.0 / (
                1.0 + average_gain / average_loss
            )
    return pd.Series(result, index=close.index, name="rsi")


@dataclass(frozen=True)
class RsiBollingerStrategy(TradingStrategy):
    """Generate buy and close-long events without creating short positions.

    Entries require RSI crossing above the configured threshold and close
    crossing above the lower band on the same completed candle. Exits require
    the mirrored RSI/upper band crossings. ``entry_stop`` preserves the original Pine *entry* stop;
    it is not a protective stop-loss. Because close has already crossed above
    that price, the order is activated when the signal is generated.
    """

    rsi_period: int
    bb_period: int
    bb_multiplier: float
    rsi_threshold: float

    def __post_init__(self) -> None:
        for name in ("rsi_period", "bb_period"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if (
            isinstance(self.bb_multiplier, bool)
            or not isinstance(self.bb_multiplier, Real)
            or not np.isfinite(self.bb_multiplier)
            or self.bb_multiplier <= 0
        ):
            raise ValueError("bb_multiplier must be a finite positive number.")
        if (
            isinstance(self.rsi_threshold, bool)
            or not isinstance(self.rsi_threshold, Real)
            or not np.isfinite(self.rsi_threshold)
            or not 0 < self.rsi_threshold < 100
        ):
            raise ValueError("rsi_threshold must be between 0 and 100.")

    @property
    def minimum_bars(self) -> int:
        """History required for both indicator crossings."""
        return max(self.bb_period + 1, self.rsi_period + 2)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = normalize_ohlcv(data)
        close = frame["close"]
        frame["rsi"] = _wilder_rsi(close, self.rsi_period)
        frame["bb_basis"] = close.rolling(self.bb_period).mean()
        deviation = close.rolling(self.bb_period).std(ddof=0) * self.bb_multiplier
        frame["bb_upper"] = frame["bb_basis"] + deviation
        frame["bb_lower"] = frame["bb_basis"] - deviation

        rsi_cross_up = (frame["rsi"] > self.rsi_threshold) & (
            frame["rsi"].shift(1) <= self.rsi_threshold
        )
        price_cross_up = (close > frame["bb_lower"]) & (
            close.shift(1) <= frame["bb_lower"].shift(1)
        )
        rsi_cross_down = (frame["rsi"] < self.rsi_threshold) & (
            frame["rsi"].shift(1) >= self.rsi_threshold
        )
        price_cross_down = (close < frame["bb_upper"]) & (
            close.shift(1) >= frame["bb_upper"].shift(1)
        )
        frame["buy_signal"] = rsi_cross_up & price_cross_up
        frame["exit_signal"] = rsi_cross_down & price_cross_down
        frame["entry_stop"] = frame["bb_lower"].where(frame["buy_signal"])
        frame["signal"] = np.select(
            [frame["buy_signal"], frame["exit_signal"]],
            ["BUY", "EXIT"],
            default="HOLD",
        )
        return frame
