"""Long-only adaptation of ChartArt's RSI + Bollinger Bands v1.1.

ChartArt's original Pine strategy (January 18, 2015) requires both crossings
on the same candle. Its short-entry condition is retained only as an instruction
to close an existing long, as requested for this adaptation.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd


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


def normalize_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Copy, validate and chronologically order a single-symbol OHLCV frame.

    Open, high, low and close are required; volume is optional because the
    source strategy does not use it. Numeric strings are accepted, while
    missing prices and duplicate timestamps are rejected rather than filled.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Price data must be a pandas DataFrame.")
    if isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Price data must have single-level OHLCV columns.")
    columns = [str(column).strip().lower() for column in data.columns]
    if len(columns) != len(set(columns)):
        raise ValueError("Price data has duplicate column names.")
    frame = data.copy()
    frame.columns = columns
    required = ("open", "high", "low", "close")
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing required OHLC columns: {', '.join(missing)}")
    if isinstance(frame.index, pd.MultiIndex):
        raise ValueError("Price data must have a single timestamp index.")
    if frame.index.hasnans:
        raise ValueError("Price data has missing timestamps.")
    if frame.index.has_duplicates:
        raise ValueError("Price data has duplicate timestamps.")
    try:
        frame = frame.sort_index()
    except TypeError as exc:
        raise ValueError("Price timestamps must be chronologically sortable.") from exc

    numeric_columns = list(required) + (["volume"] if "volume" in frame else [])
    for column in numeric_columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Column {column!r} must contain numeric values.") from exc
        values = frame[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"Column {column!r} must contain finite values.")
        if column == "volume":
            if (values < 0).any():
                raise ValueError("Volume cannot be negative.")
        elif (values <= 0).any():
            raise ValueError(f"Column {column!r} must contain positive prices.")

    invalid_high = frame["high"] < frame[["open", "low", "close"]].max(axis=1)
    invalid_low = frame["low"] > frame[["open", "high", "close"]].min(axis=1)
    if invalid_high.any() or invalid_low.any():
        raise ValueError("OHLC prices must satisfy low <= open/close <= high.")
    return frame


@dataclass(frozen=True)
class RsiBollingerStrategy:
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
