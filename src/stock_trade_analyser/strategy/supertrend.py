# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Original Pine Script © KivancOzbilgic. Python adaptation: long positions only.

"""KivancOzbilgic's Pine v4 SuperTrend strategy, adapted for long-only use."""

from dataclasses import dataclass
from datetime import date
from numbers import Integral, Real
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from .base import TradingStrategy
from ..tools.data_utils import normalize_ohlcv


def _configured_date(value: str, name: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a calendar date in YYYY-MM-DD format.")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a calendar date in YYYY-MM-DD format."
        ) from exc
    if parsed.isoformat() != value or parsed.year < 999:
        raise ValueError(
            f"{name} must use YYYY-MM-DD format with a year from 0999 to 9999."
        )
    return parsed


def _atr(frame: pd.DataFrame, period: int, use_wilder: bool) -> np.ndarray:
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    previous_close = frame["close"].shift(1).to_numpy()
    true_range = np.maximum.reduce(
        [high - low, np.abs(high - previous_close), np.abs(low - previous_close)]
    )
    if not use_wilder:
        # Pine's raw `tr` is NaN without a previous close; SMA ignores that
        # missing observation and needs `period` subsequent valid ranges.
        return pd.Series(true_range).rolling(period).mean().to_numpy()

    result = np.full(len(frame), np.nan)
    if len(frame) < period:
        return result
    # Pine atr(period) uses tr(true), whose first-bar fallback is high - low.
    true_range[0] = high[0] - low[0]
    result[period - 1] = np.mean(true_range[:period])
    for position in range(period, len(frame)):
        result[position] = (
            result[position - 1] * (period - 1) + true_range[position]
        ) / period
    return result


@dataclass(frozen=True)
class SuperTrendStrategy(TradingStrategy):
    """Generate close-confirmed BUY/EXIT events from Pine's carried bands.

    A bearish-to-bullish reversal opens a long; the opposite reversal only
    closes an existing long. No protective stop is implied by either band.
    Indicator history includes candles outside the configured order window.
    Naive timestamps are interpreted as UTC, matching the shared price loader,
    and compared in the configured exchange timezone. Parameters come from the
    caller's config, without a second set of defaults in this implementation.
    """

    atr_period: int
    atr_multiplier: float
    source: str
    change_atr: bool
    from_date: str
    to_date: str
    timezone: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.atr_period, bool)
            or not isinstance(self.atr_period, Integral)
            or self.atr_period < 1
        ):
            raise ValueError("atr_period must be a positive integer.")
        if (
            isinstance(self.atr_multiplier, bool)
            or not isinstance(self.atr_multiplier, Real)
            or not np.isfinite(self.atr_multiplier)
            or self.atr_multiplier <= 0
        ):
            raise ValueError("atr_multiplier must be a finite positive number.")
        if not isinstance(self.source, str) or self.source not in (
            "open", "high", "low", "close", "hl2", "hlc3", "ohlc4"
        ):
            raise ValueError(
                "source must be open, high, low, close, hl2, hlc3, or ohlc4."
            )
        if not isinstance(self.change_atr, bool):
            raise ValueError("change_atr must be a boolean.")
        start = _configured_date(self.from_date, "from_date")
        finish = _configured_date(self.to_date, "to_date")
        if start > finish:
            raise ValueError("from_date must be on or before to_date.")
        if not isinstance(self.timezone, str):
            raise ValueError("timezone must identify a valid IANA timezone.")
        try:
            ZoneInfo(self.timezone)
        except (ValueError, ZoneInfoNotFoundError) as exc:
            raise ValueError("timezone must identify a valid IANA timezone.") from exc

    @property
    def minimum_bars(self) -> int:
        """History required for ATR and a subsequent trend reversal."""
        return self.atr_period + (1 if self.change_atr else 2)

    def _in_window(self, index: pd.DatetimeIndex) -> np.ndarray:
        if not isinstance(index, pd.DatetimeIndex):
            raise ValueError("SuperTrend price data must have a DatetimeIndex.")
        if index.tz is None:
            index = index.tz_localize("UTC")
        local = index.tz_convert(self.timezone)
        start_date = _configured_date(self.from_date, "from_date")
        finish_date = _configured_date(self.to_date, "to_date")
        start = (start_date.year, start_date.month, start_date.day, 0, 0, 0, 0, 0)
        # timestamp(ToYear, ToMonth, ToDay, 23, 59) is exactly 23:59:00.
        finish = (
            finish_date.year, finish_date.month, finish_date.day, 23, 59, 0, 0, 0
        )
        # Calendar tuples avoid pandas' nanosecond Timestamp bounds at 9999.
        return np.array(
            [
                start
                <= (
                    timestamp.year, timestamp.month, timestamp.day,
                    timestamp.hour, timestamp.minute, timestamp.second,
                    timestamp.microsecond, timestamp.nanosecond,
                )
                <= finish
                for timestamp in local
            ],
            dtype=bool,
        )

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = normalize_ohlcv(data)
        in_window = self._in_window(frame.index)
        price_sources = {
            "open": frame["open"],
            "high": frame["high"],
            "low": frame["low"],
            "close": frame["close"],
            "hl2": (frame["high"] + frame["low"]) / 2,
            "hlc3": (frame["high"] + frame["low"] + frame["close"]) / 3,
            "ohlc4": frame[["open", "high", "low", "close"]].mean(axis=1),
        }
        atr = _atr(frame, self.atr_period, self.change_atr)
        price = price_sources[self.source].to_numpy()
        close = frame["close"].to_numpy()
        up = price - self.atr_multiplier * atr
        dn = price + self.atr_multiplier * atr
        trend = np.ones(len(frame), dtype=np.int8)
        for position in range(len(frame)):
            # nz(previous band, current band) matters at ATR's first valid row.
            up1 = (
                up[position - 1]
                if position > 0 and not np.isnan(up[position - 1])
                else up[position]
            )
            dn1 = (
                dn[position - 1]
                if position > 0 and not np.isnan(dn[position - 1])
                else dn[position]
            )
            if position > 0:
                if close[position - 1] > up1:
                    up[position] = max(up[position], up1)
                if close[position - 1] < dn1:
                    dn[position] = min(dn[position], dn1)
                trend[position] = trend[position - 1]
            if trend[position] == -1 and close[position] > dn1:
                trend[position] = 1
            elif trend[position] == 1 and close[position] < up1:
                trend[position] = -1

        # Pine trend[1] is unavailable on the first candle, so even an ATR(1)
        # reversal there must not emit an event.
        previous_trend = np.concatenate(([0], trend[:-1])) if len(frame) else trend
        frame["atr"] = atr
        frame["up"] = up
        frame["dn"] = dn
        frame["trend"] = trend
        frame["supertrend"] = np.where(trend == 1, up, dn)
        frame["in_window"] = in_window
        frame["buy_signal"] = (trend == 1) & (previous_trend == -1) & in_window
        frame["exit_signal"] = (trend == -1) & (previous_trend == 1) & in_window
        frame["entry_stop"] = np.nan
        frame["signal"] = np.select(
            [frame["buy_signal"], frame["exit_signal"]],
            ["BUY", "EXIT"],
            default="HOLD",
        )
        return frame
