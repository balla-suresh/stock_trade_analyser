# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# Original Pine Script © kingshukghosh71. Python adaptation: long positions only.

"""Kingshuk Ghosh's Wyckoff Schematic events with configurable long-only orders.

The supplied Pine is an indicator rather than an order-entry strategy. Its raw
event conditions and sequential phase transitions are preserved here, while the
caller explicitly chooses which events buy and which close an existing long.
Rendering, label spacing and the decorative wave chart do not filter events.
"""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np
import pandas as pd

from .base import TradingStrategy
from ..tools.data_utils import normalize_ohlcv


EVENT_NAMES = (
    "PS", "SC", "AR", "ST", "SPRING", "TEST", "SOS", "LPS",
    "PSY", "BC", "AR_DIST", "ST_DIST", "UTAD", "LPSY",
)
EVENT_COLUMNS = tuple(event.lower() for event in EVENT_NAMES)


def _phase_history(events: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Apply both Pine if/else chains in their original order on each bar."""
    phases = np.empty(len(events), dtype=object)
    patterns = np.empty(len(events), dtype=object)
    phase = "None"
    pattern = "None"
    for position, flags in enumerate(events.itertuples(index=False, name=None)):
        (
            ps, sc, ar, _st, spring, test, sos, lps,
            psy, bc, ar_dist, _st_dist, utad, lpsy,
        ) = flags
        if ps or sc:
            phase, pattern = "Phase A", "Accumulation"
        elif ar and phase == "Phase A":
            phase, pattern = "Phase B", "Accumulation"
        elif test and phase == "Phase B":
            phase, pattern = "Phase C", "Accumulation"
        elif spring and phase == "Phase C":
            pattern = "Spring"
        elif sos and phase in ("Phase C", "Phase B"):
            phase, pattern = "Phase D", "SOS"
        elif lps and phase == "Phase D":
            phase, pattern = "Phase E", "LPS"

        # This is a second independent chain in the source, so a distribution
        # event can overwrite the accumulation update on the very same candle.
        if psy or bc:
            phase, pattern = "Phase A (Dist)", "Distribution"
        elif ar_dist and phase == "Phase A (Dist)":
            phase, pattern = "Phase B (Dist)", "Distribution"
        elif utad and phase == "Phase B (Dist)":
            phase, pattern = "Phase C (Dist)", "UTAD"
        elif lpsy and phase == "Phase C (Dist)":
            phase, pattern = "Phase D (Dist)", "LPSY"

        phases[position] = phase
        patterns[position] = pattern
    return phases, patterns


@dataclass(frozen=True)
class WyckoffStrategy(TradingStrategy):
    """Port all fourteen source events without lookahead or phase gating.

    All detection thresholds and order-event choices come from configuration.
    Percentage fields use fractions: ``0.01`` means one percent. Rolling
    extrema include the available history; moving averages need a full window.
    With the supplied defaults their longer volume window also prevents events
    before the extrema windows fill. A valuewhen reference includes the current
    candle when its condition is true. Exit events take priority when both
    configured directions match, and the original raw events remain visible.
    """

    lookback: int
    volume_multiplier: float
    climax_volume_addition: float
    range_period: int
    narrow_range_multiplier: float
    preliminary_period: int
    climax_period: int
    sweep_period: int
    rally_change_pct: float
    secondary_test_tolerance_pct: float
    spring_recovery_pct: float
    test_range_tolerance: float
    test_volume_multiplier: float
    sos_volume_multiplier: float
    buy_events: tuple[str, ...] | list[str]
    exit_events: tuple[str, ...] | list[str]

    def __post_init__(self) -> None:
        for name in (
            "lookback", "range_period", "preliminary_period", "climax_period",
            "sweep_period",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 20 <= self.lookback <= 300:
            raise ValueError("lookback must be between 20 and 300.")
        for name in (
            "volume_multiplier", "narrow_range_multiplier",
            "test_volume_multiplier", "sos_volume_multiplier",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be a finite positive number.")
        if self.volume_multiplier < 1:
            raise ValueError("volume_multiplier must be at least one.")
        value = self.climax_volume_addition
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("climax_volume_addition must be a finite nonnegative number.")
        for name in (
            "rally_change_pct", "secondary_test_tolerance_pct",
            "spring_recovery_pct", "test_range_tolerance",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or not 0 < value < 1
            ):
                raise ValueError(f"{name} must be a fraction greater than zero and below one.")
        for name in ("buy_events", "exit_events"):
            configured = getattr(self, name)
            if not isinstance(configured, (list, tuple)) or any(
                not isinstance(event, str) or event not in EVENT_NAMES
                for event in configured
            ):
                raise ValueError(
                    f"{name} must be a list of event names from {', '.join(EVENT_NAMES)}."
                )
            if len(set(configured)) != len(configured):
                raise ValueError(f"{name} must not contain duplicate event names.")
            # The dataclass is frozen; do not retain a mutable caller-owned list.
            object.__setattr__(self, name, tuple(configured))

    @property
    def minimum_bars(self) -> int:
        """History required for both full moving-average windows."""
        return max(self.lookback, self.range_period)

    def data_message(self, signals: pd.DataFrame) -> str:
        if not signals["volume"].gt(0).any():
            return (
                "Volume is zero for every bar; Wyckoff volume-based events "
                "cannot be detected. Use an instrument with reported trading volume."
            )
        return ""

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        frame = normalize_ohlcv(data)
        if "volume" not in frame:
            raise ValueError(
                "Wyckoff requires a volume column; provide OHLCV data with volume."
            )
        open_ = frame["open"]
        high = frame["high"]
        low = frame["low"]
        close = frame["close"]
        volume = frame["volume"]
        avg_volume = volume.rolling(self.lookback).mean()
        price_range = high - low
        avg_range = price_range.rolling(self.range_period).mean()
        high_volume = volume > avg_volume * self.volume_multiplier
        climax_volume = volume > avg_volume * (
            self.volume_multiplier + self.climax_volume_addition
        )
        narrow_range = price_range < avg_range * self.narrow_range_multiplier
        resistance = high.rolling(self.lookback, min_periods=1).max()
        support = low.rolling(self.lookback, min_periods=1).min()
        range_size = resistance - support
        mid_range = (resistance + support) / 2

        events = pd.DataFrame(index=frame.index)
        events["ps"] = (
            (low == low.rolling(self.preliminary_period, min_periods=1).min())
            & high_volume & (close > open_)
        )
        events["sc"] = (
            climax_volume & (close < open_)
            & (low <= low.rolling(self.climax_period, min_periods=1).min())
            & (close < close.shift(1))
        )
        events["ar"] = (
            (close > open_) & (volume > avg_volume)
            & (
                low.shift(1)
                == low.shift(1).rolling(self.climax_period, min_periods=1).min()
            )
            & (close > close.shift(1) * (1 + self.rally_change_pct))
        )
        # where(...).ffill() is ta.valuewhen(condition, source, 0): the most
        # recent matching source, including this candle, with NaN before any.
        last_sc_low = low.where(events["sc"]).ffill()
        events["st"] = (
            (low < last_sc_low * (1 + self.secondary_test_tolerance_pct))
            & (low > last_sc_low * (1 - self.secondary_test_tolerance_pct))
            & (volume < avg_volume) & (close > open_)
        )
        events["spring"] = (
            (low < low.shift(1).rolling(self.sweep_period, min_periods=1).min())
            & (close > open_) & (volume > avg_volume)
            & (close > low * (1 + self.spring_recovery_pct))
        )
        events["test"] = (
            ((low - last_sc_low).abs() < range_size * self.test_range_tolerance)
            & (volume < avg_volume * self.test_volume_multiplier) & narrow_range
        )
        events["sos"] = (
            (close > close.shift(1))
            & (volume > avg_volume * self.sos_volume_multiplier)
            & (close - open_ > avg_range) & (close > mid_range)
        )
        last_spring_or_st_low = low.where(events["spring"] | events["st"]).ffill()
        events["lps"] = (
            (low > last_spring_or_st_low) & (close > open_)
            & (volume < avg_volume) & (close > mid_range)
        )
        events["psy"] = (
            (high == high.rolling(self.preliminary_period, min_periods=1).max())
            & high_volume & (close < open_)
        )
        events["bc"] = (
            climax_volume & (close > open_)
            & (high >= high.rolling(self.climax_period, min_periods=1).max())
        )
        events["ar_dist"] = (
            (close < open_) & (volume > avg_volume)
            & (
                high.shift(1)
                == high.shift(1).rolling(self.climax_period, min_periods=1).max()
            )
            & (close < close.shift(1) * (1 - self.rally_change_pct))
        )
        last_bc_high = high.where(events["bc"]).ffill()
        events["st_dist"] = (
            (high > last_bc_high * (1 - self.secondary_test_tolerance_pct))
            & (high < last_bc_high * (1 + self.secondary_test_tolerance_pct))
            & (volume < avg_volume)
        )
        events["utad"] = (
            (high > high.shift(1).rolling(self.sweep_period, min_periods=1).max())
            & (close < open_) & (volume > avg_volume)
            & (close < high * (1 - self.spring_recovery_pct))
        )
        last_utad_or_st_high = high.where(events["utad"] | events["st_dist"]).ffill()
        events["lpsy"] = (
            (high < last_utad_or_st_high) & (close < open_)
            & (volume < avg_volume) & (close < mid_range)
        )

        frame["avg_volume"] = avg_volume
        frame["avg_range"] = avg_range
        frame["price_range"] = price_range
        frame["range_size"] = range_size
        frame["narrow_range"] = narrow_range
        frame["high_volume"] = high_volume
        frame["climax_volume"] = climax_volume
        frame["support"] = support
        frame["resistance"] = resistance
        frame["mid_range"] = mid_range
        frame[list(EVENT_COLUMNS)] = events[list(EVENT_COLUMNS)]
        frame["phase"], frame["pattern"] = _phase_history(events[list(EVENT_COLUMNS)])
        frame["events"] = [
            ",".join(name for name, matches in zip(EVENT_NAMES, flags) if matches)
            for flags in events[list(EVENT_COLUMNS)].itertuples(index=False, name=None)
        ]
        frame["exit_signal"] = (
            events[[name.lower() for name in self.exit_events]].any(axis=1)
            if self.exit_events else False
        )
        frame["buy_signal"] = (
            events[[name.lower() for name in self.buy_events]].any(axis=1)
            if self.buy_events else False
        ) & ~frame["exit_signal"]
        frame["entry_stop"] = np.nan
        frame["signal"] = np.select(
            [frame["exit_signal"], frame["buy_signal"]],
            ["EXIT", "BUY"],
            default="HOLD",
        )
        return frame
