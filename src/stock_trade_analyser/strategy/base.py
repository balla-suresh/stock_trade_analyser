"""Common strategy hooks used by the shared scanner and backtest runner."""

from typing import Any


class TradingStrategy:
    """Keep indicator-specific history requirements and diagnostics together.

    Implementations provide ``generate_signals`` and ``minimum_bars``. The
    runner can use these hooks without knowing which indicators they contain.
    This base has no constructor or fields, so frozen dataclass strategies can
    inherit it without changing their configuration or initialization.
    """

    @property
    def minimum_bars(self) -> int:
        """Number of candles needed before signals can be evaluated."""
        raise NotImplementedError

    def data_message(self, signals: Any) -> str:
        """Return an optional explanation about the generated signal data."""
        return ""
