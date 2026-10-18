"""Formula and event parity checks for the ChartArt long-only adaptation."""

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.strategy.rsi_bollinger import RsiBollingerStrategy


def strategy(**overrides):
    """Keep formula fixtures independent of user-editable configuration."""
    parameters = {
        "rsi_period": 6,
        "bb_period": 200,
        "bb_multiplier": 2.0,
        "rsi_threshold": 50.0,
    }
    parameters.update(overrides)
    return RsiBollingerStrategy(**parameters)


def bars(closes):
    close = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
        },
        index=pd.date_range("2024-01-01", periods=len(close), freq="D"),
    )


def test_wilder_seed_and_recursive_rsi_and_population_bands():
    data = bars([100, 99, 98, 101, 100, 103])
    result = strategy(rsi_period=2, bb_period=3).generate_signals(data)
    # Initial averages are gain=0, loss=1; each later update has alpha=1/2.
    expected_rsi = [np.nan, np.nan, 0, 75, 50, 83.33333333333333]
    np.testing.assert_allclose(result["rsi"], expected_rsi, equal_nan=True)
    assert result["bb_basis"].iloc[2] == pytest.approx(99)
    assert result["bb_lower"].iloc[2] == pytest.approx(99 - 2 * np.sqrt(2 / 3))
    assert result["bb_upper"].iloc[2] == pytest.approx(99 + 2 * np.sqrt(2 / 3))
    assert result["bb_lower"].iloc[:2].isna().all()


def test_default_source_events_require_both_crossovers_and_never_short():
    result = strategy().generate_signals(
        bars([100] * 200 + [90, 110, 120, 80])
    )
    assert np.flatnonzero(result["buy_signal"]).tolist() == [201]
    assert np.flatnonzero(result["exit_signal"]).tolist() == [203]
    assert result["signal"].iloc[201] == "BUY"
    assert result["signal"].iloc[203] == "EXIT"
    assert set(result["signal"]) == {"BUY", "EXIT", "HOLD"}
    assert result["entry_stop"].iloc[201] == result["bb_lower"].iloc[201]
    assert result["entry_stop"].iloc[201] < result["close"].iloc[201]
    assert result.loc[~result["buy_signal"], "entry_stop"].isna().all()


def test_crossovers_on_different_candles_do_not_generate_a_buy():
    result = strategy(
        rsi_period=2, bb_period=3, bb_multiplier=1
    ).generate_signals(bars([100, 95, 90, 91, 93, 96]))
    # Close reclaims its band at index 3, but RSI does not cross 50 until 5.
    assert result["close"].iloc[2] < result["bb_lower"].iloc[2]
    assert result["close"].iloc[3] > result["bb_lower"].iloc[3]
    assert result["rsi"].iloc[3] < 50
    assert result["rsi"].iloc[4] == pytest.approx(50)
    assert result["rsi"].iloc[5] > 50
    assert not result["buy_signal"].any()


def test_equal_previous_values_count_as_crossing_but_equal_current_does_not():
    result = strategy(
        rsi_period=2, bb_period=3, bb_multiplier=1
    ).generate_signals(bars([100, 99, 98, 101, 100, 103]))
    assert result["rsi"].iloc[4] == pytest.approx(50)
    assert result["close"].iloc[3] > result["bb_upper"].iloc[3]
    assert result["close"].iloc[4] < result["bb_upper"].iloc[4]
    assert not result["exit_signal"].iloc[4]
    after_equal = strategy(
        rsi_period=2, bb_period=3, bb_multiplier=0.1
    ).generate_signals(bars([100, 101, 102, 99, 100, 103]))
    assert after_equal["rsi"].iloc[4] == pytest.approx(50)
    assert after_equal["close"].iloc[4] < after_equal["bb_lower"].iloc[4]
    assert after_equal["buy_signal"].iloc[5]


def test_indicators_and_events_do_not_use_future_candles():
    data = bars([100] * 200 + [90, 110, 120, 80, 95])
    configured_strategy = strategy()
    prefix = configured_strategy.generate_signals(data.iloc[:-1])
    full = configured_strategy.generate_signals(data)
    pd.testing.assert_frame_equal(prefix, full.iloc[:-1])


def test_normalizes_titles_sorts_dates_and_does_not_mutate_input():
    data = bars([100, 99, 98, 101]).iloc[::-1].rename(columns=str.title)
    original = data.copy(deep=True)
    result = strategy(rsi_period=2, bb_period=3).generate_signals(data)
    pd.testing.assert_frame_equal(data, original)
    assert result.index.is_monotonic_increasing
    assert {"open", "high", "low", "close", "volume"} <= set(result.columns)


def test_flat_prices_have_undefined_rsi_and_no_orders():
    result = strategy(rsi_period=2, bb_period=3).generate_signals(
        bars([100] * 8)
    )
    assert result["rsi"].isna().all()
    assert not result["buy_signal"].any()
    assert not result["exit_signal"].any()


@pytest.mark.parametrize("closes, expected", [([100, 101, 102, 103], 100), ([103, 102, 101, 100], 0)])
def test_one_sided_price_changes(closes, expected):
    result = strategy(rsi_period=2, bb_period=3).generate_signals(bars(closes))
    assert result["rsi"].iloc[2:].tolist() == [expected, expected]


def test_empty_data_and_warmup_produce_no_signals():
    configured_strategy = strategy()
    assert configured_strategy.generate_signals(bars([])).empty
    short = configured_strategy.generate_signals(bars([100, 99, 98]))
    assert short["signal"].tolist() == ["HOLD"] * 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rsi_period": 0}, {"rsi_period": 2.5}, {"rsi_period": True},
        {"bb_period": -1}, {"bb_period": False},
        {"bb_multiplier": 0}, {"bb_multiplier": float("nan")},
        {"bb_multiplier": True}, {"rsi_threshold": 0},
        {"rsi_threshold": 100}, {"rsi_threshold": float("inf")},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        strategy(**kwargs)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -1, 0, "invalid"])
def test_invalid_prices_are_rejected(bad_value):
    data = bars([100, 99])
    data["close"] = data["close"].astype(object)
    data.loc[data.index[0], "close"] = bad_value
    with pytest.raises(ValueError):
        strategy().generate_signals(data)


def test_invalid_price_geometry_and_duplicate_timestamps_are_rejected():
    data = bars([100, 99])
    data.loc[data.index[0], "high"] = 99
    with pytest.raises(ValueError, match="OHLC"):
        strategy().generate_signals(data)
    duplicate = bars([100, 99])
    duplicate.index = [duplicate.index[0]] * 2
    with pytest.raises(ValueError, match="duplicate timestamps"):
        strategy().generate_signals(duplicate)


def test_missing_ohlc_duplicate_columns_and_negative_volume_are_rejected():
    configured_strategy = strategy()
    data = bars([100, 99])
    with pytest.raises(ValueError, match="Missing required OHLC"):
        configured_strategy.generate_signals(data.drop(columns="open"))
    data["Close"] = data["close"]
    with pytest.raises(ValueError, match="duplicate column"):
        configured_strategy.generate_signals(data)
    data = bars([100, 99])
    data.loc[data.index[0], "volume"] = -1
    with pytest.raises(ValueError, match="negative"):
        configured_strategy.generate_signals(data)


def test_volume_is_optional():
    result = strategy().generate_signals(bars([100]).drop(columns="volume"))
    assert result["signal"].tolist() == ["HOLD"]
