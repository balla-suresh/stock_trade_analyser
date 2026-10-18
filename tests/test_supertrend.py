"""Numerical and event checks for the supplied Pine v4 SuperTrend strategy."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.strategy.supertrend import SuperTrendStrategy


def strategy(**overrides):
    parameters = {
        "atr_period": 2,
        "atr_multiplier": 1.0,
        "source": "hl2",
        "change_atr": True,
        "from_date": "2018-09-01",
        "to_date": "9999-01-01",
        "timezone": "Asia/Kolkata",
    }
    parameters.update(overrides)
    return SuperTrendStrategy(**parameters)


def prices(close=(10, 11, 12, 9, 8, 13, 14), index=None):
    close = np.asarray(close, dtype=float)
    if index is None:
        index = pd.date_range("2024-01-01", periods=len(close))
    return pd.DataFrame(
        {"open": close, "high": close + 1, "low": close - 1, "close": close},
        index=index,
    )


def test_wilder_atr_and_pine_carried_bands_match_hand_calculation():
    result = strategy().generate_signals(prices())
    # True ranges: 2, 2, 2, 4, 2, 6, 2. RMA(2) seeds at the second bar.
    np.testing.assert_allclose(
        result.atr, [np.nan, 2, 2, 3, 2.5, 4.25, 3.125], equal_nan=True
    )
    np.testing.assert_allclose(
        result.up, [np.nan, 9, 10, 10, 5.5, 8.75, 10.875], equal_nan=True
    )
    np.testing.assert_allclose(
        result.dn, [np.nan, 13, 13, 12, 10.5, 10.5, 17.125], equal_nan=True
    )
    assert result.trend.tolist() == [1, 1, 1, -1, -1, 1, 1]
    assert result.signal.tolist() == ["HOLD", "HOLD", "HOLD", "EXIT", "HOLD", "BUY", "HOLD"]
    np.testing.assert_allclose(
        result.supertrend,
        [np.nan, 9, 10, 12, 10.5, 8.75, 10.875],
        equal_nan=True,
    )
    assert result.entry_stop.isna().all()


def test_sma_atr_uses_raw_tr_and_seeds_one_bar_later():
    result = strategy(change_atr=False).generate_signals(prices())
    # Unlike atr(), raw Pine tr is unavailable on the very first candle.
    np.testing.assert_allclose(
        result.atr, [np.nan, np.nan, 2, 3, 3, 4, 4], equal_nan=True
    )
    np.testing.assert_allclose(
        result.up, [np.nan, np.nan, 10, 10, 5, 9, 10], equal_nan=True
    )
    np.testing.assert_allclose(
        result.dn, [np.nan, np.nan, 14, 12, 11, 11, 18], equal_nan=True
    )
    assert result.trend.tolist() == [1, 1, 1, -1, -1, 1, 1]


def test_reversals_compare_strictly_with_previous_carried_bands():
    result = strategy(atr_period=1).generate_signals(prices([100, 98, 97, 99, 100, 101]))
    assert result.trend.tolist() == [1, 1, -1, -1, 1, 1]
    # Equality to previous up/dn is not a reversal.
    assert result.iloc[1].close == result.iloc[0].up
    assert result.iloc[1].signal == "HOLD"
    assert result.iloc[3].close == result.iloc[2].dn
    assert result.iloc[3].signal == "HOLD"
    # The current band is already on the other side; the old band is decisive.
    assert result.iloc[2].close > result.iloc[2].up
    assert result.iloc[2].signal == "EXIT"
    assert result.iloc[4].close < result.iloc[4].dn
    assert result.iloc[4].signal == "BUY"


def test_initial_bullish_state_does_not_open_a_long():
    result = strategy(atr_period=1).generate_signals(prices([10, 11, 12, 13]))
    assert result.trend.eq(1).all()
    assert not result.buy_signal.any()
    assert not result.exit_signal.any()


def test_first_bar_reversal_has_no_previous_trend_to_emit_an_exit():
    frame = pd.DataFrame(
        {"open": [100], "high": [110], "low": [90], "close": [91]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    result = strategy(atr_period=1, atr_multiplier=0.1, source="high").generate_signals(frame)
    assert result.trend.tolist() == [-1]
    assert result.signal.tolist() == ["HOLD"]


@pytest.mark.parametrize(
    ("source", "expected"),
    [("open", 100), ("high", 110), ("low", 90), ("close", 105),
     ("hl2", 100), ("hlc3", 305 / 3), ("ohlc4", 405 / 4)],
)
def test_source_price_inputs(source, expected):
    frame = pd.DataFrame(
        {"open": [100], "high": [110], "low": [90], "close": [105]},
        index=pd.date_range("2024-01-01", periods=1),
    )
    result = strategy(atr_period=1, source=source).generate_signals(frame)
    assert result.iloc[0].up == pytest.approx(expected - 20)
    assert result.iloc[0].dn == pytest.approx(expected + 20)


@pytest.mark.parametrize("change_atr", [True, False])
def test_indicator_and_events_do_not_change_when_future_bars_are_appended(change_atr):
    frame = prices()
    partial = strategy(change_atr=change_atr).generate_signals(frame.iloc[:5])
    full = strategy(change_atr=change_atr).generate_signals(frame)
    pd.testing.assert_frame_equal(partial, full.iloc[:5])


def test_window_masks_orders_without_resetting_indicator_history():
    frame = prices()
    full = strategy().generate_signals(frame)
    limited = strategy(from_date="2024-01-05", to_date="2024-01-06").generate_signals(frame)
    pd.testing.assert_series_equal(full.trend, limited.trend)
    pd.testing.assert_series_equal(full.atr, limited.atr)
    assert limited.in_window.tolist() == [False, False, False, False, True, True, False]
    assert not limited.exit_signal.any()
    assert limited.buy_signal.tolist() == [False, False, False, False, False, True, False]


def test_window_includes_local_start_and_finish_but_not_later_seconds():
    local_index = pd.DatetimeIndex(
        ["2024-01-01 23:59:59", "2024-01-02 00:00:00", "2024-01-02 23:59:00",
         "2024-01-02 23:59:01", "2024-01-03 00:00:00"],
        tz="Asia/Kolkata",
    )
    frame = prices([10] * 5, index=local_index.tz_convert("UTC").tz_localize(None))
    result = strategy(from_date="2024-01-02", to_date="2024-01-02").generate_signals(frame)
    assert result.in_window.tolist() == [False, True, True, False, False]
    pd.testing.assert_index_equal(result.index, frame.index)


def test_aware_timestamps_use_configured_exchange_calendar():
    index = pd.DatetimeIndex(["2024-01-01 20:00:00", "2024-01-02 20:00:00"], tz="UTC")
    result = strategy(from_date="2024-01-02", to_date="2024-01-02").generate_signals(
        prices([10, 11], index=index)
    )
    assert result.in_window.tolist() == [True, False]


def test_default_far_future_finish_does_not_overflow_nanosecond_timestamps():
    result = strategy(to_date="9999-01-01").generate_signals(prices())
    assert result.in_window.all()


def test_empty_history_produces_correctly_typed_empty_signals():
    result = strategy().generate_signals(prices([]))
    assert result.empty
    assert result.buy_signal.dtype == bool
    assert result.exit_signal.dtype == bool
    assert result.in_window.dtype == bool


def test_input_is_not_mutated_and_uppercase_columns_are_supported():
    original = prices().rename(columns=str.upper).iloc[::-1]
    saved = original.copy(deep=True)
    result = strategy().generate_signals(original)
    pd.testing.assert_frame_equal(original, saved)
    assert result.index.is_monotonic_increasing
    assert "close" in result
    assert set(result.signal.unique()) <= {"BUY", "EXIT", "HOLD"}


def test_missing_timestamps_are_not_silently_treated_as_epoch_nanoseconds():
    frame = prices().reset_index(drop=True)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        strategy().generate_signals(frame)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("atr_period", 0), ("atr_period", -1), ("atr_period", 1.5),
        ("atr_period", True), ("atr_multiplier", 0), ("atr_multiplier", -1),
        ("atr_multiplier", np.nan), ("atr_multiplier", np.inf),
        ("atr_multiplier", True), ("source", "volume"), ("source", None),
        ("change_atr", "false"), ("change_atr", 1),
        ("from_date", "2024-02-30"), ("from_date", "20240101"),
        ("from_date", "0001-01-01"), ("from_date", None),
        ("to_date", "not-a-date"), ("to_date", 20240101),
        ("timezone", "Unknown/Exchange"), ("timezone", ""), ("timezone", None),
    ],
)
def test_invalid_configuration_has_clear_field_error(name, value):
    with pytest.raises(ValueError, match=name):
        strategy(**{name: value})


def test_reversed_calendar_window_is_rejected():
    with pytest.raises(ValueError, match="from_date must be on or before to_date"):
        strategy(from_date="2025-01-01", to_date="2024-01-01")


def test_parameters_have_no_hardcoded_python_defaults_and_are_immutable():
    with pytest.raises(TypeError):
        SuperTrendStrategy()
    assert len(fields(SuperTrendStrategy)) == 7
    with pytest.raises(FrozenInstanceError):
        strategy().atr_period = 14
