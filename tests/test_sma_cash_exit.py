"""Completed SMA crossover signals and cash-exit parameters from the Pine port."""

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.strategy.sma_cash_exit import SmaCashExitStrategy


SETTINGS = {
    "fast_period": 2,
    "slow_period": 3,
    "take_profit_cash": 100.0,
    "stop_loss_cash": 20.0,
    "point_value": 1.0,
    "tick_size": 0.01,
}


def prices(closes=(10, 10, 10, 10, 12, 14, 14, 10, 8, 8, 12, 14)):
    return pd.DataFrame(
        {"open": closes, "high": np.asarray(closes) + 1, "low": np.asarray(closes) - 1,
         "close": closes, "volume": 1_000},
        index=pd.date_range("2025-01-01", periods=len(closes), name="date"),
    )


def test_complete_crossovers_emit_buy_and_close_long_events_with_hand_calculated_averages():
    signals = SmaCashExitStrategy(**SETTINGS).generate_signals(prices())

    assert signals.index[signals["buy_signal"]].tolist() == [signals.index[4], signals.index[10]]
    assert signals.index[signals["exit_signal"]].tolist() == [signals.index[7]]
    assert signals["fast_sma"].iloc[4] == 11
    assert signals["slow_sma"].iloc[4] == pytest.approx(32 / 3)
    assert signals["fast_sma"].iloc[7] == 12
    assert signals["slow_sma"].iloc[7] == pytest.approx(38 / 3)
    assert signals["signal"].tolist() == [
        "HOLD", "HOLD", "HOLD", "HOLD", "BUY", "HOLD",
        "HOLD", "EXIT", "HOLD", "HOLD", "BUY", "HOLD",
    ]
    assert signals["entry_stop"].isna().all()
    assert not (signals["buy_signal"] & signals["exit_signal"]).any()


def test_initial_uptrend_does_not_buy_without_a_previous_complete_crossover():
    strategy = SmaCashExitStrategy(**SETTINGS)
    signals = strategy.generate_signals(prices((10, 11, 12, 13, 14)))
    assert signals["fast_sma"].iloc[:1].isna().all()
    assert signals["slow_sma"].iloc[:2].isna().all()
    assert not signals["buy_signal"].any()
    assert not signals["exit_signal"].any()
    assert strategy.minimum_bars == 4


def test_current_equality_is_not_a_cross_but_leaving_previous_equality_is():
    signals = SmaCashExitStrategy(**SETTINGS).generate_signals(prices((10, 10, 10, 12)))
    assert signals["signal"].tolist() == ["HOLD", "HOLD", "HOLD", "BUY"]


def test_signal_generation_preserves_input_and_earlier_signals_when_future_changes():
    strategy = SmaCashExitStrategy(**SETTINGS)
    history = prices()
    original = history.copy(deep=True)
    prefix = strategy.generate_signals(history.iloc[:8])
    complete = strategy.generate_signals(history)
    pd.testing.assert_frame_equal(prefix, complete.iloc[:8])
    pd.testing.assert_frame_equal(history, original)


def test_cash_configuration_is_carried_without_changing_signal_timing():
    parameters = {**SETTINGS, "take_profit_cash": 125.5, "stop_loss_cash": 12.25,
                  "point_value": 2.0, "tick_size": 0.05}
    signals = SmaCashExitStrategy(**parameters).generate_signals(prices())
    reference = SmaCashExitStrategy(**SETTINGS).generate_signals(prices())
    pd.testing.assert_series_equal(signals["signal"], reference["signal"])
    for name in ("take_profit_cash", "stop_loss_cash", "point_value", "tick_size"):
        assert signals[name].tolist() == [parameters[name]] * len(signals)


@pytest.mark.parametrize("name,value", [
    ("fast_period", 0), ("fast_period", True), ("fast_period", 2.0),
    ("slow_period", -1), ("slow_period", False), ("slow_period", "3"),
    ("fast_period", 3), ("fast_period", 4),
])
def test_invalid_periods_identify_the_parameter(name, value):
    with pytest.raises(ValueError, match=name):
        SmaCashExitStrategy(**{**SETTINGS, name: value})


@pytest.mark.parametrize("name", ["take_profit_cash", "stop_loss_cash", "point_value", "tick_size"])
@pytest.mark.parametrize("value", [0, -1, True, "1", np.nan, np.inf])
def test_cash_and_instrument_settings_must_be_finite_positive_numbers(name, value):
    with pytest.raises(ValueError, match=name):
        SmaCashExitStrategy(**{**SETTINGS, name: value})


def test_subtick_cash_targets_remain_valid_fractional_values():
    strategy = SmaCashExitStrategy(**{**SETTINGS, "take_profit_cash": 0.001, "tick_size": 0.05})
    assert strategy.generate_signals(prices())["take_profit_cash"].iloc[-1] == 0.001
