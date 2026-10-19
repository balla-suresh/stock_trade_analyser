"""Hand-calculated cash SL/TP fills, order timing, and point-value accounting."""

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.backtest.engine import BacktestConfig, LongOnlyBacktester


POLICY = {"take_profit_cash": 100.0, "stop_loss_cash": 20.0, "point_value": 1.0, "tick_size": 0.01}


def signals(candles, *, buys=(0,), exits=(), **policy):
    """Independent OHLC signals isolate broker behavior from SMA calculations."""
    frame = pd.DataFrame(
        candles, columns=["open", "high", "low", "close"],
        index=pd.date_range("2025-01-01", periods=len(candles), name="date"),
        dtype=float,
    )
    frame["buy_signal"] = False
    frame["exit_signal"] = False
    frame["entry_stop"] = np.nan
    for offset in buys:
        frame.loc[frame.index[offset], "buy_signal"] = True
    for offset in exits:
        frame.loc[frame.index[offset], "exit_signal"] = True
    for name, value in {**POLICY, **policy}.items():
        frame[name] = value
    return frame


@pytest.mark.parametrize("quantity,point_value,active_bar,expected_price,expected_profit,reason", [
    (1, 1, (100, 205, 95, 105), 200, 100, "TAKE_PROFIT"),
    (1, 1, (100, 110, 75, 95), 80, -20, "STOP_LOSS"),
    (10, 1, (100, 111, 99, 105), 110, 100, "TAKE_PROFIT"),
    (10, 1, (100, 105, 97, 100), 98, -20, "STOP_LOSS"),
    (5, 2, (100, 111, 99, 105), 110, 100, "TAKE_PROFIT"),
    (5, 2, (100, 105, 97, 100), 98, -20, "STOP_LOSS"),
])
def test_cash_targets_apply_to_the_whole_position_before_costs(
    quantity, point_value, active_bar, expected_price, expected_profit, reason,
):
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), active_bar], point_value=point_value)
    result = LongOnlyBacktester(BacktestConfig(quantity=quantity)).run(frame)
    trade = result.trades.iloc[0]
    assert trade["entry_time"] == frame.index[1]
    assert trade["entry_price"] == 100
    assert trade["exit_time"] == frame.index[2]
    assert trade["exit_price"] == pytest.approx(expected_price)
    assert trade["gross_profit"] == pytest.approx(expected_profit)
    assert trade["net_profit"] == pytest.approx(expected_profit)
    assert trade["exit_reason"] == reason
    assert trade["return_pct"] == pytest.approx(expected_profit / (100 * quantity * point_value) * 100)
    assert result.metrics["final_equity"] == pytest.approx(100_000 + expected_profit)
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == result.metrics["total_trades"] == 1


def test_new_position_is_unprotected_during_its_entry_bar():
    frame = signals([
        (99, 100, 98, 99),
        (100, 250, 50, 100),  # Both levels are crossed before the bracket exists.
        (100, 110, 90, 100),
        (100, 205, 95, 105),
    ])
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    assert result.equity.loc[frame.index[1], "quantity"] == 1
    assert result.equity.loc[frame.index[2], "quantity"] == 1
    assert result.trades.iloc[0]["exit_time"] == frame.index[3]
    assert result.trades.iloc[0]["exit_reason"] == "TAKE_PROFIT"


def test_entry_close_below_new_stop_exits_next_open_even_after_price_recovers():
    frame = signals([
        (99, 100, 98, 99),
        (100, 101, 70, 75),
        (90, 100, 85, 95),  # The activated stop is now a pending market order.
    ])
    result = LongOnlyBacktester(BacktestConfig(initial_capital=1_000)).run(frame)
    trade = result.trades.iloc[0]
    assert trade["exit_time"] == frame.index[2]
    assert trade["exit_price"] == 90
    assert trade["exit_reason"] == "STOP_LOSS"
    assert trade["gross_profit"] == -10
    assert result.metrics["max_drawdown_pct"] == pytest.approx(2.5)


@pytest.mark.parametrize("active_bar,price,profit,reason", [
    ((220, 225, 210, 221), 220, 120, "TAKE_PROFIT"),
    ((70, 75, 65, 72), 70, -30, "STOP_LOSS"),
])
def test_gap_through_a_bracket_fills_at_the_open(active_bar, price, profit, reason):
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), active_bar])
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    trade = result.trades.iloc[0]
    assert trade["exit_price"] == price
    assert trade["gross_profit"] == profit
    assert trade["exit_reason"] == reason


@pytest.mark.parametrize("active_bar,price,reason", [
    ((150, 205, 75, 150), 200, "TAKE_PROFIT"),  # Open is nearer the high.
    ((100, 205, 75, 100), 80, "STOP_LOSS"),    # Open is nearer the low.
    ((140, 205, 75, 140), 80, "STOP_LOSS"),    # Equal distances: visit the low first.
])
def test_both_touched_brackets_follow_the_ohlc_broker_path(active_bar, price, reason):
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), active_bar])
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    assert result.trades.iloc[0]["exit_price"] == price
    assert result.trades.iloc[0]["exit_reason"] == reason


def test_next_open_market_exit_has_priority_over_a_gap_stop():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (70, 75, 65, 72)], exits=(1,),
    )
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    assert result.trades.iloc[0]["exit_price"] == 70
    assert result.trades.iloc[0]["exit_reason"] == "EXIT_SIGNAL"
    assert result.metrics["total_sells"] == 1


def test_limit_profit_retains_its_price_while_both_fills_pay_commission():
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), (110, 210, 95, 120)])
    result = LongOnlyBacktester(BacktestConfig(commission_bps=10, slippage_bps=10)).run(frame)
    trade = result.trades.iloc[0]
    # Entry 100.10 includes 0.1% slippage; its limit is 100 cash above that fill.
    assert trade["entry_price"] == pytest.approx(100.1)
    assert trade["exit_price"] == pytest.approx(200.1)
    assert trade["gross_profit"] == pytest.approx(100)
    assert trade["fees"] == pytest.approx(0.3002)
    assert trade["net_profit"] == pytest.approx(99.6998)
    assert result.metrics["final_equity"] == pytest.approx(100_099.6998)


def test_stop_fill_has_adverse_slippage_and_fees_after_cash_distance():
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), (95, 110, 75, 95)])
    result = LongOnlyBacktester(BacktestConfig(commission_bps=10, slippage_bps=10)).run(frame)
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.1)
    assert trade["exit_price"] == pytest.approx(80.0199)
    assert trade["gross_profit"] == pytest.approx(-20.0801)
    assert trade["fees"] == pytest.approx(0.1801199)
    assert trade["net_profit"] == pytest.approx(-20.2602199)
    assert trade["exit_reason"] == "STOP_LOSS"


def test_fractional_tick_distance_is_not_rounded_to_a_whole_tick():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (100, 101, 90, 99)], tick_size=0.05,
    )
    result = LongOnlyBacktester(BacktestConfig(quantity=3)).run(frame)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(100 - 20 / 3)
    assert result.trades.iloc[0]["gross_profit"] == pytest.approx(-20)


def test_cash_target_smaller_than_one_tick_can_fill_at_its_fractional_price():
    frame = signals(
        [(99, 100, 98, 99), (100, 100.0005, 99, 100), (100, 100.002, 99, 100)],
        take_profit_cash=0.001, tick_size=0.05,
    )
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    assert result.trades.iloc[0]["exit_price"] == pytest.approx(100.001)
    assert result.trades.iloc[0]["gross_profit"] == pytest.approx(0.001)
    assert result.trades.iloc[0]["exit_reason"] == "TAKE_PROFIT"


def test_equity_reports_fixed_levels_and_actual_bracket_activation():
    frame = signals([
        (99, 100, 98, 99), (100, 101, 99, 100),
        (100, 110, 90, 100), (100, 205, 95, 105),
    ])
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    entry_bar = result.equity.loc[frame.index[1]]
    active_bar = result.equity.loc[frame.index[2]]
    assert entry_bar["stop_loss_price"] == active_bar["stop_loss_price"] == 80
    assert entry_bar["take_profit_price"] == active_bar["take_profit_price"] == 200
    assert not entry_bar["bracket_active"]
    assert active_bar["bracket_active"]
    assert not result.equity.loc[frame.index[3], "bracket_active"]


def test_point_value_scales_notional_cash_equity_fees_and_unrealized_profit():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (102, 103, 101, 103)], point_value=2,
    )
    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=1_000, quantity=2, commission_bps=100, close_at_end=False),
    ).run(frame)
    assert result.equity.loc[frame.index[1], "cash"] == pytest.approx(596)
    assert result.equity.loc[frame.index[1], "equity"] == pytest.approx(996)
    assert result.metrics["final_equity"] == pytest.approx(1_008)
    assert result.metrics["unrealized_profit"] == pytest.approx(8)
    assert result.metrics["total_fees"] == pytest.approx(4)
    assert result.metrics["open_positions"] == 1
    assert result.trades.empty


def test_point_value_notional_and_commission_must_fit_cash_before_entry():
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), (101, 102, 100, 101)], point_value=2)
    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=400, quantity=2, commission_bps=1),
    ).run(frame)
    assert result.orders.iloc[0]["status"] == "REJECTED"
    assert result.orders.iloc[0]["reason"] == "INSUFFICIENT_CASH"
    assert result.metrics["total_buys"] == 0
    assert result.metrics["final_equity"] == 400


def test_repeated_buys_do_not_pyramid_and_flat_exits_do_not_short_with_brackets():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (100, 110, 90, 100),
         (100, 205, 95, 105), (105, 106, 104, 105)],
        buys=(0, 1, 2), exits=(4,),
    )
    result = LongOnlyBacktester(BacktestConfig(quantity=1)).run(frame)
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 1
    assert result.equity["quantity"].between(0, 1).all()
    assert (result.orders["reason"] == "POSITION_OPEN").sum() == 2
    assert (result.orders["reason"] == "NO_POSITION").sum() == 1
    assert set(result.orders["side"]) <= {"BUY", "EXIT"}


def test_bracket_exit_can_be_followed_by_a_fresh_close_signal_for_next_open():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (100, 205, 95, 105),
         (105, 106, 104, 105), (105, 110, 100, 106)], buys=(0, 2),
    )
    result = LongOnlyBacktester(BacktestConfig()).run(frame)
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 2
    assert result.trades["entry_time"].tolist() == [frame.index[1], frame.index[3]]
    assert result.trades["exit_reason"].tolist() == ["TAKE_PROFIT", "END_OF_DATA"]


def test_appending_future_candles_does_not_change_earlier_protective_fills():
    frame = signals(
        [(99, 100, 98, 99), (100, 101, 99, 100), (100, 205, 95, 105),
         (105, 250, 50, 110)],
    )
    engine = LongOnlyBacktester(BacktestConfig(close_at_end=False))
    prefix = engine.run(frame.iloc[:3])
    complete = engine.run(frame)
    pd.testing.assert_frame_equal(prefix.trades, complete.trades)
    pd.testing.assert_frame_equal(prefix.equity, complete.equity.iloc[:3])


@pytest.mark.parametrize("fault,field", [
    ("missing", "tick_size"),
    ("varying", "stop_loss_cash"),
    ("invalid", "point_value"),
])
def test_incomplete_or_nonconstant_cash_policy_is_rejected(fault, field):
    frame = signals([(99, 100, 98, 99), (100, 101, 99, 100), (100, 101, 99, 100)])
    if fault == "missing":
        frame = frame.drop(columns=field)
    elif fault == "varying":
        frame.loc[frame.index[-1], field] = 21
    else:
        frame[field] = 0
    with pytest.raises(ValueError, match=field):
        LongOnlyBacktester().run(frame)
