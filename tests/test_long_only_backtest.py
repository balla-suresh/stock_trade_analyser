"""Behavioral checks for the long-only execution and accounting model."""

from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.backtest.engine import (
    BacktestConfig,
    LongOnlyBacktester,
)


def market_data(opens, closes=None, *, buys=(), exits=(), entry_stop=95.0):
    """Build independent signals so these tests exercise execution, not RSI."""
    opens = np.asarray(opens, dtype=float)
    closes = opens.copy() if closes is None else np.asarray(closes, dtype=float)
    frame = pd.DataFrame(
        {
            "open": opens,
            "high": np.maximum(opens, closes) + 2,
            "low": np.minimum(opens, closes) - 2,
            "close": closes,
            "volume": 1_000.0,
            "buy_signal": False,
            "exit_signal": False,
            "entry_stop": np.nan,
        },
        index=pd.date_range("2025-01-01", periods=len(opens), freq="D"),
    )
    for offset in buys:
        frame.loc[frame.index[offset], ["buy_signal", "entry_stop"]] = [True, entry_stop]
    for offset in exits:
        frame.loc[frame.index[offset], "exit_signal"] = True
    return frame


def test_entry_stop_is_already_triggered_and_fills_next_open_even_on_gap_down():
    frame = market_data([100, 90, 105, 108], [100, 92, 106, 108], buys=[0], exits=[1])

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    trade = result.trades.iloc[0]
    assert trade["entry_signal_time"] == frame.index[0]
    assert trade["entry_time"] == frame.index[1]
    assert trade["entry_price"] == 90
    assert trade["exit_signal_time"] == frame.index[1]
    assert trade["exit_time"] == frame.index[2]
    assert trade["exit_price"] == 105
    assert trade["exit_reason"] == "EXIT_SIGNAL"
    assert trade["net_profit"] == 15
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 1
    assert result.equity.loc[frame.index[0], "quantity"] == 0
    assert result.equity.loc[frame.index[1], "quantity"] == 1


def test_costs_and_cash_equity_match_a_hand_calculated_round_trip():
    frame = market_data(
        [95, 100, 106, 110, 111],
        [96, 105, 107, 112, 111],
        buys=[0],
        exits=[2],
    )
    config = BacktestConfig(
        initial_capital=10_000,
        quantity=10,
        commission_bps=20,
        slippage_bps=10,
    )

    result = LongOnlyBacktester(config).run(frame)

    # Buy 10 at 100.10; sell 10 at 109.89; commissions are 0.2% on each fill.
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == pytest.approx(100.1)
    assert trade["exit_price"] == pytest.approx(109.89)
    assert trade["quantity"] == 10
    assert trade["gross_profit"] == pytest.approx(97.9)
    assert trade["fees"] == pytest.approx(4.1998)
    assert trade["net_profit"] == pytest.approx(93.7002)
    assert result.equity.loc[frame.index[1], "cash"] == pytest.approx(8_996.998)
    assert result.equity.loc[frame.index[1], "equity"] == pytest.approx(10_046.998)
    assert result.metrics["total_fees"] == pytest.approx(4.1998)
    assert result.metrics["closed_trade_profit"] == pytest.approx(93.7002)
    assert result.metrics["net_profit"] == pytest.approx(93.7002)
    assert result.metrics["total_return_pct"] == pytest.approx(0.937002)
    assert result.metrics["final_equity"] == pytest.approx(10_093.7002)
    assert result.metrics["win_rate_pct"] == 100
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 1
    assert result.metrics["winning_trades"] == result.metrics["total_trades"] == 1
    assert result.metrics["losing_trades"] == result.metrics["breakeven_trades"] == 0


def test_lower_bollinger_entry_stop_never_becomes_a_protective_stop():
    frame = market_data([99, 100, 70, 110], [100, 70, 90, 111], buys=[0], exits=[2])
    frame.loc[frame.index[1], "low"] = 50

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["exit_time"] == frame.index[3]
    assert result.trades.iloc[0]["exit_price"] == 110
    assert result.equity.loc[frame.index[2], "quantity"] == 1


def test_repeated_buys_do_not_pyramid_and_exits_never_open_shorts():
    frame = market_data(
        [100, 101, 102, 103, 104, 105, 106],
        buys=[0, 1, 2],
        exits=[3, 5],
    )

    result = LongOnlyBacktester(BacktestConfig(quantity=3)).run(frame)

    assert len(result.trades) == 1
    assert result.trades.iloc[0]["quantity"] == 3
    assert result.trades.iloc[0]["entry_time"] == frame.index[1]
    assert result.trades.iloc[0]["exit_time"] == frame.index[4]
    assert result.equity["quantity"].min() == 0
    assert result.equity["quantity"].max() == 3
    assert (
        (result.orders["status"] == "SKIPPED")
        & (result.orders["reason"] == "POSITION_OPEN")
    ).sum() == 2
    assert (
        (result.orders["status"] == "SKIPPED")
        & (result.orders["reason"] == "NO_POSITION")
    ).sum() == 1
    assert set(result.orders["side"]) <= {"BUY", "EXIT"}
    assert result.metrics["open_positions"] == 0
    assert result.metrics["buy_signals"] == 3
    assert result.metrics["exit_signals"] == 2
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 1


def test_exit_signal_while_flat_keeps_cash_unchanged():
    frame = market_data([100, 90, 110], exits=[0])

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    assert result.trades.empty
    assert result.metrics["net_profit"] == 0
    assert (result.equity["quantity"] == 0).all()
    assert result.orders.iloc[0]["side"] == "EXIT"
    assert result.orders.iloc[0]["status"] == "SKIPPED"
    assert result.orders.iloc[0]["reason"] == "NO_POSITION"
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 0


def test_last_bar_buy_cannot_fill_without_a_future_open():
    frame = market_data([100, 101, 102], buys=[2])

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    assert result.trades.empty
    assert result.metrics["net_profit"] == 0
    assert result.metrics["open_positions"] == 0
    assert result.orders.iloc[0]["status"] == "NO_NEXT_BAR"
    assert pd.isna(result.orders.iloc[0]["execution_time"])
    assert result.metrics["buy_signals"] == 1
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 0


def test_last_bar_exit_is_unfilled_but_final_valuation_can_liquidate():
    frame = market_data([100, 101, 103], [100, 102, 110], buys=[0], exits=[2])

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    trade = result.trades.iloc[0]
    assert trade["exit_time"] == frame.index[-1]
    assert trade["exit_price"] == 110
    assert trade["exit_reason"] == "END_OF_DATA"
    pending_exit = result.orders[
        (result.orders["signal_time"] == frame.index[-1])
        & (result.orders["side"] == "EXIT")
    ]
    assert (pending_exit["status"] == "NO_NEXT_BAR").any()
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 1


def test_final_liquidation_and_mark_to_market_are_reported_separately():
    frame = market_data([100, 100, 115], [100, 105, 120], buys=[0])
    base_config = {"initial_capital": 1_000, "quantity": 2, "commission_bps": 100}

    closed = LongOnlyBacktester(BacktestConfig(**base_config)).run(frame)
    held = LongOnlyBacktester(BacktestConfig(**base_config, close_at_end=False)).run(frame)

    assert closed.metrics["total_trades"] == 1
    assert closed.trades.iloc[0]["exit_reason"] == "END_OF_DATA"
    assert closed.metrics["net_profit"] == pytest.approx(35.6)
    assert closed.metrics["total_fees"] == pytest.approx(4.4)
    assert closed.metrics["open_positions"] == 0
    assert closed.metrics["total_buys"] == closed.metrics["total_sells"] == 1
    assert held.trades.empty
    assert held.metrics["total_trades"] == 0
    assert held.metrics["open_positions"] == 1
    assert held.metrics["net_profit"] == pytest.approx(38)
    assert held.metrics["final_equity"] == pytest.approx(1_038)
    assert held.metrics["total_fees"] == pytest.approx(2)
    assert held.metrics["closed_trade_profit"] == 0
    assert held.metrics["unrealized_profit"] == pytest.approx(38)
    assert held.equity.iloc[-1]["quantity"] == 2
    assert held.metrics["total_buys"] == 1
    assert held.metrics["total_sells"] == 0


def test_drawdown_includes_starting_capital_and_the_entry_fee():
    frame = market_data([100, 100], [100, 100], buys=[0])

    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=1_000, quantity=1, commission_bps=100)
    ).run(frame)

    assert result.metrics["net_profit"] == pytest.approx(-2)
    assert result.metrics["max_drawdown_pct"] == pytest.approx(0.2)
    assert result.equity.iloc[-1]["drawdown_pct"] == pytest.approx(0.2)


def test_insufficient_cash_accounts_for_fees_and_rejects_the_whole_order():
    frame = market_data([99, 100, 110], buys=[0])

    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=1_000, quantity=10, commission_bps=10)
    ).run(frame)

    assert result.trades.empty
    assert result.metrics["total_fees"] == 0
    assert result.metrics["final_equity"] == 1_000
    assert (result.equity["quantity"] == 0).all()
    assert result.orders.iloc[0]["status"] == "REJECTED"
    assert result.orders.iloc[0]["reason"] == "INSUFFICIENT_CASH"
    assert result.metrics["buy_signals"] == 1
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 0


def test_no_trades_have_undefined_win_rate_and_valid_empty_tables():
    frame = market_data([100, 90, 110])

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    assert result.metrics["total_trades"] == 0
    assert result.metrics["winning_trades"] == 0
    assert result.metrics["losing_trades"] == 0
    assert result.metrics["breakeven_trades"] == 0
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 0
    assert result.metrics["win_rate_pct"] is None
    assert result.metrics["total_return_pct"] == 0
    assert result.metrics["max_drawdown_pct"] == 0
    assert result.metrics["final_equity"] == 100_000
    assert result.trades.empty
    assert {"entry_time", "exit_time", "net_profit"} <= set(result.trades.columns)
    assert result.orders.empty
    assert {"signal_time", "execution_time", "status"} <= set(result.orders.columns)


def test_multiple_round_trips_count_wins_losses_and_breakevens_after_execution():
    frame = market_data(
        [100, 100, 110, 110, 110, 105, 105, 105, 105],
        buys=[0, 3, 6],
        exits=[1, 4, 7],
    )

    result = LongOnlyBacktester(BacktestConfig()).run(frame)

    assert result.trades["net_profit"].tolist() == [10, -5, 0]
    assert result.metrics["total_trades"] == 3
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 3
    assert result.metrics["winning_trades"] == 1
    assert result.metrics["losing_trades"] == 1
    assert result.metrics["breakeven_trades"] == 1
    assert result.metrics["win_rate_pct"] == pytest.approx(100 / 3)
    assert result.metrics["final_equity"] == 100_005
    assert result.metrics["net_profit"] == 5


def test_appending_future_bars_does_not_change_earlier_fills_or_equity():
    frame = market_data([100, 105, 110, 80, 200], [101, 106, 112, 90, 190], buys=[0])
    config = BacktestConfig(close_at_end=False)

    prefix = LongOnlyBacktester(config).run(frame.iloc[:3])
    extended = LongOnlyBacktester(config).run(frame)

    pd.testing.assert_frame_equal(prefix.equity, extended.equity.iloc[:3])
    first_fill = prefix.orders[prefix.orders["status"] == "FILLED"].iloc[0]
    extended_fill = extended.orders[extended.orders["status"] == "FILLED"].iloc[0]
    assert first_fill["execution_time"] == extended_fill["execution_time"]
    assert first_fill["price"] == extended_fill["price"] == 105


def test_impossible_simultaneous_buy_and_exit_signals_are_rejected():
    frame = market_data([100, 101, 102], buys=[0], exits=[0])

    with pytest.raises(ValueError, match="(?i)(simultaneous|both|buy.*exit)"):
        LongOnlyBacktester(BacktestConfig()).run(frame)


@pytest.mark.parametrize(
    "invalid",
    [
        {"initial_capital": 0},
        {"initial_capital": -1},
        {"initial_capital": float("nan")},
        {"initial_capital": float("inf")},
        {"quantity": 0},
        {"quantity": -1},
        {"quantity": 1.5},
        {"commission_bps": -1},
        {"commission_bps": float("nan")},
        {"slippage_bps": -1},
        {"slippage_bps": float("inf")},
    ],
)
def test_configuration_rejects_invalid_execution_parameters(invalid):
    with pytest.raises(ValueError):
        BacktestConfig(**invalid)


def test_configuration_is_immutable():
    config = BacktestConfig()
    with pytest.raises(FrozenInstanceError):
        config.quantity = 2
