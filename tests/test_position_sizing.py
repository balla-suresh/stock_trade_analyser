"""Capital-based whole-share sizing and its CLI/configuration precedence."""

import json

import numpy as np
import pandas as pd
import pytest

from stock_trade_analyser.backtest.engine import BacktestConfig, LongOnlyBacktester
from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import downloader


def signals(opens, *, closes=None, buys=(0,), exits=(), cash_policy=False):
    opening = np.asarray(opens, dtype=float)
    closing = opening.copy() if closes is None else np.asarray(closes, dtype=float)
    frame = pd.DataFrame(
        {
            "open": opening, "high": np.maximum(opening, closing) + 0.001,
            "low": np.minimum(opening, closing) - 0.001, "close": closing,
            "buy_signal": False, "exit_signal": False, "entry_stop": np.nan,
        },
        index=pd.date_range("2025-01-01", periods=len(opening), name="date"),
    )
    for offset in buys:
        frame.loc[frame.index[offset], "buy_signal"] = True
    for offset in exits:
        frame.loc[frame.index[offset], "exit_signal"] = True
    if cash_policy:
        for name, value in {"take_profit_cash": 100.0, "stop_loss_cash": 20.0,
                            "point_value": 1.0, "tick_size": 0.01}.items():
            frame[name] = value
    return frame


@pytest.mark.parametrize("first_exit,expected_quantities,final_equity", [
    (120, [10, 12], 1_320),
    (80, [10, 8], 880),
])
def test_each_entry_sizes_from_current_cash_after_prior_profit_or_loss(
    first_exit, expected_quantities, final_equity,
):
    frame = signals([100, 100, 110, first_exit, 100, 100, 105, 110], buys=(0, 4), exits=(2, 6))
    result = LongOnlyBacktester(BacktestConfig(initial_capital=1_000, position_sizing="capital")).run(frame)
    assert result.trades["quantity"].tolist() == expected_quantities
    assert result.metrics["final_equity"] == final_equity
    filled = result.orders.loc[result.orders["status"] == "FILLED"]
    assert filled.loc[filled["side"] == "BUY", "quantity"].tolist() == expected_quantities
    assert filled.loc[filled["side"] == "EXIT", "quantity"].tolist() == expected_quantities
    assert (result.equity["cash"] >= 0).all()


def test_entry_quantity_uses_actual_next_open_instead_of_signal_close():
    frame = signals([50, 200, 210])
    result = LongOnlyBacktester(BacktestConfig(initial_capital=1_000, position_sizing="capital")).run(frame)
    trade = result.trades.iloc[0]
    assert trade["entry_price"] == 200
    assert trade["quantity"] == 5
    assert trade["entry_time"] == frame.index[1]
    assert result.metrics["net_profit"] == 50


def test_allocation_limits_total_entry_cost_including_commission_and_slippage():
    frame = signals([99, 100, 105])
    config = BacktestConfig(
        initial_capital=1_000, position_sizing="capital", capital_allocation_pct=50,
        slippage_bps=100, commission_bps=100, close_at_end=False,
    )
    result = LongOnlyBacktester(config).run(frame)
    # Each share costs 101 plus 1.01 fee. Four shares fit the 500 budget.
    assert result.equity.iloc[1]["quantity"] == 4
    assert result.equity.iloc[1]["cash"] == pytest.approx(591.96)
    assert result.metrics["total_fees"] == pytest.approx(4.04)
    assert result.metrics["final_equity"] == pytest.approx(1_011.96)
    assert (result.equity["cash"] >= 0).all()


def test_point_value_is_included_in_capital_sizing_and_position_valuation():
    frame = signals([99, 100, 101], cash_policy=True)
    frame["point_value"] = 2.0
    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=1_000, position_sizing="capital", close_at_end=False),
    ).run(frame)
    assert result.equity.iloc[1]["quantity"] == 5
    assert result.equity.iloc[1]["cash"] == 0
    assert result.metrics["final_equity"] == 1_010
    assert result.metrics["unrealized_profit"] == 10
    assert result.metrics["open_positions"] == 1


@pytest.mark.parametrize("capital,expected_quantity", [(1_000, 10), (2_000, 20)])
def test_cash_bracket_distance_uses_filled_capital_sized_quantity(capital, expected_quantity):
    frame = signals([99, 100, 100], cash_policy=True)
    frame.loc[frame.index[2], ["high", "low", "close"]] = [111, 99.5, 105]
    result = LongOnlyBacktester(BacktestConfig(initial_capital=capital, position_sizing="capital")).run(frame)
    trade = result.trades.iloc[0]
    assert trade["quantity"] == expected_quantity
    assert trade["exit_price"] == pytest.approx(100 + 100 / expected_quantity)
    assert trade["gross_profit"] == pytest.approx(100)
    assert trade["exit_reason"] == "TAKE_PROFIT"
    assert result.equity.iloc[1]["stop_loss_price"] == pytest.approx(100 - 20 / expected_quantity)
    assert result.metrics["final_equity"] == pytest.approx(capital + 100)


@pytest.mark.parametrize("capital,price,quantity", [
    (1_000, 100, 10),
    (299.99, 100, 2),
    (0.3, 0.1, 3),
])
def test_whole_share_floor_respects_exact_budget_and_float_rounding(capital, price, quantity):
    frame = signals([price, price, price])
    result = LongOnlyBacktester(
        BacktestConfig(initial_capital=capital, position_sizing="capital", close_at_end=False),
    ).run(frame)
    assert result.equity.iloc[1]["quantity"] == quantity
    assert (result.equity["cash"] >= 0).all()
    assert result.metrics["final_equity"] == pytest.approx(capital)


def test_unaffordable_capital_entry_is_rejected_with_zero_quantity():
    frame = signals([90, 100, 105])
    result = LongOnlyBacktester(BacktestConfig(initial_capital=99, position_sizing="capital")).run(frame)
    assert result.orders.iloc[0]["status"] == "REJECTED"
    assert result.orders.iloc[0]["reason"] == "INSUFFICIENT_CASH"
    assert result.orders.iloc[0]["quantity"] == 0
    assert result.metrics["total_buys"] == result.metrics["total_sells"] == 0
    assert result.metrics["final_equity"] == 99


def test_unfilled_capital_buy_keeps_unknown_quantity_until_a_future_open_exists():
    frame = signals([100, 101], buys=(1,))
    result = LongOnlyBacktester(BacktestConfig(initial_capital=1_000, position_sizing="capital")).run(frame)
    assert result.orders.iloc[0]["status"] == "NO_NEXT_BAR"
    assert pd.isna(result.orders.iloc[0]["quantity"])
    assert result.metrics["total_buys"] == 0


def test_direct_engine_api_preserves_explicit_fixed_quantity_behavior():
    assert BacktestConfig().position_sizing == "fixed"
    frame = signals([99, 100, 110])
    result = LongOnlyBacktester(BacktestConfig(initial_capital=1_000, quantity=2)).run(frame)
    assert result.trades.iloc[0]["quantity"] == 2
    assert result.metrics["final_equity"] == 1_020


@pytest.mark.parametrize("settings,field", [
    ({"position_sizing": "all"}, "position_sizing"),
    ({"position_sizing": None}, "position_sizing"),
    ({"capital_allocation_pct": 0}, "capital_allocation_pct"),
    ({"capital_allocation_pct": -1}, "capital_allocation_pct"),
    ({"capital_allocation_pct": 100.1}, "capital_allocation_pct"),
    ({"capital_allocation_pct": True}, "capital_allocation_pct"),
    ({"capital_allocation_pct": np.nan}, "capital_allocation_pct"),
    ({"capital_allocation_pct": np.inf}, "capital_allocation_pct"),
])
def test_invalid_sizing_modes_and_allocations_identify_the_setting(settings, field):
    with pytest.raises(ValueError, match=field):
        BacktestConfig(**settings)


@pytest.fixture
def capital_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config_dir = root / "config"
    cache_dir = root / "output" / "day"
    config_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / "tickers.csv").write_text("ticker\nTEST\n", encoding="utf-8")
    settings = {
        "ticker_file": "data/tickers.csv",
        "download": {"period": "5y", "interval": "1d", "is_download": False},
        "supertrend": {
            "atr_period": 3, "atr_multiplier": 1.0, "source": "hl2", "change_atr": True,
            "from_date": "2018-09-01", "to_date": "9999-01-01", "timezone": "Asia/Kolkata",
            "backtest": {"initial_capital": 1_000},
        },
    }
    config_path = config_dir / "day.json"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    closes = [100] * 5 + [80, 80, 120, 121, 122, 70, 69, 69]
    pd.DataFrame(
        {"open": closes, "high": [value + 1 for value in closes],
         "low": [value - 1 for value in closes], "close": closes, "volume": 1_000},
        index=pd.date_range("2024-01-01", periods=len(closes), name="date"),
    ).to_csv(cache_dir / "TEST.csv")

    def no_network(*args, **kwargs):
        pytest.fail("Capital sizing integration must consume its isolated cached prices")

    monkeypatch.setattr(cli, "ROOT_DIR", root)
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(downloader, "download_prices", no_network)
    return root, config_path


def run_capital_project(*arguments):
    return cli.main(["--strategy", "supertrend", *arguments], mode="backtest")


def test_real_cli_registry_defaults_to_full_available_capital(capital_project, capsys):
    root, _ = capital_project
    assert cli.REGISTRY.backtest_defaults["position_sizing"] == "capital"
    assert cli.REGISTRY.backtest_defaults["capital_allocation_pct"] == 100
    assert run_capital_project() == 0
    detail = root / "backtests" / "supertrend" / "day"
    trades = pd.read_csv(detail / "TEST" / "trades.csv")
    assert trades["quantity"].tolist() == [8]  # 1000 / the actual 121 entry fill.
    assert trades["entry_price"].tolist() == [121]
    assert trades["exit_price"].tolist() == [69]
    metadata = json.loads((detail / "config.json").read_text())
    assert metadata["backtest"]["position_sizing"] == "capital"
    assert metadata["backtest"]["capital_allocation_pct"] == 100
    row = pd.read_csv(root / "predictions" / "supertrend_backtest.csv").iloc[0]
    assert row["initial_capital"] == 1_000
    assert row["final_equity"] == 584
    assert row["net_profit"] == -416
    orders = pd.read_csv(detail / "TEST" / "orders.csv")
    assert orders.loc[orders["status"] == "FILLED", "quantity"].tolist() == [8, 8]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("configured,arguments,expected_quantity,expected_mode,expected_allocation", [
    ({"capital_allocation_pct": 25}, [], 2, "capital", 25),
    ({"capital_allocation_pct": 25}, ["--capital-allocation-pct", "50"], 4, "capital", 50),
    ({"position_sizing": "fixed", "quantity": 3}, [], 3, "fixed", 100),
    ({}, ["--quantity", "2"], 2, "fixed", 100),
    ({}, ["--position-sizing", "fixed", "--quantity", "3"], 3, "fixed", 100),
    ({"position_sizing": "fixed", "quantity": 3}, ["--position-sizing", "capital"], 8, "capital", 100),
])
def test_strategy_config_and_cli_sizing_overrides_take_precedence(
    capital_project, configured, arguments, expected_quantity, expected_mode, expected_allocation,
):
    root, path = capital_project
    settings = json.loads(path.read_text())
    settings["supertrend"]["backtest"].update(configured)
    path.write_text(json.dumps(settings), encoding="utf-8")
    assert run_capital_project(*arguments) == 0
    detail = root / "backtests" / "supertrend" / "day"
    assert pd.read_csv(detail / "TEST" / "trades.csv")["quantity"].tolist() == [expected_quantity]
    metadata = json.loads((detail / "config.json").read_text())["backtest"]
    assert metadata["position_sizing"] == expected_mode
    assert metadata["capital_allocation_pct"] == expected_allocation


@pytest.mark.parametrize("arguments,field", [
    (["--quantity", "2", "--position-sizing", "capital"], "--quantity"),
    (["--capital-allocation-pct", "0"], "capital_allocation_pct"),
    (["--capital-allocation-pct", "101"], "capital_allocation_pct"),
])
def test_conflicting_or_invalid_cli_sizing_fails_before_reports(capital_project, capsys, arguments, field):
    root, _ = capital_project
    with pytest.raises(SystemExit) as caught:
        run_capital_project(*arguments)
    assert caught.value.code == 2
    message = capsys.readouterr().err
    assert field in message
    assert "Traceback" not in message
    assert not (root / "predictions").exists()
    assert not (root / "backtests").exists()
