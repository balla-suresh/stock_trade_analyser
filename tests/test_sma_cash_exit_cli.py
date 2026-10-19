"""Offline integration for the registered SMA cash-exit scanner and backtest."""

import json
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import data_utils, downloader


# These cases verify explicit fixed-quantity execution, not the CLI default.
pytestmark = pytest.mark.usefixtures("fixed_position_registry")


ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "fast_period": 2, "slow_period": 3, "take_profit_cash": 100.0,
    "stop_loss_cash": 20.0, "point_value": 1.0, "tick_size": 0.01,
}


@pytest.fixture
def project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config = root / "config"
    config.mkdir(parents=True)
    (root / "data").mkdir()
    closes = [100, 100, 100, 100, 102, 104, 104, 100, 98, 98, 102, 104]
    frame = pd.DataFrame(
        {"open": closes, "high": [value + 1 for value in closes],
         "low": [value - 1 for value in closes], "close": closes, "volume": 1_000},
        index=pd.date_range("2025-01-01", periods=len(closes), name="date"),
    )
    for data_type, symbol, interval in (("day", "DAY", "1d"), ("intraday", "INTRA", "1h")):
        (root / "data" / f"{data_type}.csv").write_text(f"ticker\n{symbol}\n", encoding="utf-8")
        settings = {
            "ticker_file": f"data/{data_type}.csv",
            "download": {"period": "2y", "interval": interval, "is_download": False},
            "sma_cash_exit": SETTINGS.copy(),
        }
        (config / f"{data_type}.json").write_text(json.dumps(settings), encoding="utf-8")
        cache = root / "output" / data_type
        cache.mkdir(parents=True)
        frame.to_csv(cache / f"{symbol}.csv")

    def no_network(*args, **kwargs):
        pytest.fail("SMA cash-exit CLI tests must use cached prices or a batch mock")

    monkeypatch.setattr(cli, "ROOT_DIR", root)
    monkeypatch.setattr(cli, "CONFIG_DIR", config)
    monkeypatch.setattr(downloader, "download_prices", no_network)
    return root, frame


def run(*arguments, mode="scan"):
    return cli.main(["--strategy", "sma_cash_exit", *arguments], mode=mode)


@pytest.mark.parametrize("data_type,symbol", [("day", "DAY"), ("intraday", "INTRA")])
@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_configured_strategy_runs_quietly_and_writes_separate_reports(project, capsys, data_type, symbol, mode):
    root, _ = project
    assert run("--data-type", data_type, mode=mode) == 0
    filename = "sma_cash_exit_backtest.csv" if mode == "backtest" else "sma_cash_exit.csv"
    assert sorted(path.name for path in (root / "predictions").iterdir()) == [filename]
    report = pd.read_csv(root / "predictions" / filename)
    row = report.iloc[0]
    assert row["ticker"] == symbol
    if mode == "backtest":
        assert list(report.columns) == list(cli.REGISTRY.backtest_columns)
        assert row["total_buys"] == row["total_sells"] == row["total_trades"] == 2
        assert row["net_profit"] == -6
        assert row["initial_capital"] == 100_000
        assert row["final_equity"] == 99_994
        assert row["losing_trades"] == row["breakeven_trades"] == 1
        assert row["win_rate_pct"] == 0
        details = root / "backtests" / "sma_cash_exit" / data_type
        metadata = json.loads((details / "config.json").read_text())
        assert {name: metadata[name] for name in SETTINGS} == SETTINGS
        assert metadata["ticker_file"] == str(root / "data" / f"{data_type}.csv")
        assert "after the entry bar closes" in metadata["entry"]
        assert "following bar" in metadata["exit"]
        trades = pd.read_csv(details / symbol / "trades.csv")
        assert trades["entry_price"].tolist() == [104, 104]
        assert trades["exit_price"].tolist() == [98, 104]
        assert trades["exit_reason"].tolist() == ["EXIT_SIGNAL", "END_OF_DATA"]
        orders = pd.read_csv(details / symbol / "orders.csv")
        assert set(orders["side"]) <= {"BUY", "EXIT"}
        equity = pd.read_csv(details / symbol / "equity.csv")
        assert equity["quantity"].between(0, 1).all()
    else:
        assert list(report.columns) == [
            "ticker", "signal", "close", "fast_sma", "slow_sma", "buy_signals", "exit_signals", "message",
        ]
        assert row["signal"] == "HOLD"
        assert row["close"] == 104
        assert row["fast_sma"] == 103
        assert row["slow_sma"] == pytest.approx(304 / 3)
        assert row["buy_signals"] == 2
        assert row["exit_signals"] == 1
        assert not (root / "backtests").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("data_type", ["day", "intraday"])
def test_bundled_configs_construct_the_registered_strategy_with_configured_values(data_type):
    config = json.loads((ROOT / "src" / "stock_trade_analyser" / "config" / f"{data_type}.json").read_text())
    definition = cli.REGISTRY.strategies["sma_cash_exit"]
    parameters = {name: config["sma_cash_exit"][name] for name in definition.parameter_names}
    strategy = definition.create(parameters)
    assert {name: getattr(strategy, name) for name in parameters} == parameters


def test_all_indicator_and_instrument_settings_can_be_overridden(project, tmp_path):
    root, _ = project
    selected = {
        "fast_period": 1, "slow_period": 4, "take_profit_cash": 8.5,
        "stop_loss_cash": 3.25, "point_value": 2.0, "tick_size": 0.05,
    }
    output = tmp_path / "custom_reports"
    arguments = ["--output-dir", str(output), "--quantity", "2"]
    for name, value in selected.items():
        arguments.extend([f"--{name.replace('_', '-')}", str(value)])
    assert run(*arguments, mode="backtest") == 0
    metadata = json.loads((output / "config.json").read_text())
    assert {name: metadata[name] for name in selected} == selected
    assert metadata["backtest"]["quantity"] == 2
    assert (output / "sma_cash_exit_backtest.csv").is_file()
    assert not (root / "predictions").exists()
    generated = pd.read_csv(output / "DAY" / "signals.csv")
    assert generated["fast_sma"].first_valid_index() == 0
    assert generated["slow_sma"].first_valid_index() == 3
    for name in ("take_profit_cash", "stop_loss_cash", "point_value", "tick_size"):
        assert generated[name].eq(selected[name]).all()


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_downloads_all_symbols_once_and_uses_returned_prices_directly(project, monkeypatch, capsys, mode):
    root, frame = project
    config_path = root / "config" / "day.json"
    settings = json.loads(config_path.read_text())
    settings["download"]["is_download"] = True
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    (root / "data" / "day.csv").write_text("ticker\nDAY\nOTHER\nDAY\n", encoding="utf-8")
    combined = pd.concat({"DAY": frame, "OTHER": frame}, axis=1, names=["Ticker", "Price"])
    original = combined.copy(deep=True)
    calls = []

    def download(tickers, period, interval):
        calls.append((list(tickers), period, interval))
        return combined

    def no_reread(*args, **kwargs):
        pytest.fail("A downloaded price object must be consumed directly")

    monkeypatch.setattr(downloader, "download_prices", download)
    monkeypatch.setattr(data_utils, "load_prices", no_reread)
    assert run(mode=mode) == 0
    assert calls == [(["DAY", "OTHER"], "2y", "1d")]
    filename = "sma_cash_exit_backtest.csv" if mode == "backtest" else "sma_cash_exit.csv"
    report = pd.read_csv(root / "predictions" / filename)
    assert report["ticker"].tolist() == ["DAY", "OTHER"]
    pd.testing.assert_frame_equal(combined, original)
    assert (root / "output" / "day" / "OTHER.csv").is_file()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_sma_reports_preserve_other_strategies_and_scan_summary(project):
    root, _ = project
    predictions = root / "predictions"
    predictions.mkdir()
    preserved = {name: "existing result\n" for name in (
        "rsi_bolinger.csv", "rsi_bolinger_backtest.csv", "supertrend.csv",
        "supertrend_backtest.csv", "wyckoff.csv", "wyckoff_backtest.csv",
    )}
    for name, content in preserved.items():
        (predictions / name).write_text(content, encoding="utf-8")
    assert run() == 0
    scan_before = (predictions / "sma_cash_exit.csv").read_bytes()
    assert run(mode="backtest") == 0
    assert (predictions / "sma_cash_exit.csv").read_bytes() == scan_before
    for name, content in preserved.items():
        assert (predictions / name).read_text() == content


@pytest.mark.parametrize("fault,field", [
    ("missing", "point_value"), ("periods", "fast_period"), ("cash", "stop_loss_cash"),
])
def test_invalid_strategy_config_fails_before_network_or_reports(project, capsys, fault, field):
    root, _ = project
    path = root / "config" / "day.json"
    settings = json.loads(path.read_text())
    if fault == "missing":
        del settings["sma_cash_exit"][field]
    elif fault == "periods":
        settings["sma_cash_exit"][field] = 3
    else:
        settings["sma_cash_exit"][field] = 0
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        run(mode="backtest")
    assert caught.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert field in message
    assert "Traceback" not in message
    assert not (root / "predictions").exists()
    assert not (root / "backtests").exists()


@pytest.mark.parametrize("module", ["strategy", "backtest"])
def test_help_is_available_without_third_party_dependencies(module):
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{module}",
         "--strategy", "sma_cash_exit", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--fast-period", "--slow-period", "--take-profit-cash", "--stop-loss-cash", "--point-value", "--tick-size"):
        assert flag in result.stdout
    assert "--atr-period" not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("prefix,module", [("run", "strategy"), ("backtest", "backtest")])
def test_make_targets_explicitly_select_sma_cash_exit(prefix, module):
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", f"{prefix}-sma-cash-exit", "PYTHON=/test/python"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout) == [
        "/test/python", "-m", f"src.stock_trade_analyser.{module}", "--strategy", "sma_cash_exit",
    ]
