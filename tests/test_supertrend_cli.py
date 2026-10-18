"""Offline integration of the Pine SuperTrend rules, reports, and batch input."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli


ROOT = Path(__file__).resolve().parents[1]
SUPER_SETTINGS = {
    "atr_period": 3,
    "atr_multiplier": 1.0,
    "source": "hl2",
    "change_atr": True,
    "from_date": "2018-09-01",
    "to_date": "9999-01-01",
    "timezone": "Asia/Kolkata",
}
BACKTEST_COLUMNS = [
    "ticker", "initial_capital", "final_equity", "total_trades", "total_buys",
    "total_sells", "winning_trades", "losing_trades", "breakeven_trades",
    "net_profit", "total_return_pct", "win_rate_pct", "max_drawdown_pct", "message",
]


@pytest.fixture(autouse=True)
def prevent_live_downloads(monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("SuperTrend CLI tests must mock download_prices or disable downloading")

    monkeypatch.setattr(cli, "download_prices", unexpected_download)


@pytest.fixture
def run_config(tmp_path, monkeypatch):
    directory = tmp_path / "config"
    directory.mkdir()
    # No RSI section: a SuperTrend run must need only its selected strategy.
    for data_type, interval in (("day", "1d"), ("intraday", "1h")):
        settings = {
            "download": {"period": "5y", "interval": interval, "is_download": False},
            "supertrend": SUPER_SETTINGS.copy(),
        }
        (directory / f"{data_type}.json").write_text(json.dumps(settings))
    monkeypatch.setattr(cli, "CONFIG_DIR", directory)
    return directory


@pytest.fixture
def prices(tmp_path):
    directory = tmp_path / "prices"
    directory.mkdir()
    # The fall at index 5 establishes a bearish trend, the rise at index 7
    # buys at the next open of 121, and the fall at index 10 exits at 69.
    closes = [100.0] * 5 + [80.0, 80.0, 120.0, 121.0, 122.0, 70.0, 69.0, 69.0]
    frame = pd.DataFrame(
        {
            "Open": closes,
            "High": [value + 1 for value in closes],
            "Low": [value - 1 for value in closes],
            "Close": closes,
            "Volume": 1_000,
        },
        index=pd.date_range("2024-01-01", periods=len(closes), name="Date"),
    )
    frame.to_csv(directory / "TEST.csv")
    ticker_file = tmp_path / "tickers.csv"
    ticker_file.write_text("symbol\nTEST\n")
    return directory, ticker_file, frame


def arguments(prices, output_dir, *extra):
    data_dir, ticker_file, _ = prices
    return [
        "--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
        "--output-dir", str(output_dir), *extra,
    ]


def run_supertrend(argv, *, mode="scan"):
    return cli.main(argv, mode=mode, strategy_name="supertrend")


def test_scan_only_writes_final_supertrend_summary(tmp_path, run_config, prices, capsys):
    output = tmp_path / "scan"
    assert run_supertrend(arguments(prices, output)) == 0
    assert sorted(path.name for path in output.iterdir()) == ["supertrend.csv"]
    report = pd.read_csv(output / "supertrend.csv")
    assert list(report.columns) == [
        "ticker", "signal", "close", "trend", "supertrend", "atr",
        "buy_signals", "exit_signals", "message",
    ]
    row = report.iloc[0]
    assert row["ticker"] == "TEST"
    assert row["signal"] == "HOLD"
    assert row["close"] == 69
    assert row["trend"] == -1
    assert row["supertrend"] > row["close"]
    assert row["atr"] > 0
    # The first bearish flip is an exit event while flat, never a short.
    assert row["buy_signals"] == 1
    assert row["exit_signals"] == 2
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_backtest_fills_next_open_and_never_opens_a_short(tmp_path, run_config, prices, capsys):
    output = tmp_path / "backtest"
    assert run_supertrend(arguments(prices, output), mode="backtest") == 0
    report = pd.read_csv(output / "supertrend_backtest.csv")
    assert list(report.columns) == BACKTEST_COLUMNS
    row = report.iloc[0]
    assert row["initial_capital"] == 100_000
    assert row["final_equity"] == 99_948
    assert row["total_trades"] == row["total_buys"] == row["total_sells"] == 1
    assert row["winning_trades"] == row["breakeven_trades"] == 0
    assert row["losing_trades"] == 1
    assert row["net_profit"] == -52
    assert row["total_return_pct"] == pytest.approx(-0.052)
    assert row["win_rate_pct"] == 0
    assert row["max_drawdown_pct"] == pytest.approx(53 / 100_001 * 100)
    trades = pd.read_csv(output / "TEST" / "trades.csv")
    trade = trades.iloc[0]
    assert pd.Timestamp(trade["entry_signal_time"]) == pd.Timestamp("2024-01-08")
    assert pd.Timestamp(trade["entry_time"]) == pd.Timestamp("2024-01-09")
    assert pd.Timestamp(trade["exit_signal_time"]) == pd.Timestamp("2024-01-11")
    assert pd.Timestamp(trade["exit_time"]) == pd.Timestamp("2024-01-12")
    assert trade["entry_price"] == 121
    assert trade["exit_price"] == 69
    assert trade["exit_reason"] == "EXIT_SIGNAL"
    equity = pd.read_csv(output / "TEST" / "equity.csv")
    assert equity["quantity"].between(0, 1).all()
    orders = pd.read_csv(output / "TEST" / "orders.csv")
    assert set(orders["side"]) == {"BUY", "EXIT"}
    assert orders.iloc[0]["reason"] == "NO_POSITION"
    assert orders.iloc[0]["status"] == "SKIPPED"
    assert orders.loc[orders["status"] == "FILLED", "side"].tolist() == ["BUY", "EXIT"]
    for name in ("signals.csv", "metrics.json"):
        assert (output / "TEST" / name).is_file()
    metadata = json.loads((output / "config.json").read_text())
    assert {key: metadata[key] for key in SUPER_SETTINGS} == SUPER_SETTINGS
    assert metadata["backtest"]["quantity"] == 1
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_cli_overrides_selected_config_settings(tmp_path, run_config, prices):
    output = tmp_path / "overrides"
    overrides = {
        "atr_period": 2, "atr_multiplier": 1.5, "source": "close", "change_atr": False,
        "from_date": "2024-01-01", "to_date": "2024-01-12", "timezone": "UTC",
    }
    args = arguments(
        prices, output, "--atr-period", "2", "--atr-multiplier", "1.5",
        "--source", "close", "--no-change-atr", "--from-date", "2024-01-01",
        "--to-date", "2024-01-12", "--timezone", "UTC",
    )
    assert run_supertrend(args, mode="backtest") == 0
    metadata = json.loads((output / "config.json").read_text())
    assert {key: metadata[key] for key in overrides} == overrides


@pytest.mark.parametrize("change_atr", [True, False])
def test_selected_atr_method_changes_real_indicator_values(tmp_path, run_config, prices, change_atr):
    output = tmp_path / f"atr_{change_atr}"
    option = "--change-atr" if change_atr else "--no-change-atr"
    assert run_supertrend(arguments(prices, output, option), mode="backtest") == 0
    signals = pd.read_csv(output / "TEST" / "signals.csv")
    expected_atr = 481 / 27 if change_atr else 64 / 3
    assert signals.iloc[7]["atr"] == pytest.approx(expected_atr)
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["change_atr"] is change_atr


def test_start_date_does_not_enter_from_an_earlier_reversal(tmp_path, run_config, prices):
    output = tmp_path / "late_start"
    assert run_supertrend(
        arguments(prices, output, "--from-date", "2024-01-09"), mode="backtest",
    ) == 0
    summary = pd.read_csv(output / "supertrend_backtest.csv").iloc[0]
    assert summary["total_buys"] == summary["total_sells"] == summary["total_trades"] == 0
    assert summary["initial_capital"] == summary["final_equity"] == 100_000
    assert pd.isna(summary["win_rate_pct"])
    orders = pd.read_csv(output / "TEST" / "orders.csv")
    assert not (orders["status"] == "FILLED").any()


def test_end_date_stops_signal_exits_without_creating_a_short(tmp_path, run_config, prices):
    output = tmp_path / "end_window"
    assert run_supertrend(
        arguments(prices, output, "--to-date", "2024-01-10", "--keep-open"),
        mode="backtest",
    ) == 0
    summary = pd.read_csv(output / "supertrend_backtest.csv").iloc[0]
    assert summary["total_buys"] == 1
    assert summary["total_sells"] == summary["total_trades"] == 0
    equity = pd.read_csv(output / "TEST" / "equity.csv")
    assert equity.iloc[-1]["quantity"] == 1
    metrics = json.loads((output / "TEST" / "metrics.json").read_text())
    assert metrics["open_positions"] == 1


@pytest.mark.parametrize("data_type", ["day", "intraday"])
def test_configured_parameters_used_without_needing_valid_rsi_section(tmp_path, run_config, prices, data_type):
    path = run_config / f"{data_type}.json"
    settings = json.loads(path.read_text())
    settings["rsi_bollinger"] = {"rsi_period": "irrelevant invalid RSI setting"}
    settings["supertrend"]["atr_multiplier"] = 1.25
    path.write_text(json.dumps(settings))
    output = tmp_path / data_type
    assert run_supertrend(arguments(prices, output, "--data-type", data_type), mode="backtest") == 0
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["atr_multiplier"] == 1.25
    assert metadata["interval"] == ("1d" if data_type == "day" else "1h")
    assert metadata["to_date"] == "9999-01-01"


def test_all_tickers_downloaded_once_and_processed_in_memory(tmp_path, run_config, prices, monkeypatch):
    data_dir, ticker_file, frame = prices
    ticker_file.write_text("symbol\nTEST\nOTHER\nTEST\n")
    combined = pd.concat({"TEST": frame, "OTHER": frame}, axis=1, names=["Ticker", "Price"])
    original = combined.copy(deep=True)
    requests = []

    def batch_download(tickers, period, interval):
        requests.append((tickers.copy(), period, interval))
        return combined

    def no_disk_reread(*args, **kwargs):
        pytest.fail("Freshly downloaded data must be used directly in memory")

    monkeypatch.setattr(cli, "download_prices", batch_download)
    monkeypatch.setattr(cli, "load_prices", no_disk_reread)
    path = run_config / "day.json"
    settings = json.loads(path.read_text())
    settings["download"]["is_download"] = True
    path.write_text(json.dumps(settings))
    output = tmp_path / "batch"
    assert run_supertrend(arguments(prices, output), mode="backtest") == 0
    assert requests == [(["TEST", "OTHER"], "5y", "1d")]
    summary = pd.read_csv(output / "supertrend_backtest.csv")
    assert summary["ticker"].tolist() == ["TEST", "OTHER"]
    assert summary["total_trades"].tolist() == [1, 1]
    assert (data_dir / "OTHER.csv").is_file()
    pd.testing.assert_frame_equal(combined, original)


def test_default_reports_coexist_with_rsi_outputs(tmp_path, run_config, prices, monkeypatch):
    monkeypatch.setattr(cli, "ROOT_DIR", tmp_path)
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    existing = {
        "rsi_bolinger.csv": "existing RSI scanner output\n",
        "rsi_bolinger_backtest.csv": "existing RSI backtest output\n",
    }
    for name, content in existing.items():
        (predictions / name).write_text(content)
    data_dir, ticker_file, _ = prices
    args = ["--ticker-file", str(ticker_file), "--data-dir", str(data_dir)]
    assert run_supertrend(args) == 0
    assert run_supertrend(args, mode="backtest") == 0
    assert sorted(path.name for path in predictions.iterdir()) == sorted([
        *existing, "supertrend.csv", "supertrend_backtest.csv",
    ])
    for name, content in existing.items():
        assert (predictions / name).read_text() == content
    details = tmp_path / "backtests" / "supertrend" / "day"
    assert (details / "config.json").is_file()
    assert (details / "TEST" / "trades.csv").is_file()


@pytest.mark.parametrize("module", ["strategy", "backtest"])
def test_supertrend_module_help_needs_no_third_party_dependencies(module):
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{module}",
         "--strategy", "supertrend", "--help"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--atr-period" in result.stdout
    assert "--atr-multiplier" in result.stdout
    assert "--no-change-atr" in result.stdout
    assert "--from-date" in result.stdout


def test_strategy_flag_selects_supertrend_without_python_keyword(tmp_path, run_config, prices):
    output = tmp_path / "selected_by_flag"
    assert cli.main(arguments(prices, output, "--strategy", "supertrend"), mode="backtest") == 0
    report = pd.read_csv(output / "supertrend_backtest.csv")
    assert report.iloc[0]["total_trades"] == 1
    assert not (output / "rsi_bolinger_backtest.csv").exists()


@pytest.mark.parametrize("missing", ["atr_period", "from_date", "timezone"])
def test_missing_selected_config_value_is_reported_before_downloading(tmp_path, run_config, prices, missing, capsys):
    path = run_config / "day.json"
    settings = json.loads(path.read_text())
    del settings["supertrend"][missing]
    settings["download"]["is_download"] = True
    path.write_text(json.dumps(settings))
    with pytest.raises(SystemExit) as error:
        run_supertrend(arguments(prices, tmp_path / "invalid"))
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert f"missing supertrend.{missing}" in captured.err
    assert "Traceback" not in captured.err
