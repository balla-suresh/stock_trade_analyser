"""Offline scanner/backtest contracts for the Kingshuk Ghosh Wyckoff rules."""

import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import data_utils, downloader
from test_wyckoff import SETTINGS, spring_and_upthrust


# These cases verify explicit fixed-quantity execution, not the CLI default.
pytestmark = pytest.mark.usefixtures("fixed_position_registry")


ROOT = Path(__file__).resolve().parents[1]
SCAN_COLUMNS = [
    "ticker", "signal", "close", "phase", "pattern", "events", "support",
    "resistance", "buy_signals", "exit_signals", "message",
]
BACKTEST_COLUMNS = [
    "ticker", "initial_capital", "final_equity", "total_trades", "total_buys",
    "total_sells", "winning_trades", "losing_trades", "breakeven_trades",
    "net_profit", "total_return_pct", "win_rate_pct", "max_drawdown_pct", "message",
]


@pytest.fixture
def configured_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config = root / "config"
    config.mkdir(parents=True)
    (root / "data").mkdir()
    for data_type, symbol, interval in (("day", "DAY", "1d"), ("intraday", "INTRA", "1h")):
        (root / "data" / f"{data_type}.csv").write_text(f"symbol\n{symbol}\n")
        settings = {
            "ticker_file": f"data/{data_type}.csv",
            "download": {"period": "5y", "interval": interval, "is_download": False},
            "wyckoff": SETTINGS.copy(),
        }
        (config / f"{data_type}.json").write_text(json.dumps(settings))
        cache = root / "output" / data_type
        cache.mkdir(parents=True)
        spring_and_upthrust().to_csv(cache / f"{symbol}.csv")
        spring_and_upthrust().to_csv(cache / "OVERRIDE.csv")
    (root / "data" / "override.csv").write_text("ticker\nOVERRIDE\n")

    def unexpected_download(*args, **kwargs):
        pytest.fail("Wyckoff integration tests must not access the live network")

    monkeypatch.setattr(cli, "ROOT_DIR", root)
    monkeypatch.setattr(cli, "CONFIG_DIR", config)
    monkeypatch.setattr(downloader, "download_prices", unexpected_download)
    return root


def run(argv=(), *, mode="scan"):
    return cli.main(list(argv), mode=mode, strategy_name="wyckoff")


@pytest.mark.parametrize("data_type,symbol,interval", [("day", "DAY", "1d"), ("intraday", "INTRA", "1h")])
@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_selected_config_supplies_tickers_parameters_and_quiet_separate_reports(
    configured_project, capsys, data_type, symbol, interval, mode,
):
    root = configured_project
    assert run(["--data-type", data_type], mode=mode) == 0
    filename = "wyckoff_backtest.csv" if mode == "backtest" else "wyckoff.csv"
    assert sorted(path.name for path in (root / "predictions").iterdir()) == [filename]
    summary = pd.read_csv(root / "predictions" / filename)
    assert list(summary.columns) == (BACKTEST_COLUMNS if mode == "backtest" else SCAN_COLUMNS)
    row = summary.iloc[0]
    assert row["ticker"] == symbol
    if mode == "backtest":
        assert row["initial_capital"] == 100_000
        assert row["final_equity"] == 99_996
        assert row["total_buys"] == row["total_sells"] == row["total_trades"] == 1
        assert row["net_profit"] == -4
        assert row["win_rate_pct"] == 0
        assert row["max_drawdown_pct"] == pytest.approx(5/100_001*100)
        details = root / "backtests" / "wyckoff" / data_type
        metadata = json.loads((details / "config.json").read_text())
        assert {name: metadata[name] for name in SETTINGS} == SETTINGS
        assert metadata["ticker_file"] == str(root / "data" / f"{data_type}.csv")
        assert metadata["interval"] == interval
        trades = pd.read_csv(details / symbol / "trades.csv")
        assert trades.iloc[0]["entry_price"] == 103
        assert trades.iloc[0]["exit_price"] == 99
        orders = pd.read_csv(details / symbol / "orders.csv")
        assert orders.loc[orders["status"] == "FILLED", "side"].tolist() == ["BUY", "EXIT"]
        assert "POSITION_OPEN" in orders["reason"].tolist()
        equity = pd.read_csv(details / symbol / "equity.csv")
        assert equity["quantity"].between(0, 1).all()
    else:
        assert row["signal"] == "HOLD"
        assert row["close"] == 100
        assert row["support"] == 95
        assert row["resistance"] == 110
        assert row["buy_signals"] == 2
        assert row["exit_signals"] == 1
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_cli_overrides_config_tickers_and_event_mapping(configured_project, tmp_path):
    output = tmp_path / "custom"
    assert run([
        "--ticker-file", "override.csv", "--output-dir", str(output),
        "--lookback", "20", "--volume-multiplier", "2",
        "--buy-events", "SPRING", "--exit-events", "UTAD",
    ], mode="backtest") == 0
    metadata = json.loads((output / "config.json").read_text())
    assert metadata["lookback"] == 20
    assert metadata["volume_multiplier"] == 2
    assert metadata["buy_events"] == ["SPRING"]
    assert metadata["exit_events"] == ["UTAD"]
    assert pd.read_csv(output / "wyckoff_backtest.csv")["ticker"].tolist() == ["OVERRIDE"]


def test_one_batch_download_used_directly_without_cached_csv_reread(configured_project, monkeypatch):
    root = configured_project
    settings_path = root / "config" / "day.json"
    settings = json.loads(settings_path.read_text())
    settings["download"]["is_download"] = True
    settings_path.write_text(json.dumps(settings))
    (root / "data" / "day.csv").write_text("symbol\nDAY\nOTHER\nDAY\n")
    frame = spring_and_upthrust().rename(columns=str.title)
    combined = pd.concat({"DAY": frame, "OTHER": frame}, axis=1, names=["Ticker", "Price"])
    original = combined.copy(deep=True)
    calls = []

    def download(tickers, period, interval):
        calls.append((list(tickers), period, interval))
        return combined

    def no_reread(*args, **kwargs):
        pytest.fail("Downloaded price data must be consumed in memory")

    monkeypatch.setattr(downloader, "download_prices", download)
    monkeypatch.setattr(data_utils, "load_prices", no_reread)
    assert run(mode="backtest") == 0
    assert calls == [(["DAY", "OTHER"], "5y", "1d")]
    summary = pd.read_csv(root / "predictions" / "wyckoff_backtest.csv")
    assert summary["ticker"].tolist() == ["DAY", "OTHER"]
    assert summary["total_trades"].tolist() == [1, 1]
    assert (root / "output" / "day" / "OTHER.csv").is_file()
    pd.testing.assert_frame_equal(combined, original)


def test_wyckoff_summary_files_coexist_with_existing_strategy_outputs(configured_project):
    predictions = configured_project / "predictions"
    predictions.mkdir()
    preserved = {
        "rsi_bolinger.csv": "existing RSI scan\n",
        "rsi_bolinger_backtest.csv": "existing RSI backtest\n",
        "supertrend.csv": "existing SuperTrend scan\n",
        "supertrend_backtest.csv": "existing SuperTrend backtest\n",
    }
    for filename, content in preserved.items():
        (predictions / filename).write_text(content)
    assert run() == 0
    assert run(mode="backtest") == 0
    assert sorted(path.name for path in predictions.iterdir()) == sorted([
        *preserved, "wyckoff.csv", "wyckoff_backtest.csv",
    ])
    for filename, content in preserved.items():
        assert (predictions / filename).read_text() == content


@pytest.mark.parametrize("fault", ["missing_data", "missing_volume", "negative_volume"])
def test_bad_data_creates_explanatory_error_row_without_fabricated_performance(configured_project, capsys, fault):
    root = configured_project
    path = root / "output" / "day" / "DAY.csv"
    if fault == "missing_data":
        path.unlink()
    else:
        frame = spring_and_upthrust()
        if fault == "missing_volume":
            frame = frame.drop(columns="volume")
        else:
            frame.iloc[0, frame.columns.get_loc("volume")] = -1
        frame.to_csv(path)
    assert run(mode="backtest") == 1
    row = pd.read_csv(root / "predictions" / "wyckoff_backtest.csv").iloc[0]
    assert row["ticker"] == "DAY"
    assert row["initial_capital"] == 100_000
    for field in ("final_equity", "total_trades", "total_buys", "total_sells", "net_profit", "win_rate_pct"):
        assert pd.isna(row[field])
    assert isinstance(row["message"], str) and row["message"]
    if fault == "missing_data":
        assert str(path) in row["message"]
    else:
        assert "volume" in row["message"].lower()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_zero_volume_is_explained_in_report_without_fake_events(configured_project, mode):
    frame = spring_and_upthrust()
    frame["volume"] = 0
    frame.to_csv(configured_project / "output" / "day" / "DAY.csv")
    assert run(mode=mode) == 0
    filename = "wyckoff_backtest.csv" if mode == "backtest" else "wyckoff.csv"
    row = pd.read_csv(configured_project / "predictions" / filename).iloc[0]
    assert "volume" in row["message"].lower() and "zero" in row["message"].lower()
    if mode == "backtest":
        assert row["total_trades"] == row["total_buys"] == row["total_sells"] == 0
        assert row["final_equity"] == row["initial_capital"] == 100_000
    else:
        assert row["signal"] == "HOLD"
        assert row["buy_signals"] == row["exit_signals"] == 0


@pytest.mark.parametrize("fault", ["section", "parameter", "invalid"])
def test_invalid_selected_config_fails_before_downloading_or_writing(configured_project, capsys, fault):
    path = configured_project / "config" / "day.json"
    settings = json.loads(path.read_text())
    if fault == "section":
        del settings["wyckoff"]
    elif fault == "parameter":
        del settings["wyckoff"]["sos_volume_multiplier"]
    else:
        settings["wyckoff"]["buy_events"] = ["UNKNOWN"]
    path.write_text(json.dumps(settings))
    with pytest.raises(SystemExit) as error:
        run()
    assert error.value.code == 2
    assert str(path) in capsys.readouterr().err
    assert not (configured_project / "predictions").exists()


@pytest.mark.parametrize("module", ["strategy", "backtest"])
def test_wyckoff_help_loads_without_optional_dependencies(module):
    completed = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{module}",
         "--strategy", "wyckoff", "--help"],
        cwd=ROOT, text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0
    assert "Wyckoff" in completed.stdout
    assert "--buy-events" in completed.stdout
    assert completed.stderr == ""
