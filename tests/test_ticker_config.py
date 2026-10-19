"""Configured ticker lists are shared by quiet scanning and backtesting."""

import json

import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import downloader, file_utils


# These cases verify explicit fixed-quantity execution, not the CLI default.
pytestmark = pytest.mark.usefixtures("fixed_position_registry")


@pytest.fixture
def configured_universe(tmp_path, monkeypatch):
    root = tmp_path / "project"
    data_dir = root / "data"
    config_dir = root / "config"
    data_dir.mkdir(parents=True)
    config_dir.mkdir()
    for data_type, ticker, interval in (
        ("day", "DAY_SYMBOL", "1d"),
        ("intraday", "INTRADAY_SYMBOL", "1h"),
    ):
        ticker_path = data_dir / f"{data_type}_symbols.csv"
        ticker_path.write_text(f"ticker\n{ticker}\n", encoding="utf-8")
        config = {
            "ticker_file": f" data/{ticker_path.name} ",
            "download": {"period": "7y", "interval": interval, "is_download": False},
            "rsi_bollinger": {
                "rsi_period": 6, "bb_period": 200,
                "bb_multiplier": 2.0, "rsi_threshold": 50.0,
            },
            "supertrend": {
                "atr_period": 10, "atr_multiplier": 3.0,
                "source": "hl2", "change_atr": True,
                "from_date": "2018-09-01", "to_date": "9999-01-01",
                "timezone": "Asia/Kolkata",
            },
        }
        (config_dir / f"{data_type}.json").write_text(json.dumps(config), encoding="utf-8")
        cache = root / "output" / data_type
        cache.mkdir(parents=True)
        closes = [100.0 + position % 7 for position in range(220)]
        frame = pd.DataFrame(
            {
                "open": closes, "high": [close + 1 for close in closes],
                "low": [close - 1 for close in closes], "close": closes,
            },
            index=pd.date_range("2024-01-01", periods=len(closes), name="date"),
        )
        frame.to_csv(cache / f"{ticker}.csv")
        frame.to_csv(cache / "OVERRIDE.csv")
    (data_dir / "override.csv").write_text("symbol\nOVERRIDE\n", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        pytest.fail("Ticker configuration tests must use the prepared offline cache")

    monkeypatch.setattr(cli, "ROOT_DIR", root)
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(downloader, "download_prices", unexpected_download)
    return root, config_dir


@pytest.mark.parametrize("strategy_name", ["rsi_bollinger", "supertrend"])
@pytest.mark.parametrize("mode", ["scan", "backtest"])
@pytest.mark.parametrize("data_type,ticker", [("day", "DAY_SYMBOL"), ("intraday", "INTRADAY_SYMBOL")])
def test_each_strategy_and_mode_uses_selected_config_tickers(
    tmp_path, configured_universe, capsys, strategy_name, mode, data_type, ticker,
):
    root, _ = configured_universe
    output = tmp_path / "reports"
    assert cli.main(
        ["--data-type", data_type, "--output-dir", str(output)],
        mode=mode, strategy_name=strategy_name,
    ) == 0
    summary = pd.read_csv(output / cli.SUMMARY_FILENAMES[strategy_name][mode])
    assert summary["ticker"].tolist() == [ticker]
    if mode == "backtest":
        metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
        assert metadata["ticker_file"] == str(root / "data" / f"{data_type}_symbols.csv")
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("configured_value", [None, "", "   ", 42, False, [], {}])
def test_invalid_default_fails_before_loading_tickers_or_writing_reports(
    tmp_path, configured_universe, capsys, monkeypatch, configured_value,
):
    _, config_dir = configured_universe
    config_path = config_dir / "day.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    settings["ticker_file"] = configured_value
    settings["download"]["is_download"] = True
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    output = tmp_path / "reports"

    def unexpected_ticker_read(*args, **kwargs):
        pytest.fail("Invalid ticker configuration must fail before reading ticker files")

    monkeypatch.setattr(file_utils, "read_tickers", unexpected_ticker_read)
    with pytest.raises(SystemExit) as error:
        cli.main(["--output-dir", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert str(config_path) in captured.err
    assert "ticker_file" in captured.err
    assert not output.exists()


def test_missing_default_reports_the_selected_config_path(tmp_path, configured_universe, capsys):
    _, config_dir = configured_universe
    config_path = config_dir / "intraday.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    del settings["ticker_file"]
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        cli.main(["--data-type", "intraday", "--output-dir", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert str(config_path) in captured.err
    assert "ticker_file" in captured.err
    assert not output.exists()


@pytest.mark.parametrize("default_settings", [{}, {"ticker_file": None}, {"ticker_file": 42}])
@pytest.mark.parametrize("override_form", ["bare", "relative", "absolute"])
def test_explicit_override_works_without_a_valid_unused_default(
    tmp_path, configured_universe, capsys, default_settings, override_form,
):
    root, config_dir = configured_universe
    config_path = config_dir / "day.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    del settings["ticker_file"]
    settings.update(default_settings)
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    ticker_path = root / "data" / "override.csv"
    override = {
        "bare": "override.csv", "relative": "data/override.csv", "absolute": str(ticker_path),
    }[override_form]
    output = tmp_path / "reports"
    assert cli.main(
        ["--ticker-file", override, "--output-dir", str(output)], mode="backtest",
    ) == 0
    summary = pd.read_csv(output / "rsi_bolinger_backtest.csv")
    assert summary["ticker"].tolist() == ["OVERRIDE"]
    metadata = json.loads((output / "config.json").read_text(encoding="utf-8"))
    assert metadata["ticker_file"] == str(ticker_path)
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_missing_configured_ticker_file_reports_its_resolved_path(
    tmp_path, configured_universe, capsys,
):
    root, config_dir = configured_universe
    config_path = config_dir / "day.json"
    settings = json.loads(config_path.read_text(encoding="utf-8"))
    settings["ticker_file"] = "missing.csv"
    config_path.write_text(json.dumps(settings), encoding="utf-8")
    output = tmp_path / "reports"
    with pytest.raises(SystemExit) as error:
        cli.main(["--output-dir", str(output)])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert str(root / "data" / "missing.csv") in captured.err
    assert not output.exists()
