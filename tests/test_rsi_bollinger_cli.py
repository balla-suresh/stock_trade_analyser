"""Offline CLI integration and data-quality reporting checks."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from stock_trade_analyser.strategy.cli import main
from stock_trade_analyser.tools.data_utils import load_prices
from stock_trade_analyser.tools.file_utils import read_tickers
from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import data_utils, downloader


# These cases verify explicit fixed-quantity execution, not the CLI default.
pytestmark = pytest.mark.usefixtures("fixed_position_registry")


ROOT = Path(__file__).resolve().parents[1]
DAY_CONFIG = json.loads(
    (ROOT / "src" / "stock_trade_analyser" / "config" / "day.json").read_text()
)
DAY_DOWNLOAD_CONFIG = DAY_CONFIG["download"]
DAY_STRATEGY_CONFIG = DAY_CONFIG["rsi_bollinger"]
DAY_PERIOD = DAY_DOWNLOAD_CONFIG["period"]
DAY_INTERVAL = DAY_DOWNLOAD_CONFIG["interval"]
BACKTEST_SUMMARY = "rsi_bolinger_backtest.csv"
SCAN_SUMMARY = "rsi_bolinger.csv"
BACKTEST_COLUMNS = [
    "ticker", "initial_capital", "final_equity", "total_trades", "total_buys",
    "total_sells", "winning_trades", "losing_trades", "breakeven_trades",
    "net_profit", "total_return_pct", "win_rate_pct", "max_drawdown_pct", "message",
]


def summary_filename(mode):
    return BACKTEST_SUMMARY if mode == "backtest" else SCAN_SUMMARY


@pytest.fixture(autouse=True)
def prevent_live_downloads(monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("CLI tests must use an explicit download mock or --no-download")

    monkeypatch.setattr(downloader, "download_prices", unexpected_download)


@pytest.fixture
def cached_prices(tmp_path):
    data_dir = tmp_path / "prices"
    data_dir.mkdir()
    # Stable history supplies BB200 warmup. An oversold excursion/recovery and
    # mirrored overbought excursion/drop satisfy both simultaneous crossings.
    close = [100.0] * 200 + [90.0, 85.0, 80.0, 100.0, 110.0, 115.0, 120.0, 100.0, 100.0]
    opening = [close[0], *close[:-1]]
    frame = pd.DataFrame(
        {"Open": opening, "High": [max(a, b) + 1 for a, b in zip(opening, close)],
         "Low": [min(a, b) - 1 for a, b in zip(opening, close)], "Close": close,
         "Volume": [1000] * len(close)},
        index=pd.date_range("2024-01-01", periods=len(close), name="Date"),
    )
    frame.to_csv(data_dir / "TEST.csv")
    ticker_file = tmp_path / "tickers.csv"
    ticker_file.write_text("\ufeffsymbol\nTEST\n\nTEST\n", encoding="utf-8")
    return data_dir, ticker_file


def test_offline_backtest_reports_signals_trades_and_configuration(tmp_path, cached_prices, capsys):
    data_dir, ticker_file = cached_prices
    report_dir = tmp_path / "backtest"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--no-download"], mode="backtest") == 0
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert len(summary) == 1
    assert list(summary.columns) == BACKTEST_COLUMNS
    row = summary.iloc[0]
    # Entry executes at 100 on the bar after the recovery; the original sell
    # condition closes at 100 on its following bar. The peak holding value is
    # 120, giving a 20 drawdown from the portfolio peak of 100,020.
    assert row["initial_capital"] == row["final_equity"] == 100_000
    assert row["total_trades"] == row["total_buys"] == row["total_sells"] == 1
    assert row["winning_trades"] == row["losing_trades"] == 0
    assert row["breakeven_trades"] == 1
    assert row["net_profit"] == row["total_return_pct"] == row["win_rate_pct"] == 0
    assert row["max_drawdown_pct"] == pytest.approx(20 / 100_020 * 100)
    assert pd.isna(row["message"])
    for filename in ("signals.csv", "trades.csv", "equity.csv", "orders.csv", "metrics.json"):
        assert (report_dir / "TEST" / filename).is_file()
    metrics = json.loads((report_dir / "TEST" / "metrics.json").read_text())
    assert metrics["total_trades"] == 1
    config = json.loads((report_dir / "config.json").read_text())
    assert {name: config[name] for name in DAY_STRATEGY_CONFIG} == DAY_STRATEGY_CONFIG
    assert config["backtest"]["quantity"] == 1
    assert config["backtest"]["close_at_end"] is True
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_scan_writes_only_final_summary_with_latest_state_and_signal_counts(tmp_path, cached_prices, capsys):
    data_dir, ticker_file = cached_prices
    output_dir = tmp_path / "scan"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(output_dir), "--no-download"]) == 0
    summary = pd.read_csv(output_dir / "rsi_bolinger.csv")
    assert list(summary.columns) == [
        "ticker", "signal", "close", "rsi", "bb_lower", "bb_upper",
        "buy_signals", "exit_signals", "message",
    ]
    assert summary.iloc[0]["signal"] == "HOLD"
    assert summary.iloc[0]["buy_signals"] == 1
    assert summary.iloc[0]["exit_signals"] == 1
    assert summary.iloc[0]["close"] == 100.0
    assert sorted(path.name for path in output_dir.iterdir()) == ["rsi_bolinger.csv"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_missing_and_invalid_data_are_reported_without_fake_zero_returns(tmp_path, cached_prices, capsys):
    data_dir, ticker_file = cached_prices
    ticker_file.write_text("symbol\nMISSING\nBROKEN\n")
    (data_dir / "BROKEN.csv").write_text("Date,Close\n2024-01-01,100\n")
    report_dir = tmp_path / "report"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--no-download"], mode="backtest") == 1
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert list(summary["initial_capital"]) == [100_000, 100_000]
    for column in BACKTEST_COLUMNS[2:-1]:
        assert summary[column].isna().all(), column
    assert "No cached CSV" in summary.iloc[0]["message"]
    assert "missing columns" in summary.iloc[1]["message"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_rerun_removes_only_requested_tickers_generated_reports(tmp_path, cached_prices, mode):
    data_dir, ticker_file = cached_prices
    report_dir = tmp_path / "backtest"
    arguments = ["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--no-download"]
    assert main(arguments, mode=mode) == 0
    target = report_dir / "TEST"
    target.mkdir(exist_ok=True)
    (target / "signals.csv").write_text("stale signal report")
    (target / "notes.txt").write_text("User notes")
    if mode == "scan":
        (report_dir / "config.json").write_text('{"legacy": true}')
        (target / "trades.csv").write_text("Existing backtest report")
    unrelated = report_dir / "OTHER"
    unrelated.mkdir()
    (unrelated / "metrics.json").write_text('{"total_trades": 9}')
    (data_dir / "TEST.csv").unlink()

    assert main(arguments, mode=mode) == 1
    expected_files = ["notes.txt", "trades.csv"] if mode == "scan" else ["notes.txt"]
    assert sorted(path.name for path in target.iterdir()) == expected_files
    if mode == "scan":
        assert not (report_dir / "config.json").exists()
        assert (target / "trades.csv").read_text() == "Existing backtest report"
    assert (unrelated / "metrics.json").read_text() == '{"total_trades": 9}'
    summary = pd.read_csv(report_dir / summary_filename(mode))
    if mode == "backtest":
        assert "No cached CSV" in summary.iloc[0]["message"]
        assert pd.isna(summary.iloc[0]["net_profit"])
    else:
        assert "status" not in summary.columns
        assert "date" not in summary.columns
        assert "No cached CSV" in summary.iloc[0]["message"]
        assert pd.isna(summary.iloc[0]["close"])


def test_short_history_is_explicit_and_win_rate_is_unavailable(tmp_path, cached_prices):
    data_dir, ticker_file = cached_prices
    frame = pd.read_csv(data_dir / "TEST.csv").iloc[:20]
    frame.to_csv(data_dir / "TEST.csv", index=False)
    report_dir = tmp_path / "short"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--no-download"], mode="backtest") == 0
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert "20 bars; at least" in summary.iloc[0]["message"]
    assert summary.iloc[0]["initial_capital"] == summary.iloc[0]["final_equity"] == 100_000
    assert summary.iloc[0]["total_trades"] == 0
    assert summary.iloc[0]["total_buys"] == summary.iloc[0]["total_sells"] == 0
    assert pd.isna(summary.iloc[0]["win_rate_pct"])


@pytest.mark.parametrize("mode", ["strategy", "backtest"])
def test_module_help_needs_no_third_party_dependencies(mode):
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{mode}", "--help"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--ticker-file" in result.stdout
    assert "--download" in result.stdout


@pytest.mark.parametrize("mode", ["strategy", "backtest"])
def test_module_missing_dependencies_explains_how_to_use_trade_environment(mode):
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{mode}",
         "--ticker-file", "tickers_50.csv"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 2
    assert "Missing runtime dependencies" in result.stderr
    assert "pandas" in result.stderr
    assert "workon trade" in result.stderr
    assert "Traceback" not in result.stderr


def test_load_prices_sorts_dates_and_rejects_invalid_rows(tmp_path):
    path = tmp_path / "prices.csv"
    path.write_text("Date,Open,High,Low,Close\n2024-01-02,2,3,1,2\n2024-01-01,1,2,0.5,1\n")
    frame = load_prices(path)
    assert frame.index.is_monotonic_increasing
    assert list(frame.columns) == ["open", "high", "low", "close"]
    path.write_text("Date,Open,High,Low,Close\nnot-a-date,1,2,0.5,1\n")
    with pytest.raises(ValueError, match="invalid dates"):
        load_prices(path)


def test_read_tickers_rejects_paths(tmp_path):
    path = tmp_path / "tickers.csv"
    path.write_text("../outside\n")
    with pytest.raises(ValueError, match="not a path"):
        read_tickers(path)


@pytest.mark.parametrize("header", ["Close,Close", "Close, close "])
def test_load_prices_rejects_duplicate_source_headers_before_pandas_mangles_them(tmp_path, header):
    path = tmp_path / "duplicate.csv"
    path.write_text(f"Date,Open,High,Low,{header}\n2025-01-01,1,2,0.5,1,99\n")
    with pytest.raises(ValueError, match="duplicate column names"):
        load_prices(path)


def test_load_prices_parses_numeric_calendar_dates_and_bom_header(tmp_path):
    path = tmp_path / "compact.csv"
    path.write_text("\ufeffDate,Open,High,Low,Close\n20250102,2,3,1,2\n20250101,1,2,0.5,1\n", encoding="utf-8")
    frame = load_prices(path)
    assert list(frame.index) == [pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-02")]


@pytest.mark.parametrize("timestamp", ["1735689600", "1735689600000", "20250101.0", "20250230"])
def test_load_prices_rejects_ambiguous_numeric_epochs_and_invalid_calendar_dates(tmp_path, timestamp):
    path = tmp_path / "bad-date.csv"
    path.write_text(f"Date,Open,High,Low,Close\n{timestamp},1,2,0.5,1\n")
    with pytest.raises(ValueError, match="dates"):
        load_prices(path)


def test_load_prices_preserves_saved_index_and_iso_datetime_support(tmp_path):
    path = tmp_path / "index.csv"
    path.write_text(
        ",Open,High,Low,Close\n2025-01-01T09:15:00+05:30,1,2,0.5,1\n"
        "2025-01-02T09:15:00+05:30,2,3,1,2\n"
    )
    frame = load_prices(path)
    assert list(frame.index) == [pd.Timestamp("2025-01-01 03:45:00"), pd.Timestamp("2025-01-02 03:45:00")]


def configure_download(monkeypatch, tmp_path, settings, *, data_type="day", strategy=None):
    config_dir = tmp_path / "config"
    config_dir.mkdir(exist_ok=True)
    path = config_dir / f"{data_type}.json"
    path.write_text(json.dumps({
        "download": settings,
        "rsi_bollinger": dict(DAY_STRATEGY_CONFIG) if strategy is None else strategy,
    }), encoding="utf-8")
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    return path


def mock_cached_download(monkeypatch, cached_prices):
    data_dir, _ = cached_prices
    frame = pd.read_csv(data_dir / "TEST.csv", index_col=0, parse_dates=True)
    calls = []

    def download(tickers, period, interval):
        calls.append((list(tickers), period, interval))
        return pd.concat({ticker: frame.copy() for ticker in tickers}, axis=1)

    monkeypatch.setattr(downloader, "download_prices", download)
    return calls


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_default_command_downloads_day_config_history_and_produces_results(
    tmp_path, cached_prices, monkeypatch, mode, capsys,
):
    _, ticker_file = cached_prices
    calls = mock_cached_download(monkeypatch, cached_prices)
    destination = tmp_path / "downloaded"
    report_dir = tmp_path / "report"
    assert not destination.exists()

    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(destination),
                 "--output-dir", str(report_dir)], mode=mode) == 0

    assert calls == [(["TEST"], DAY_PERIOD, DAY_INTERVAL)]
    assert (destination / "TEST.csv").is_file()
    summary = pd.read_csv(report_dir / summary_filename(mode))
    if mode == "backtest":
        assert summary.iloc[0]["total_trades"] == 1
        assert summary.iloc[0]["total_buys"] == summary.iloc[0]["total_sells"] == 1
        assert (report_dir / "TEST" / "trades.csv").is_file()
        config = json.loads((report_dir / "config.json").read_text())
        assert config["download"] is True
        assert config["period"] == DAY_PERIOD
        assert config["interval"] == DAY_INTERVAL
        assert Path(config["config_file"]) == ROOT / "src" / "stock_trade_analyser" / "config" / "day.json"
    else:
        assert "status" not in summary.columns
        assert "date" not in summary.columns
        assert summary.iloc[0]["signal"] == "HOLD"
        assert summary.iloc[0]["buy_signals"] == 1
        assert sorted(path.name for path in report_dir.iterdir()) == ["rsi_bolinger.csv"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_bare_ticker_file_command_downloads_and_reports_in_default_directories(
    tmp_path, cached_prices, monkeypatch, mode, capsys,
):
    _, ticker_file = cached_prices
    calls = mock_cached_download(monkeypatch, cached_prices)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "tickers_50.csv").write_bytes(ticker_file.read_bytes())
    monkeypatch.setattr(cli, "ROOT_DIR", tmp_path)

    assert main(["--ticker-file", "tickers_50.csv"], mode=mode) == 0

    assert calls == [(["TEST"], DAY_PERIOD, DAY_INTERVAL)]
    summary = pd.read_csv(tmp_path / "predictions" / summary_filename(mode))
    detail_root = "backtests" if mode == "backtest" else "predictions"
    detail_dir = tmp_path / detail_root / "rsi_bollinger" / "day"
    assert not (detail_dir / "summary.csv").exists()
    assert not (detail_dir / "rsi_bolinger.csv").exists()
    assert not (detail_dir / BACKTEST_SUMMARY).exists()
    if mode == "backtest":
        assert summary.iloc[0]["total_trades"] == 1
        assert summary.iloc[0]["total_buys"] == summary.iloc[0]["total_sells"] == 1
        assert (detail_dir / "TEST" / "signals.csv").is_file()
        assert (detail_dir / "TEST" / "trades.csv").is_file()
        assert (detail_dir / "config.json").is_file()
    else:
        assert "status" not in summary.columns
        assert "date" not in summary.columns
        assert summary.iloc[0]["signal"] == "HOLD"
        assert summary.iloc[0]["buy_signals"] == 1
        assert not detail_dir.exists()
        assert sorted(path.name for path in (tmp_path / "predictions").iterdir()) == ["rsi_bolinger.csv"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("custom_output", [False, True], ids=["default-folders", "shared-output-folder"])
def test_scan_and_backtest_preserve_each_others_reports(
    tmp_path, cached_prices, monkeypatch, custom_output,
):
    data_dir, ticker_file = cached_prices
    monkeypatch.setattr(cli, "ROOT_DIR", tmp_path)
    arguments = ["--ticker-file", str(ticker_file), "--data-dir", str(data_dir), "--no-download"]
    report_dir = tmp_path / "reports" if custom_output else tmp_path / "predictions"
    if custom_output:
        arguments.extend(["--output-dir", str(report_dir)])

    assert main(arguments, mode="scan") == 0
    scan_path = report_dir / SCAN_SUMMARY
    scan_before = scan_path.read_bytes()
    assert main([*arguments, "--initial-capital", "12345", "--quantity", "2"], mode="backtest") == 0

    backtest_path = report_dir / BACKTEST_SUMMARY
    backtest_before = backtest_path.read_bytes()
    assert scan_path.read_bytes() == scan_before
    backtest_row = pd.read_csv(backtest_path).iloc[0]
    assert backtest_row["initial_capital"] == backtest_row["final_equity"] == 12_345
    assert backtest_row["total_buys"] == backtest_row["total_sells"] == 1
    assert main(arguments, mode="scan") == 0
    assert backtest_path.read_bytes() == backtest_before
    assert scan_path.read_bytes() == scan_before


def test_download_defaults_follow_config_changes_and_refresh_cached_prices(tmp_path, cached_prices, monkeypatch):
    data_dir, ticker_file = cached_prices
    config_path = configure_download(monkeypatch, tmp_path, {
        "period": "3y", "interval": "1wk", "is_download": True,
    })
    calls = mock_cached_download(monkeypatch, cached_prices)
    report_dir = tmp_path / "report"

    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)], mode="backtest") == 0

    assert calls == [(["TEST"], "3y", "1wk")]
    config = json.loads((report_dir / "config.json").read_text())
    assert config["config_file"] == str(config_path)
    assert config["period"] == "3y"
    assert config["interval"] == "1wk"


@pytest.mark.parametrize("is_download", [False, "False", "false", "FALSE"])
def test_disabled_download_config_uses_cache_without_network(tmp_path, cached_prices, monkeypatch, is_download):
    data_dir, ticker_file = cached_prices
    configure_download(monkeypatch, tmp_path, {
        "period": "2y", "interval": "1d", "is_download": is_download,
    })
    report_dir = tmp_path / "report"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)], mode="backtest") == 0
    config = json.loads((report_dir / "config.json").read_text())
    assert config["download"] is False
    assert pd.read_csv(report_dir / BACKTEST_SUMMARY).iloc[0]["total_trades"] == 1


@pytest.mark.parametrize("is_download", [True, "True", "true", "TRUE"])
def test_enabled_download_config_supports_json_and_legacy_booleans(tmp_path, cached_prices, monkeypatch, is_download):
    data_dir, ticker_file = cached_prices
    configure_download(monkeypatch, tmp_path, {
        "period": "2y", "interval": "1d", "is_download": is_download,
    })
    calls = mock_cached_download(monkeypatch, cached_prices)
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(tmp_path / "report")], mode="backtest") == 0
    assert calls == [(["TEST"], "2y", "1d")]


def test_intraday_uses_its_own_config_defaults(tmp_path, cached_prices, monkeypatch):
    data_dir, ticker_file = cached_prices
    configure_download(monkeypatch, tmp_path, {
        "period": "7y", "interval": "1d", "is_download": False,
    })
    intraday_strategy = {
        "rsi_period": 2, "bb_period": 3, "bb_multiplier": 1.0, "rsi_threshold": 55.0,
    }
    config_path = configure_download(monkeypatch, tmp_path, {
        "period": "5d", "interval": "1m", "is_download": "True",
    }, data_type="intraday", strategy=intraday_strategy)
    calls = mock_cached_download(monkeypatch, cached_prices)
    report_dir = tmp_path / "report"

    assert main(["--ticker-file", str(ticker_file), "--data-type", "intraday",
                 "--data-dir", str(data_dir), "--output-dir", str(report_dir)], mode="backtest") == 0

    assert calls == [(["TEST"], "5d", "1m")]
    config = json.loads((report_dir / "config.json").read_text())
    assert config["config_file"] == str(config_path)
    assert config["data_type"] == "intraday"
    assert {name: config[name] for name in intraday_strategy} == intraday_strategy


def test_download_flags_and_history_flags_override_config(tmp_path, cached_prices, monkeypatch):
    data_dir, ticker_file = cached_prices
    configure_download(monkeypatch, tmp_path, {
        "period": "7y", "interval": "1d", "is_download": False,
    })
    calls = mock_cached_download(monkeypatch, cached_prices)
    report_dir = tmp_path / "report"

    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--download", "--period", "1y",
                 "--interval", "1wk"], mode="backtest") == 0

    assert calls == [(["TEST"], "1y", "1wk")]
    config = json.loads((report_dir / "config.json").read_text())
    assert config["download"] is True
    assert config["period"] == "1y"
    assert config["interval"] == "1wk"


def test_no_download_overrides_enabled_config_and_preserves_explicit_history_settings(tmp_path, cached_prices):
    data_dir, ticker_file = cached_prices
    report_dir = tmp_path / "report"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir), "--no-download", "--period", "1y",
                 "--interval", "1wk"], mode="backtest") == 0
    config = json.loads((report_dir / "config.json").read_text())
    assert config["download"] is False
    assert config["period"] == "1y"
    assert config["interval"] == "1wk"


def test_download_failure_reports_error_and_does_not_backtest_stale_cache(
    tmp_path, cached_prices, monkeypatch, capsys,
):
    data_dir, ticker_file = cached_prices
    report_dir = tmp_path / "report"

    def fail_download(*args):
        raise ValueError("Yahoo Finance returned no price data")

    monkeypatch.setattr(downloader, "download_prices", fail_download)
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)], mode="backtest") == 1
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert summary.iloc[0]["initial_capital"] == 100_000
    assert "no price data" in summary.iloc[0]["message"]
    assert pd.isna(summary.iloc[0]["total_trades"])
    assert pd.isna(summary.iloc[0]["total_return_pct"])
    assert not (report_dir / "TEST" / "trades.csv").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_full_ticker_list_downloads_once_and_uses_memory_instead_of_csv(
    tmp_path, cached_prices, monkeypatch, mode, capsys,
):
    _, ticker_file = cached_prices
    ticker_file.write_text("symbol\nFIRST\nSECOND\nFIRST\n")
    calls = mock_cached_download(monkeypatch, cached_prices)
    destination = tmp_path / "downloaded"
    report_dir = tmp_path / "report"

    def unexpected_cache_read(*args):
        pytest.fail("Downloaded prices must be passed directly to the strategy")

    monkeypatch.setattr(data_utils, "load_prices", unexpected_cache_read)
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(destination),
                 "--output-dir", str(report_dir)], mode=mode) == 0

    assert calls == [(["FIRST", "SECOND"], DAY_PERIOD, DAY_INTERVAL)]
    summary = pd.read_csv(report_dir / summary_filename(mode))
    assert list(summary["ticker"]) == ["FIRST", "SECOND"]
    if mode == "backtest":
        assert list(summary["total_trades"]) == [1, 1]
        assert list(summary["total_buys"]) == list(summary["total_sells"]) == [1, 1]
    else:
        assert "status" not in summary.columns
        assert "date" not in summary.columns
        assert list(summary["signal"]) == ["HOLD"] * 2
        assert list(summary["buy_signals"]) == [1, 1]
    for ticker in ("FIRST", "SECOND"):
        assert (destination / f"{ticker}.csv").is_file()
        if mode == "backtest":
            assert (report_dir / ticker / "signals.csv").is_file()
            assert (report_dir / ticker / "equity.csv").is_file()
            assert (report_dir / ticker / "trades.csv").is_file()
        else:
            assert not (report_dir / ticker).exists()
    if mode == "scan":
        assert sorted(path.name for path in report_dir.iterdir()) == ["rsi_bolinger.csv"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_bulk_download_and_cached_run_produce_identical_backtest_results(
    tmp_path, cached_prices, monkeypatch,
):
    _, ticker_file = cached_prices
    ticker_file.write_text("symbol\nFIRST\nSECOND\n")
    calls = mock_cached_download(monkeypatch, cached_prices)
    destination = tmp_path / "downloaded"
    download_report = tmp_path / "download-report"
    cache_report = tmp_path / "cache-report"
    arguments = ["--ticker-file", str(ticker_file), "--data-dir", str(destination)]

    assert main([*arguments, "--output-dir", str(download_report)], mode="backtest") == 0
    assert main([*arguments, "--output-dir", str(cache_report), "--no-download"], mode="backtest") == 0

    assert calls == [(["FIRST", "SECOND"], DAY_PERIOD, DAY_INTERVAL)]
    pd.testing.assert_frame_equal(
        pd.read_csv(download_report / BACKTEST_SUMMARY),
        pd.read_csv(cache_report / BACKTEST_SUMMARY),
    )
    for ticker in ("FIRST", "SECOND"):
        for filename in ("signals.csv", "trades.csv", "equity.csv", "orders.csv"):
            pd.testing.assert_frame_equal(
                pd.read_csv(download_report / ticker / filename),
                pd.read_csv(cache_report / ticker / filename),
                check_dtype=False,
            )
        assert json.loads((download_report / ticker / "metrics.json").read_text()) == json.loads(
            (cache_report / ticker / "metrics.json").read_text()
        )


@pytest.mark.parametrize("missing_ticker", ["absent", "all_nan"])
def test_partial_bulk_failure_keeps_successes_without_retry_or_stale_cache_fallback(
    tmp_path, cached_prices, monkeypatch, missing_ticker, capsys,
):
    data_dir, ticker_file = cached_prices
    ticker_file.write_text("symbol\nGOOD\nMISSING\nBROKEN\n")
    frame = pd.read_csv(data_dir / "TEST.csv", index_col=0, parse_dates=True)
    broken = frame.copy()
    broken.loc[broken.index[10], "Close"] = float("nan")
    parts = {"GOOD": frame, "BROKEN": broken}
    if missing_ticker == "all_nan":
        parts["MISSING"] = frame * float("nan")
    for ticker in ("MISSING", "BROKEN"):
        frame.to_csv(data_dir / f"{ticker}.csv")
    calls = []

    def partial_download(tickers, period, interval):
        calls.append((list(tickers), period, interval))
        return pd.concat(parts, axis=1)

    monkeypatch.setattr(downloader, "download_prices", partial_download)
    report_dir = tmp_path / "report"
    for ticker in ("MISSING", "BROKEN"):
        target = report_dir / ticker
        target.mkdir(parents=True)
        (target / "trades.csv").write_text("stale trade report")
        (target / "metrics.json").write_text('{"total_trades": 999}')

    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)], mode="backtest") == 1

    assert calls == [(["GOOD", "MISSING", "BROKEN"], DAY_PERIOD, DAY_INTERVAL)]
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert summary.iloc[0]["total_trades"] == 1
    assert summary.iloc[0]["total_buys"] == summary.iloc[0]["total_sells"] == 1
    assert summary.iloc[1:]["total_return_pct"].isna().all()
    assert summary.iloc[1:]["message"].notna().all()
    assert (report_dir / "GOOD" / "trades.csv").is_file()
    for ticker in ("MISSING", "BROKEN"):
        assert not (report_dir / ticker / "trades.csv").exists()
        assert not (report_dir / ticker / "metrics.json").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_whole_batch_failure_reports_all_tickers_after_one_attempt(
    tmp_path, cached_prices, monkeypatch, capsys,
):
    data_dir, ticker_file = cached_prices
    ticker_file.write_text("symbol\nFIRST\nSECOND\n")
    payload = (data_dir / "TEST.csv").read_bytes()
    for ticker in ("FIRST", "SECOND"):
        (data_dir / f"{ticker}.csv").write_bytes(payload)
    calls = []

    def fail_batch(tickers, period, interval):
        calls.append((list(tickers), period, interval))
        raise RuntimeError("batch connection failed")

    monkeypatch.setattr(downloader, "download_prices", fail_batch)
    report_dir = tmp_path / "report"
    assert main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)], mode="backtest") == 1

    assert calls == [(["FIRST", "SECOND"], DAY_PERIOD, DAY_INTERVAL)]
    summary = pd.read_csv(report_dir / BACKTEST_SUMMARY)
    assert list(summary["initial_capital"]) == [100_000, 100_000]
    assert summary["message"].str.contains("batch connection failed", regex=False).all()
    assert summary["total_trades"].isna().all()
    assert summary["total_return_pct"].isna().all()
    assert not (report_dir / "FIRST" / "trades.csv").exists()
    assert not (report_dir / "SECOND" / "trades.csv").exists()
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_parser_leaves_configurable_download_defaults_unset_and_flags_are_exclusive():
    parser = cli.build_parser("backtest")
    args = parser.parse_args([])
    assert args.download is None
    assert args.period is None
    assert args.interval is None
    for name in ("rsi_period", "bb_period", "bb_multiplier", "rsi_threshold"):
        assert getattr(args, name) is None
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(["--download", "--no-download"])
    assert caught.value.code == 2


@pytest.mark.parametrize("settings", [
    {"period": "7y", "interval": "1d"},
    {"period": "7y", "interval": "1d", "is_download": "sometimes"},
    {"period": "7y", "interval": "1d", "is_download": 1},
    {"period": "7y", "interval": "1d", "is_download": None},
    {"period": "", "interval": "1d", "is_download": True},
    {"period": "   ", "interval": "1d", "is_download": True},
    {"period": 7, "interval": "1d", "is_download": True},
    {"period": "7y", "interval": "", "is_download": True},
    {"period": "7y", "interval": None, "is_download": True},
])
def test_invalid_download_config_fails_clearly_before_reports_are_written(tmp_path, cached_prices, monkeypatch, capsys, settings):
    data_dir, ticker_file = cached_prices
    config_path = configure_download(monkeypatch, tmp_path, settings)
    report_dir = tmp_path / "report"
    with pytest.raises(SystemExit) as caught:
        main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
              "--output-dir", str(report_dir)], mode="backtest")
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert str(config_path) in error
    assert "download" in error.lower()
    assert "Traceback" not in error
    assert not report_dir.exists()


@pytest.mark.parametrize("content", ["{invalid json", "[]", '{"download": []}', '{}'])
def test_invalid_config_file_reports_its_path(tmp_path, cached_prices, monkeypatch, capsys, content):
    data_dir, ticker_file = cached_prices
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "day.json"
    config_path.write_text(content)
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    with pytest.raises(SystemExit) as caught:
        main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
              "--output-dir", str(tmp_path / "report")], mode="backtest")
    assert caught.value.code == 2
    assert str(config_path) in capsys.readouterr().err


def test_missing_config_file_reports_its_path(tmp_path, cached_prices, monkeypatch, capsys):
    data_dir, ticker_file = cached_prices
    config_dir = tmp_path / "missing-config"
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    with pytest.raises(SystemExit) as caught:
        main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
              "--output-dir", str(tmp_path / "report")], mode="backtest")
    assert caught.value.code == 2
    assert str(config_dir / "day.json") in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["scan", "backtest"])
@pytest.mark.parametrize("override", [False, True], ids=["from-config", "cli-override"])
def test_strategy_parameters_follow_config_and_explicit_cli_overrides(
    tmp_path, cached_prices, monkeypatch, mode, override,
):
    from stock_trade_analyser.strategy.rsi_bollinger import RsiBollingerStrategy

    data_dir, ticker_file = cached_prices
    configured = {
        "rsi_period": 2, "bb_period": 3, "bb_multiplier": 1.0, "rsi_threshold": 55.0,
    }
    selected = {
        "rsi_period": 3, "bb_period": 5, "bb_multiplier": 1.5, "rsi_threshold": 45.0,
    } if override else configured
    config_path = configure_download(monkeypatch, tmp_path, {
        "period": "1y", "interval": "1d", "is_download": False,
    }, strategy=configured)
    report_dir = tmp_path / "report"
    arguments = ["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
                 "--output-dir", str(report_dir)]
    if override:
        for name, value in selected.items():
            arguments.extend([f"--{name.replace('_', '-')}", str(value)])

    received = []
    generate = RsiBollingerStrategy.generate_signals

    def record_parameters(strategy, prices):
        received.append({name: getattr(strategy, name) for name in selected})
        return generate(strategy, prices)

    monkeypatch.setattr(RsiBollingerStrategy, "generate_signals", record_parameters)
    assert main(arguments, mode=mode) == 0
    assert received == [selected]
    prices = load_prices(data_dir / "TEST.csv")
    window = prices["close"].tail(selected["bb_period"])
    expected_lower = window.mean() - selected["bb_multiplier"] * window.std(ddof=0)
    if mode == "backtest":
        recorded = json.loads((report_dir / "config.json").read_text())
        assert recorded["config_file"] == str(config_path)
        assert {name: recorded[name] for name in selected} == selected
        signals = pd.read_csv(report_dir / "TEST" / "signals.csv")
        assert signals["bb_lower"].first_valid_index() == selected["bb_period"] - 1
        assert signals.iloc[-1]["bb_lower"] == pytest.approx(expected_lower)
    else:
        summary = pd.read_csv(report_dir / SCAN_SUMMARY)
        assert summary.iloc[0]["bb_lower"] == pytest.approx(expected_lower)
        assert sorted(path.name for path in report_dir.iterdir()) == [SCAN_SUMMARY]


@pytest.mark.parametrize("strategy", [None, [], {}, {"rsi_period": 6}])
def test_missing_or_incomplete_strategy_config_fails_before_reports(
    tmp_path, cached_prices, monkeypatch, capsys, strategy,
):
    data_dir, ticker_file = cached_prices
    config_path = configure_download(monkeypatch, tmp_path, {
        "period": "1y", "interval": "1d", "is_download": False,
    })
    content = json.loads(config_path.read_text())
    if strategy is None:
        del content["rsi_bollinger"]
    else:
        content["rsi_bollinger"] = strategy
    config_path.write_text(json.dumps(content))
    report_dir = tmp_path / "report"
    with pytest.raises(SystemExit) as caught:
        main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
              "--output-dir", str(report_dir)], mode="backtest")
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert str(config_path) in error
    assert "rsi_bollinger" in error
    assert not report_dir.exists()


@pytest.mark.parametrize(("name", "value"), [
    ("rsi_period", "6"), ("rsi_period", 0), ("bb_period", True),
    ("bb_multiplier", "2.0"), ("bb_multiplier", -1),
    ("rsi_threshold", None), ("rsi_threshold", 100),
])
def test_invalid_strategy_values_report_field_and_config_path(
    tmp_path, cached_prices, monkeypatch, capsys, name, value,
):
    data_dir, ticker_file = cached_prices
    strategy = {**DAY_STRATEGY_CONFIG, name: value}
    config_path = configure_download(monkeypatch, tmp_path, {
        "period": "1y", "interval": "1d", "is_download": False,
    }, strategy=strategy)
    report_dir = tmp_path / "report"
    with pytest.raises(SystemExit) as caught:
        main(["--ticker-file", str(ticker_file), "--data-dir", str(data_dir),
              "--output-dir", str(report_dir)], mode="backtest")
    assert caught.value.code == 2
    error = capsys.readouterr().err
    assert str(config_path) in error
    assert name in error
    assert "Traceback" not in error
    assert not report_dir.exists()
