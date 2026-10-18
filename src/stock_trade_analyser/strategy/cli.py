"""Quiet scanning and long-only backtesting for the configured strategies."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
STRATEGY_PARAMETERS = ("rsi_period", "bb_period", "bb_multiplier", "rsi_threshold")
SUPERTREND_PARAMETERS = (
    "atr_period", "atr_multiplier", "source", "change_atr", "from_date", "to_date", "timezone",
)
STRATEGY_PARAMETER_NAMES = {
    "rsi_bollinger": STRATEGY_PARAMETERS,
    "supertrend": SUPERTREND_PARAMETERS,
}
SCAN_SUMMARY_FILENAME = "rsi_bolinger.csv"
BACKTEST_SUMMARY_FILENAME = "rsi_bolinger_backtest.csv"
SUMMARY_FILENAMES = {
    "rsi_bollinger": {"scan": SCAN_SUMMARY_FILENAME, "backtest": BACKTEST_SUMMARY_FILENAME},
    "supertrend": {"scan": "supertrend.csv", "backtest": "supertrend_backtest.csv"},
}
SCAN_SUMMARY_COLUMNS = {
    "rsi_bollinger": [
        "ticker", "signal", "close", "rsi", "bb_lower", "bb_upper", "buy_signals", "exit_signals", "message",
    ],
    "supertrend": [
        "ticker", "signal", "close", "trend", "supertrend", "atr", "buy_signals", "exit_signals", "message",
    ],
}
BACKTEST_SUMMARY_COLUMNS = [
    "ticker", "initial_capital", "final_equity", "total_trades", "total_buys",
    "total_sells", "winning_trades", "losing_trades", "breakeven_trades",
    "net_profit", "total_return_pct", "win_rate_pct", "max_drawdown_pct", "message",
]


def build_parser(mode: str = "scan", strategy_name: str = "rsi_bollinger") -> argparse.ArgumentParser:
    """Build the parser before importing numerical or download dependencies."""
    backtest = mode == "backtest"
    summary_filename = SUMMARY_FILENAMES[strategy_name]["backtest" if backtest else "scan"]
    strategy_label = "SuperTrend" if strategy_name == "supertrend" else "ChartArt RSI + Bollinger Bands"
    parser = argparse.ArgumentParser(
        description=(
            f"Backtest {strategy_label}: buy entries and long exits only."
            if backtest
            else f"Scan {strategy_label} buy signals and long exits."
        )
    )
    parser.add_argument(
        "--strategy", choices=tuple(STRATEGY_PARAMETER_NAMES), default=strategy_name,
        help=f"Strategy to run. Defaults to {strategy_name}.",
    )
    parser.add_argument(
        "--ticker-file", default=None,
        help="Ticker CSV; bare filenames resolve under data/. Default: data/tickers.csv.",
    )
    parser.add_argument("--data-type", choices=("day", "intraday"), default="day")
    parser.add_argument("--data-dir", help="Directory containing SYMBOL.csv files; default: output/DATA_TYPE.")
    parser.add_argument(
        "--output-dir", "--report-dir", dest="output_dir",
        help=(f"Override the report directory, including {summary_filename}. "
              f"Default summary: predictions/{summary_filename}. "
              + (f"Detailed reports: backtests/{strategy_name}/DATA_TYPE." if backtest
                 else "Scanner writes only the final summary.")),
    )
    download_options = parser.add_mutually_exclusive_group()
    download_options.add_argument(
        "--download", dest="download", action="store_true", default=None,
        help="Enable downloading, overriding download.is_download in the selected config.",
    )
    download_options.add_argument(
        "--no-download", dest="download", action="store_false",
        help="Use cached CSVs only, overriding download.is_download in the selected config.",
    )
    parser.add_argument("--period", help="Yahoo Finance history period; defaults to download.period in the selected config.")
    parser.add_argument("--interval", help="Yahoo Finance bar interval; defaults to download.interval in the selected config.")
    parser.add_argument("--rsi-period", type=int, help="Defaults to rsi_bollinger.rsi_period in the selected config.")
    parser.add_argument("--bb-period", type=int, help="Defaults to rsi_bollinger.bb_period in the selected config.")
    parser.add_argument("--bb-multiplier", type=float, help="Defaults to rsi_bollinger.bb_multiplier in the selected config.")
    parser.add_argument("--rsi-threshold", type=float, help="Defaults to rsi_bollinger.rsi_threshold in the selected config.")
    parser.add_argument("--atr-period", type=int, help="Defaults to supertrend.atr_period in the selected config.")
    parser.add_argument("--atr-multiplier", type=float, help="Defaults to supertrend.atr_multiplier in the selected config.")
    parser.add_argument("--source", help="SuperTrend price source; defaults to supertrend.source in the selected config.")
    parser.add_argument(
        "--change-atr", action=argparse.BooleanOptionalAction, default=None,
        help="Use Wilder ATR (--no-change-atr selects SMA); defaults to supertrend.change_atr.",
    )
    parser.add_argument("--from-date", help="Inclusive SuperTrend signal start date; defaults to supertrend.from_date.")
    parser.add_argument("--to-date", help="Inclusive SuperTrend signal end date; defaults to supertrend.to_date.")
    parser.add_argument("--timezone", help="Timezone of SuperTrend trading dates; defaults to supertrend.timezone.")
    if backtest:
        parser.add_argument("--initial-capital", type=float, default=100000.0)
        parser.add_argument("--quantity", type=int, default=1, help="Fixed shares per entry. Default: 1.")
        parser.add_argument("--commission-bps", type=float, default=0.0, help="Commission per fill in basis points (10 bps = 0.1%%).")
        parser.add_argument("--slippage-bps", type=float, default=0.0, help="Adverse slippage per fill in basis points.")
        parser.add_argument("--keep-open", action="store_true", help="Mark an open final position to market instead of closing it at the last close.")
    return parser


def load_run_config(data_type: str, strategy_name: str = "rsi_bollinger") -> tuple[dict[str, Any], Path]:
    """Read download and strategy settings from the selected config once."""
    path = CONFIG_DIR / f"{data_type}.json"
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in config {path}: {error.msg} (line {error.lineno})") from error
    if not isinstance(config, dict) or not isinstance(config.get("download"), dict):
        raise ValueError(f"Config {path} must contain a download object")
    settings = config["download"].copy()
    for name in ("period", "interval"):
        value = settings.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config {path}: download.{name} must be a nonempty string")
        settings[name] = value.strip()
    enabled = settings.get("is_download")
    if isinstance(enabled, str) and enabled.strip().lower() in {"true", "false"}:
        enabled = enabled.strip().lower() == "true"
    if not isinstance(enabled, bool):
        raise ValueError(f"Config {path}: download.is_download must be true or false")
    settings["is_download"] = enabled
    config["download"] = settings
    strategy_settings = config.get(strategy_name)
    if not isinstance(strategy_settings, dict):
        article = "an" if strategy_name == "rsi_bollinger" else "a"
        raise ValueError(f"Config {path} must contain {article} {strategy_name} object")
    for name in STRATEGY_PARAMETER_NAMES[strategy_name]:
        if name not in strategy_settings:
            raise ValueError(f"Config {path}: missing {strategy_name}.{name}")
    return config, path


def resolve_ticker_file(value: str | None) -> Path:
    if value is None:
        return ROOT_DIR / "data" / "tickers.csv"
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return ROOT_DIR / ("data" if path.parent == Path(".") else "") / path


def read_tickers(path: Path) -> list[str]:
    """Read headered or headerless ticker lists, skipping blanks and duplicates."""
    tickers: list[str] = []
    seen: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.reader(handle):
            if not row or not row[0].strip():
                continue
            ticker = row[0].strip()
            if ticker.lower() in {"ticker", "tickers", "symbol", "symbols"}:
                continue
            if "/" in ticker or "\\" in ticker or ticker in {".", ".."}:
                raise ValueError(f"Ticker must be a symbol, not a path: {ticker!r}")
            if ticker not in seen:
                seen.add(ticker)
                tickers.append(ticker)
    if not tickers:
        raise ValueError(f"No ticker symbols found in {path}")
    return tickers


def load_prices(path: Path):
    """Load ordinary OHLC CSVs with a date column or exported date index."""
    import pandas as pd

    # pandas silently renames repeated headers (Close, Close -> Close, Close.1),
    # so inspect the source header before loading or choosing any OHLC column.
    with path.open(newline="", encoding="utf-8-sig") as handle:
        header = next((row for row in csv.reader(handle) if row), None)
    if header is None:
        raise ValueError("CSV contains no header or price rows")
    normalized_header = [column.strip().lower().replace(" ", "_") for column in header]
    if len(set(normalized_header)) != len(normalized_header):
        raise ValueError("CSV contains duplicate column names")
    # Keep timestamps as text: an integer YYYYMMDD must never become nanoseconds
    # since the epoch before date-format validation runs.
    frame = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    if frame.empty:
        raise ValueError("CSV contains no price rows")
    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if frame.columns.duplicated().any():
        raise ValueError("CSV contains duplicate column names")
    date_column = next((name for name in ("date", "datetime", "timestamp", "time") if name in frame.columns), None)
    if date_column is None:
        first_column = frame.columns[0]
        if first_column.startswith("unnamed:") or first_column not in {"open", "high", "low", "close", "volume", "adj_close"}:
            date_column = first_column
    if date_column is None:
        raise ValueError("CSV needs a date/datetime/timestamp column or a saved date index")
    date_text = frame.pop(date_column).astype("string").str.strip()
    compact_dates = date_text.str.fullmatch(r"\d{8}", na=False)
    numeric_dates = date_text.str.fullmatch(r"[+-]?\d+(?:\.\d+)?", na=False)
    if (numeric_dates & ~compact_dates).any():
        raise ValueError("CSV contains ambiguous numeric dates; use ISO dates or YYYYMMDD")
    dates = pd.to_datetime(date_text, format="mixed", errors="coerce", utc=True)
    if compact_dates.any():
        dates.loc[compact_dates] = pd.to_datetime(
            date_text.loc[compact_dates], format="%Y%m%d", errors="coerce", utc=True,
        )
    if dates.isna().any():
        raise ValueError("CSV contains invalid dates")
    frame.index = pd.DatetimeIndex(dates).tz_convert(None)
    frame.index.name = "date"
    if frame.index.has_duplicates:
        raise ValueError("CSV contains duplicate dates")
    required = {"open", "high", "low", "close"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
    for column in required | ({"volume"} if "volume" in frame.columns else set()):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if not frame[list(required)].map(lambda value: math.isfinite(value)).all().all():
        raise ValueError("CSV contains missing or non-finite OHLC values")
    return frame.sort_index()


def _warm_download_cache() -> None:
    """Initialize Yahoo's local caches before its download workers start."""
    from yfinance import cache

    for getter in (cache.get_tz_cache, cache.get_cookie_cache):
        try:
            getter().initialise()
        except Exception:
            # Cache initialization is best effort, as in the shared downloader.
            pass


def download_prices(tickers: list[str], period: str, interval: str):
    """Fetch the whole universe once; return the batch for in-memory use."""
    import yfinance as yf

    if not tickers:
        raise ValueError("No ticker symbols supplied for download")
    logger = logging.getLogger("yfinance")
    was_disabled = logger.disabled
    logger.disabled = True
    try:
        _warm_download_cache()
        frame = yf.download(
            tickers=tickers, period=period, interval=interval,
            auto_adjust=False, progress=False, threads=True, group_by="ticker",
        )
    finally:
        logger.disabled = was_disabled
    if frame is None or frame.empty:
        raise ValueError("Yahoo Finance returned no price data")
    return frame


def extract_downloaded_prices(downloaded, ticker: str, tickers: list[str]):
    """Take one symbol from a batch without mutating or rereading the batch."""
    import pandas as pd
    from .rsi_bollinger import normalize_ohlcv

    if not isinstance(downloaded, pd.DataFrame) or downloaded.empty:
        raise ValueError("Yahoo Finance returned no price data")
    required = {"open", "high", "low", "close"}
    if isinstance(downloaded.columns, pd.MultiIndex):
        if downloaded.columns.nlevels != 2:
            raise ValueError("Downloaded columns must contain ticker and price levels")

        def symbol_label(level):
            return next((label for label in downloaded.columns.get_level_values(level).unique()
                         if str(label).upper() == ticker.upper()), None)

        named_levels = [index for index, name in enumerate(downloaded.columns.names)
                        if str(name).lower() in {"ticker", "symbol"}]
        if len(named_levels) == 1:
            level = named_levels[0]
            label = symbol_label(level)
            if label is None:
                raise ValueError(f"Downloaded data does not contain ticker {ticker}")
        else:
            candidates = []
            for index in range(2):
                label = symbol_label(index)
                if label is not None:
                    fields = downloaded.xs(label, axis=1, level=index).columns
                    if required.issubset({str(field).strip().lower() for field in fields}):
                        candidates.append((index, label))
            if len(candidates) != 1:
                raise ValueError(f"Cannot identify downloaded OHLC columns for ticker {ticker}")
            level, label = candidates[0]
        frame = downloaded.xs(label, axis=1, level=level).copy()
    elif len(tickers) == 1 and ticker.upper() == tickers[0].upper():
        frame = downloaded.copy()
    else:
        raise ValueError("Multiple tickers require downloaded columns grouped by ticker")

    frame.columns = [str(column).strip().lower().replace(" ", "_") for column in frame.columns]
    if frame.columns.has_duplicates:
        raise ValueError("Downloaded data has duplicate column names")
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing OHLC columns: {', '.join(sorted(missing))}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Downloaded prices must have a datetime index")
    if frame.index.tz is not None:
        frame.index = frame.index.tz_convert(None)
    frame.index.name = "date"
    # Different listing dates and trading calendars create empty rows in a
    # batch's outer join. A partially missing OHLC row still fails validation.
    frame = frame.dropna(subset=sorted(required), how="all")
    if frame.empty:
        raise ValueError(f"Yahoo Finance returned no price data for ticker {ticker}")
    if "volume" in frame:
        frame["volume"] = frame["volume"].fillna(0)
    return normalize_ohlcv(frame)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_json_safe(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(
    argv: list[str] | None = None, *, mode: str = "scan", strategy_name: str = "rsi_bollinger",
) -> int:
    # Select the help text and output paths without importing price libraries.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument("--strategy", choices=tuple(STRATEGY_PARAMETER_NAMES), default=strategy_name)
    selection, _ = selector.parse_known_args(argv)
    parser = build_parser(mode, selection.strategy)
    args = parser.parse_args(argv)
    strategy_name = args.strategy
    parameter_names = STRATEGY_PARAMETER_NAMES[strategy_name]
    try:
        tickers = read_tickers(resolve_ticker_file(args.ticker_file))
        run_settings, config_path = load_run_config(args.data_type, strategy_name)
        download_settings = run_settings["download"]
        args.download = download_settings["is_download"] if args.download is None else args.download
        args.period = download_settings["period"] if args.period is None else args.period
        args.interval = download_settings["interval"] if args.interval is None else args.interval
        for name in parameter_names:
            if getattr(args, name) is None:
                setattr(args, name, run_settings[strategy_name][name])
        if not args.period.strip() or not args.interval.strip():
            raise ValueError("--period and --interval must be nonempty strings")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    required_dependencies = ("numpy", "pandas", "yfinance") if args.download else ("numpy", "pandas")
    missing_dependencies = [
        name for name in required_dependencies
        if importlib.util.find_spec(name) is None
    ]
    if missing_dependencies:
        parser.error(
            f"Missing runtime dependencies: {', '.join(missing_dependencies)} "
            f"in {sys.executable}. Activate the project environment with "
            "'workon trade' before running python, or use the make command "
            "to select the existing trade environment. Run 'make setup' "
            "if that environment has not been created."
        )

    from dataclasses import asdict
    import pandas as pd
    try:
        parameters = {name: getattr(args, name) for name in parameter_names}
        if strategy_name == "supertrend":
            from .supertrend import SuperTrendStrategy
            strategy = SuperTrendStrategy(**parameters)
            minimum_rows = args.atr_period + (1 if args.change_atr else 2)
            entry_description = "Upward SuperTrend reversal inside the trading window; next bar open."
            exit_description = "Downward SuperTrend reversal inside the trading window; next bar open; no shorts."
            no_buys_message = "No upward SuperTrend reversal within configured trading window."
            minimum_description = "a SuperTrend reversal"
            strategy_label = "KivancOzbilgic SuperTrend (long only)"
        else:
            from .rsi_bollinger import RsiBollingerStrategy
            strategy = RsiBollingerStrategy(**parameters)
            minimum_rows = max(args.bb_period + 1, args.rsi_period + 2)
            entry_description = "Simultaneous RSI and lower-band upward crossovers; next bar open."
            exit_description = "Simultaneous RSI and upper-band downward crossovers; next bar open; no shorts."
            no_buys_message = "No simultaneous RSI/lower-band upward crossover."
            minimum_description = "both crossovers"
            strategy_label = "ChartArt RSI + Bollinger Bands (long only)"
    except ValueError as error:
        parser.error(f"Config {config_path}, {strategy_name} settings: {error}")

    try:
        engine = None
        config = None
        if mode == "backtest":
            from ..backtest.engine import BacktestConfig, LongOnlyBacktester
            config = BacktestConfig(
                initial_capital=args.initial_capital, quantity=args.quantity,
                commission_bps=args.commission_bps, slippage_bps=args.slippage_bps,
                close_at_end=not args.keep_open,
            )
            engine = LongOnlyBacktester(config)
    except ValueError as error:
        parser.error(str(error))

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ROOT_DIR / "output" / args.data_type
    default_output = "backtests" if mode == "backtest" else "predictions"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROOT_DIR / default_output / strategy_name / args.data_type
    summary_filename = SUMMARY_FILENAMES[strategy_name]["backtest" if engine else "scan"]
    summary_path = (output_dir if args.output_dir else ROOT_DIR / "predictions") / summary_filename
    if config is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "strategy": strategy_label,
            **parameters,
            "entry": entry_description,
            "exit": exit_description,
            "ticker_file": str(resolve_ticker_file(args.ticker_file)),
            "data_dir": str(data_dir), "data_type": args.data_type,
            "summary_file": str(summary_path),
            "config_file": str(config_path),
            "download": args.download, "period": args.period, "interval": args.interval,
            "backtest": asdict(config),
        }
        _write_json(output_dir / "config.json", run_config)
    else:
        (output_dir / "config.json").unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    errors = 0
    downloaded = None
    download_error = None
    if args.download:
        try:
            downloaded = download_prices(tickers, args.period, args.interval)
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            download_error = str(error)
    for ticker in tickers:
        row: dict[str, Any] = {"ticker": ticker, "status": "ERROR", "message": ""}
        if config is not None:
            row["initial_capital"] = float(config.initial_capital)
        target = output_dir / ticker
        try:
            # Remove previous generated outputs without touching user files.
            generated_files = (("signals.csv", "trades.csv", "equity.csv", "orders.csv", "metrics.json")
                               if engine else ("signals.csv",))
            for filename in generated_files:
                (target / filename).unlink(missing_ok=True)
            if not engine:
                try:
                    target.rmdir()
                except OSError:
                    pass
            source = data_dir / f"{ticker}.csv"
            if args.download:
                if download_error is not None:
                    raise ValueError(f"Batch download failed: {download_error}")
                prices = extract_downloaded_prices(downloaded, ticker, tickers)
                source.parent.mkdir(parents=True, exist_ok=True)
                prices.to_csv(source, index_label="date")
            else:
                if not source.is_file():
                    row.update(
                        status="MISSING_DATA",
                        message=(f"No cached CSV: {source}; downloading is disabled. "
                                 f"Enable download.is_download in {config_path}, use --download, or set --data-dir."),
                    )
                    errors += 1
                    rows.append(row)
                    continue
                prices = load_prices(source)
            signals = strategy.generate_signals(prices)
            if engine:
                target.mkdir(parents=True, exist_ok=True)
                signals.to_csv(target / "signals.csv", index_label="date")
            buy_count = int(signals["buy_signal"].sum())
            exit_count = int(signals["exit_signal"].sum())
            row.update(buy_signals=buy_count, exit_signals=exit_count)
            if len(signals) < minimum_rows:
                row.update(status="INSUFFICIENT_DATA", message=f"{len(signals)} bars; at least {minimum_rows} required for {minimum_description}.")
            elif not buy_count:
                row.update(status="NO_BUY_SIGNALS", message=no_buys_message)
            else:
                row["status"] = "COMPLETED" if engine else "OK"
            if engine:
                result = engine.run(signals)
                for name in ("trades", "equity", "orders"):
                    frame = getattr(result, name)
                    has_dates = isinstance(frame.index, pd.DatetimeIndex)
                    frame.to_csv(target / f"{name}.csv", index=has_dates, index_label="date" if has_dates else None)
                _write_json(target / "metrics.json", result.metrics)
                for key in BACKTEST_SUMMARY_COLUMNS:
                    if key not in {"ticker", "message"}:
                        row[key] = result.metrics[key]
                if row["status"] == "COMPLETED" and not row["total_trades"]:
                    message = (
                        "No closed trades; a long position remains open and is marked to market."
                        if result.metrics.get("open_positions")
                        else "No closed trades; see orders.csv and metrics.json for skipped entries."
                    )
                    row.update(status="NO_TRADES", message=message)
            else:
                latest = signals.iloc[-1]
                fields = (("trend", "supertrend", "atr") if strategy_name == "supertrend"
                          else ("rsi", "bb_lower", "bb_upper"))
                row.update(signal=latest["signal"], close=latest["close"])
                row.update({name: latest[name] for name in fields})
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            row.update(status="ERROR", message=str(error))
            errors += 1
        rows.append(row)

    columns = BACKTEST_SUMMARY_COLUMNS if engine else SCAN_SUMMARY_COLUMNS[strategy_name]
    summary = pd.DataFrame(rows).reindex(columns=columns)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    if not engine and not args.output_dir:
        for directory in (output_dir, output_dir.parent):
            try:
                directory.rmdir()
            except OSError:
                pass
    return 1 if errors else 0
