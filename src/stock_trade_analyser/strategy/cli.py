"""Quiet scanning and long-only backtesting for the configured strategies."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

from ..tools import data_utils, downloader, file_utils
from .registry import load_registry


ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
try:
    REGISTRY = load_registry()
except ValueError as error:
    raise SystemExit(str(error)) from None
# Retained for callers that look up report names; the registry owns the values.
SUMMARY_FILENAMES = {
    name: {"scan": definition.scan_filename, "backtest": definition.backtest_filename}
    for name, definition in REGISTRY.strategies.items()
}


def build_parser(mode: str = "scan", strategy_name: str | None = None) -> argparse.ArgumentParser:
    """Build the parser before importing numerical or download dependencies."""
    strategy_name = strategy_name or REGISTRY.default_strategy
    definition = REGISTRY.strategies[strategy_name]
    backtest = mode == "backtest"
    summary_filename = definition.backtest_filename if backtest else definition.scan_filename
    parser = argparse.ArgumentParser(
        description=(
            f"Backtest {definition.label}: buy entries and long exits only."
            if backtest
            else f"Scan {definition.label} buy signals and long exits."
        )
    )
    parser.add_argument(
        "--strategy", choices=tuple(REGISTRY.strategies), default=strategy_name,
        help=f"Strategy to run. Defaults to {strategy_name}.",
    )
    parser.add_argument(
        "--ticker-file", default=None,
        help="Ticker CSV; bare filenames resolve under data/. Defaults to ticker_file in the selected config.",
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
    definition.add_arguments(parser)
    if backtest:
        parser.add_argument("--initial-capital", type=float, help="Starting capital; defaults to the selected strategy's backtest settings.")
        parser.add_argument("--quantity", type=int, help="Fixed shares per entry; selects fixed sizing unless an explicit sizing mode is given.")
        parser.add_argument(
            "--position-sizing", choices=("capital", "fixed"),
            help="Size entries from available capital or a fixed quantity; defaults to the selected backtest settings.",
        )
        parser.add_argument(
            "--capital-allocation-pct", type=float,
            help="Percentage of available cash to use per entry in capital sizing mode.",
        )
        parser.add_argument("--commission-bps", type=float, help="Commission per fill in basis points (10 bps = 0.1%%).")
        parser.add_argument("--slippage-bps", type=float, help="Adverse slippage per fill in basis points.")
        closing_options = parser.add_mutually_exclusive_group()
        closing_options.add_argument(
            "--keep-open", dest="close_at_end", action="store_false", default=None,
            help="Mark an open final position to market instead of closing it at the last close.",
        )
        closing_options.add_argument(
            "--close-at-end", dest="close_at_end", action="store_true",
            help="Close a remaining long at the final close, overriding backtest.close_at_end.",
        )
    return parser


def load_run_config(data_type: str, strategy_name: str | None = None) -> tuple[dict[str, Any], Path]:
    """Read download and strategy settings from the selected config once."""
    strategy_name = strategy_name or REGISTRY.default_strategy
    path = CONFIG_DIR / f"{data_type}.json"
    config = file_utils.read_json(path)
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
        raise ValueError(f"Config {path} must contain a strategy object named {strategy_name}")
    for name in REGISTRY.strategies[strategy_name].parameter_names:
        if name not in strategy_settings:
            raise ValueError(f"Config {path}: missing {strategy_name}.{name}")
    return config, path


def main(
    argv: list[str] | None = None, *, mode: str = "scan", strategy_name: str | None = None,
) -> int:
    # Select the help text and output paths without importing price libraries.
    selector = argparse.ArgumentParser(add_help=False)
    selector.add_argument(
        "--strategy", choices=tuple(REGISTRY.strategies),
        default=strategy_name or REGISTRY.default_strategy,
    )
    selection, _ = selector.parse_known_args(argv)
    parser = build_parser(mode, selection.strategy)
    args = parser.parse_args(argv)
    strategy_name = args.strategy
    definition = REGISTRY.strategies[strategy_name]
    parameter_names = definition.parameter_names
    try:
        run_settings, config_path = load_run_config(args.data_type, strategy_name)
        if args.ticker_file is None:
            configured_ticker_file = run_settings.get("ticker_file")
            if not isinstance(configured_ticker_file, str) or not configured_ticker_file.strip():
                raise ValueError(f"Config {config_path}: ticker_file must be a nonempty path")
            args.ticker_file = configured_ticker_file
        ticker_path = file_utils.resolve_ticker_file(args.ticker_file, root_dir=ROOT_DIR)
        tickers = file_utils.read_tickers(ticker_path)
        download_settings = run_settings["download"]
        args.download = download_settings["is_download"] if args.download is None else args.download
        args.period = download_settings["period"] if args.period is None else args.period
        args.interval = download_settings["interval"] if args.interval is None else args.interval
        for name in parameter_names:
            if getattr(args, name, None) is None:
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
    try:
        parameters = {name: getattr(args, name) for name in parameter_names}
        strategy = definition.create(parameters)
        minimum_rows = strategy.minimum_bars
        entry_description = definition.describe(definition.entry_description, strategy)
        exit_description = definition.describe(definition.exit_description, strategy)
        no_buys_message = definition.describe(definition.no_buys_message, strategy)
        minimum_description = definition.describe(definition.minimum_description, strategy)
    except (ValueError, TypeError, ImportError, AttributeError) as error:
        parser.error(f"Config {config_path}, {strategy_name} settings: {error}")

    try:
        engine = None
        config = None
        if mode == "backtest":
            from ..backtest.engine import BacktestConfig, LongOnlyBacktester
            overrides = run_settings[strategy_name].get("backtest", {})
            if not isinstance(overrides, dict):
                raise ValueError(f"Config {config_path}: {strategy_name}.backtest must be an object")
            unknown = set(overrides).difference(REGISTRY.backtest_defaults)
            if unknown:
                raise ValueError(f"Config {config_path}: unknown {strategy_name}.backtest settings: {', '.join(sorted(unknown))}")
            backtest_settings = REGISTRY.backtest_defaults.copy()
            for layer in (definition.backtest, overrides):
                backtest_settings.update(layer)
                # A quantity override retains its existing fixed-size meaning.
                if "quantity" in layer and "position_sizing" not in layer:
                    backtest_settings["position_sizing"] = "fixed"
            for name in backtest_settings:
                value = getattr(args, name, None)
                if value is not None:
                    backtest_settings[name] = value
            if args.quantity is not None:
                if args.position_sizing == "capital":
                    raise ValueError("--quantity cannot be combined with --position-sizing capital")
                if args.position_sizing is None:
                    backtest_settings["position_sizing"] = "fixed"
            if args.capital_allocation_pct is not None and backtest_settings["position_sizing"] != "capital":
                raise ValueError("--capital-allocation-pct requires --position-sizing capital")
            config = BacktestConfig(**backtest_settings)
            engine = LongOnlyBacktester(config)
    except ValueError as error:
        parser.error(f"Config {config_path}, {strategy_name} backtest settings: {error}")

    data_dir = Path(args.data_dir).expanduser() if args.data_dir else ROOT_DIR / "output" / args.data_type
    default_output = "backtests" if mode == "backtest" else "predictions"
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else ROOT_DIR / default_output / strategy_name / args.data_type
    summary_filename = definition.backtest_filename if engine else definition.scan_filename
    summary_path = (output_dir if args.output_dir else ROOT_DIR / "predictions") / summary_filename
    if config is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_config = {
            "strategy": definition.report_label,
            **parameters,
            "entry": entry_description,
            "exit": exit_description,
            "ticker_file": str(ticker_path),
            "data_dir": str(data_dir), "data_type": args.data_type,
            "summary_file": str(summary_path),
            "config_file": str(config_path),
            "download": args.download, "period": args.period, "interval": args.interval,
            "backtest": asdict(config),
        }
        file_utils.write_json(output_dir / "config.json", run_config)
    else:
        (output_dir / "config.json").unlink(missing_ok=True)
    rows: list[dict[str, Any]] = []
    errors = 0
    downloaded = None
    download_error = None
    if args.download:
        try:
            downloaded = downloader.download_prices(tickers, args.period, args.interval)
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
                prices = data_utils.extract_downloaded_prices(downloaded, ticker, tickers)
                source.parent.mkdir(parents=True, exist_ok=True)
                file_utils.write_csv(source, prices, index_label="date")
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
                prices = data_utils.load_prices(source)
            signals = strategy.generate_signals(prices)
            if engine:
                target.mkdir(parents=True, exist_ok=True)
                file_utils.write_csv(target / "signals.csv", signals, index_label="date")
            buy_count = int(signals["buy_signal"].sum())
            exit_count = int(signals["exit_signal"].sum())
            row.update(buy_signals=buy_count, exit_signals=exit_count)
            data_message = strategy.data_message(signals)
            if data_message:
                row.update(status="NO_DATA_SIGNALS", message=data_message)
            elif len(signals) < minimum_rows:
                row.update(status="INSUFFICIENT_DATA", message=f"{len(signals)} bars; at least {minimum_rows} required for {minimum_description}.")
            elif not buy_count:
                row.update(status="NO_BUY_SIGNALS", message=no_buys_message)
            else:
                row["status"] = "COMPLETED" if engine else "OK"
            if engine:
                result = engine.run(signals)
                for name in ("trades", "equity", "orders"):
                    file_utils.write_csv(target / f"{name}.csv", getattr(result, name), index=None)
                file_utils.write_json(target / "metrics.json", result.metrics)
                for key in REGISTRY.backtest_columns:
                    if key not in {"ticker", "message"}:
                        row[key] = result.metrics[key]
                if row["status"] == "COMPLETED" and not result.metrics["total_trades"]:
                    message = (
                        "No closed trades; a long position remains open and is marked to market."
                        if result.metrics.get("open_positions")
                        else "No closed trades; see orders.csv and metrics.json for skipped entries."
                    )
                    row.update(status="NO_TRADES", message=message)
            else:
                latest = signals.iloc[-1]
                row.update(signal=latest["signal"], close=latest["close"])
                row.update({name: latest[name] for name in definition.scan_fields})
        except (OSError, ValueError, KeyError, ImportError, RuntimeError) as error:
            row.update(status="ERROR", message=str(error))
            errors += 1
        rows.append(row)

    columns = (list(REGISTRY.backtest_columns) if engine else [
        "ticker", "signal", "close", *definition.scan_fields,
        "buy_signals", "exit_signals", "message",
    ])
    file_utils.write_summary(summary_path, rows, columns)
    if not engine and not args.output_dir:
        for directory in (output_dir, output_dir.parent):
            try:
                directory.rmdir()
            except OSError:
                pass
    return 1 if errors else 0
