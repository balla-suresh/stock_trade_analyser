"""Registry changes must drive the common CLI without strategy-specific branches."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shlex
import subprocess
import sys

import pandas as pd
import pytest

from stock_trade_analyser.strategy import cli
from stock_trade_analyser.tools import downloader


# These cases verify explicit fixed-quantity execution, not the CLI default.
pytestmark = pytest.mark.usefixtures("fixed_position_registry")


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "src" / "stock_trade_analyser" / "config" / "strategies.json"


@pytest.fixture
def configured_project(tmp_path, monkeypatch):
    root = tmp_path / "project"
    config_dir = root / "config"
    prices_dir = root / "output" / "day"
    config_dir.mkdir(parents=True)
    prices_dir.mkdir(parents=True)
    (root / "data").mkdir()
    (root / "data" / "tickers.csv").write_text("ticker\nTEST\n", encoding="utf-8")
    settings = json.loads(
        (ROOT / "src" / "stock_trade_analyser" / "config" / "day.json").read_text()
    )
    settings["ticker_file"] = "data/tickers.csv"
    settings["download"]["is_download"] = False
    (config_dir / "day.json").write_text(json.dumps(settings), encoding="utf-8")
    closes = [100.0] * 200 + [90.0, 85.0, 80.0, 100.0, 110.0, 115.0, 120.0, 100.0, 100.0]
    opens = [closes[0], *closes[:-1]]
    pd.DataFrame(
        {
            "open": opens,
            "high": [max(a, b) + 1 for a, b in zip(opens, closes)],
            "low": [min(a, b) - 1 for a, b in zip(opens, closes)],
            "close": closes,
            "volume": 1_000,
        },
        index=pd.date_range("2024-01-01", periods=len(closes), name="date"),
    ).to_csv(prices_dir / "TEST.csv")

    def no_network(*args, **kwargs):
        pytest.fail("Registry integration tests must use their isolated price cache")

    monkeypatch.setattr(cli, "ROOT_DIR", root)
    monkeypatch.setattr(cli, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(downloader, "download_prices", no_network)
    return root


def edit_run_settings(root, change):
    path = root / "config" / "day.json"
    settings = json.loads(path.read_text())
    change(settings)
    path.write_text(json.dumps(settings), encoding="utf-8")


@pytest.mark.parametrize("module", ["strategy", "backtest"])
@pytest.mark.parametrize("strategy,own_flag,other_flags", [
    ("rsi_bollinger", "--rsi-period", ["--atr-period", "--lookback"]),
    ("supertrend", "--no-change-atr", ["--rsi-period", "--lookback"]),
    ("wyckoff", "--spring-recovery-pct", ["--rsi-period", "--atr-period"]),
])
def test_help_is_dependency_free_and_only_lists_selected_strategy_options(
    module, strategy, own_flag, other_flags,
):
    result = subprocess.run(
        [sys.executable, "-S", "-m", f"src.stock_trade_analyser.{module}",
         "--strategy", strategy, "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert own_flag in result.stdout
    for flag in other_flags:
        assert flag not in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize("strategy,unused_flag", [
    ("rsi_bollinger", "--atr-period"),
    ("supertrend", "--bb-period"),
    ("wyckoff", "--rsi-period"),
])
def test_cli_rejects_options_that_belong_to_another_strategy(strategy, unused_flag, capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--strategy", strategy, unused_flag, "3"])
    assert caught.value.code == 2
    assert "unrecognized arguments" in capsys.readouterr().err


def test_every_wyckoff_parameter_can_be_overridden_without_more_cli_code(configured_project, capsys):
    selected = {
        "lookback": 20,
        "volume_multiplier": 2.0,
        "climax_volume_addition": 0.75,
        "range_period": 5,
        "narrow_range_multiplier": 0.7,
        "preliminary_period": 10,
        "climax_period": 8,
        "sweep_period": 6,
        "rally_change_pct": 0.02,
        "secondary_test_tolerance_pct": 0.01,
        "spring_recovery_pct": 0.015,
        "test_range_tolerance": 0.03,
        "test_volume_multiplier": 0.9,
        "sos_volume_multiplier": 1.4,
        "buy_events": ["SPRING", "SOS"],
        "exit_events": ["UTAD"],
    }
    arguments = ["--strategy", "wyckoff"]
    for name, value in selected.items():
        arguments.append(f"--{name.replace('_', '-')}")
        arguments.extend(map(str, value if isinstance(value, list) else [value]))
    assert cli.main(arguments, mode="backtest") == 0
    metadata = json.loads(
        (configured_project / "backtests" / "wyckoff" / "day" / "config.json").read_text()
    )
    assert {name: metadata[name] for name in selected} == selected
    assert "SPRING, SOS" in metadata["entry"]
    assert "UTAD" in metadata["exit"]
    captured = capsys.readouterr()
    assert captured.out == captured.err == ""


def test_registry_controls_report_names_fields_and_labels(configured_project, monkeypatch):
    registry = deepcopy(cli.REGISTRY)
    registry.strategies["rsi_bollinger"] = replace(
        registry.strategies["rsi_bollinger"],
        label="Configured band strategy",
        report_label="Configured band strategy (long only)",
        scan_filename="configured_scan.csv",
        backtest_filename="configured_history.csv",
        scan_fields=("bb_upper", "rsi"),
        entry_description="Configured buy rule; next bar open.",
    )
    monkeypatch.setattr(cli, "REGISTRY", registry)
    assert "Configured band strategy" in cli.build_parser().description
    assert cli.main([]) == 0
    scan = pd.read_csv(configured_project / "predictions" / "configured_scan.csv")
    assert list(scan.columns) == [
        "ticker", "signal", "close", "bb_upper", "rsi", "buy_signals", "exit_signals", "message",
    ]
    assert cli.main([], mode="backtest") == 0
    predictions = configured_project / "predictions"
    assert sorted(path.name for path in predictions.iterdir()) == [
        "configured_history.csv", "configured_scan.csv",
    ]
    metadata = json.loads(
        (configured_project / "backtests" / "rsi_bollinger" / "day" / "config.json").read_text()
    )
    assert metadata["strategy"] == "Configured band strategy (long only)"
    assert metadata["entry"] == "Configured buy rule; next bar open."


def test_short_history_message_formats_configured_strategy_parameters(configured_project, monkeypatch):
    registry = deepcopy(cli.REGISTRY)
    registry.strategies["rsi_bollinger"] = replace(
        registry.strategies["rsi_bollinger"],
        minimum_description="Bollinger crossings with a {bb_period}-bar window",
    )
    monkeypatch.setattr(cli, "REGISTRY", registry)
    assert cli.main(["--bb-period", "300"]) == 0
    row = pd.read_csv(configured_project / "predictions" / "rsi_bolinger.csv").iloc[0]
    assert row["message"] == (
        "209 bars; at least 301 required for Bollinger crossings with a 300-bar window."
    )


@pytest.mark.parametrize("mode", ["scan", "backtest"])
def test_registering_a_fourth_strategy_needs_no_new_cli_branch(configured_project, monkeypatch, mode):
    registry = deepcopy(cli.REGISTRY)
    registry.strategies["custom_bands"] = replace(
        registry.strategies["rsi_bollinger"],
        name="custom_bands", label="Custom band rules", report_label="Custom band rules (long only)",
        scan_filename="custom_bands.csv", backtest_filename="custom_bands_backtest.csv",
    )
    monkeypatch.setattr(cli, "REGISTRY", registry)

    def add_strategy(settings):
        settings["custom_bands"] = settings.pop("rsi_bollinger")

    edit_run_settings(configured_project, add_strategy)
    assert cli.main(["--strategy", "custom_bands"], mode=mode) == 0
    filename = "custom_bands_backtest.csv" if mode == "backtest" else "custom_bands.csv"
    row = pd.read_csv(configured_project / "predictions" / filename).iloc[0]
    assert row["ticker"] == "TEST"
    count = "total_trades" if mode == "backtest" else "buy_signals"
    assert row[count] == 1
    if mode == "backtest":
        metadata = json.loads(
            (configured_project / "backtests" / "custom_bands" / "day" / "config.json").read_text()
        )
        assert metadata["strategy"] == "Custom band rules (long only)"


@pytest.mark.parametrize("layer", ["registry", "strategy", "run_config", "cli"])
def test_backtest_settings_merge_in_strategy_order(configured_project, monkeypatch, layer):
    registry = deepcopy(cli.REGISTRY)
    registry.backtest_defaults.update(initial_capital=10_000.0, quantity=2, commission_bps=1.0)
    expected_capital = 10_000.0
    expected_quantity = 2
    if layer in {"strategy", "run_config", "cli"}:
        registry.strategies["rsi_bollinger"].backtest.update(initial_capital=20_000.0, quantity=3)
        expected_capital, expected_quantity = 20_000.0, 3
    if layer in {"run_config", "cli"}:
        def configure_backtest(settings):
            settings["rsi_bollinger"]["backtest"] = {"initial_capital": 30_000.0, "quantity": 4}

        edit_run_settings(configured_project, configure_backtest)
        expected_capital, expected_quantity = 30_000.0, 4
    arguments = []
    if layer == "cli":
        arguments = ["--initial-capital", "40000", "--quantity", "5", "--commission-bps", "2"]
        expected_capital, expected_quantity = 40_000.0, 5
    monkeypatch.setattr(cli, "REGISTRY", registry)
    assert cli.main(arguments, mode="backtest") == 0
    metadata = json.loads(
        (configured_project / "backtests" / "rsi_bollinger" / "day" / "config.json").read_text()
    )
    assert metadata["backtest"]["initial_capital"] == expected_capital
    assert metadata["backtest"]["quantity"] == expected_quantity
    assert metadata["backtest"]["commission_bps"] == (2.0 if layer == "cli" else 1.0)
    row = pd.read_csv(configured_project / "predictions" / "rsi_bolinger_backtest.csv").iloc[0]
    assert row["initial_capital"] == expected_capital
    trades = pd.read_csv(configured_project / "backtests" / "rsi_bollinger" / "day" / "TEST" / "trades.csv")
    assert trades["quantity"].tolist() == [expected_quantity]


@pytest.mark.parametrize("configured_close,extra,expected_sells", [
    (False, [], 0),
    (True, ["--keep-open"], 0),
    (False, ["--close-at-end"], 1),
])
def test_final_position_uses_configured_close_behavior_and_cli_override(
    configured_project, configured_close, extra, expected_sells,
):
    def configure_backtest(settings):
        settings["supertrend"]["backtest"] = {"close_at_end": configured_close}
        settings["supertrend"].update(atr_period=3, atr_multiplier=1.0)

    edit_run_settings(configured_project, configure_backtest)
    # Leave the final upward reversal open: the last signal executes on the
    # final bar, with no following bearish reversal to close the position.
    closes = [100.0] * 5 + [80.0, 80.0, 120.0, 121.0]
    pd.DataFrame(
        {"open": closes, "high": [v + 1 for v in closes], "low": [v - 1 for v in closes],
         "close": closes, "volume": 1_000},
        index=pd.date_range("2024-01-01", periods=len(closes), name="date"),
    ).to_csv(configured_project / "output" / "day" / "TEST.csv")
    assert cli.main(["--strategy", "supertrend", *extra], mode="backtest") == 0
    row = pd.read_csv(configured_project / "predictions" / "supertrend_backtest.csv").iloc[0]
    assert row["total_buys"] == 1
    assert row["total_sells"] == row["total_trades"] == expected_sells
    if not expected_sells:
        assert "remains open" in row["message"]
    metadata = json.loads(
        (configured_project / "backtests" / "supertrend" / "day" / "config.json").read_text()
    )
    assert metadata["backtest"]["close_at_end"] is bool(expected_sells)


@pytest.mark.parametrize("settings,field", [
    ([], "backtest"),
    ({"unknown_quantity": 2}, "unknown_quantity"),
    ({"quantity": True}, "quantity"),
    ({"initial_capital": -1}, "initial_capital"),
    ({"close_at_end": "False"}, "close_at_end"),
])
def test_invalid_run_backtest_settings_fail_before_reports_are_written(
    configured_project, capsys, settings, field,
):
    def configure_backtest(config):
        config["rsi_bollinger"]["backtest"] = settings

    edit_run_settings(configured_project, configure_backtest)
    with pytest.raises(SystemExit) as caught:
        cli.main([], mode="backtest")
    assert caught.value.code == 2
    message = capsys.readouterr().err
    assert str(configured_project / "config" / "day.json") in message
    assert field in message
    assert "Traceback" not in message
    assert not (configured_project / "predictions").exists()
    assert not (configured_project / "backtests").exists()


def test_loading_registry_is_independent_of_the_run_config_folder(configured_project):
    from stock_trade_analyser.strategy.registry import load_registry

    assert not (configured_project / "config" / "strategies.json").exists()
    registry = load_registry()
    assert {"rsi_bollinger", "supertrend", "wyckoff"}.issubset(registry.strategies)


def test_registry_load_reports_malformed_json_and_source_path(tmp_path):
    from stock_trade_analyser.strategy.registry import load_registry

    path = tmp_path / "broken_strategies.json"
    path.write_text("{invalid JSON", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_registry(path)
    assert str(path) in str(caught.value)
    assert "line 1" in str(caught.value)


def test_registry_load_reports_missing_file_with_source_path(tmp_path):
    from stock_trade_analyser.strategy.registry import load_registry

    path = tmp_path / "missing_strategies.json"
    with pytest.raises(ValueError) as caught:
        load_registry(path)
    assert str(path) in str(caught.value)


@pytest.mark.parametrize("fault,field", [
    ("default", "default_strategy"),
    ("parameter_type", "parameters.rsi_period"),
    ("filename", "scan_filename"),
    ("missing_label", "label"),
    ("reserved_scan_field", "scan_fields"),
    ("duplicate_scan_field", "scan_fields"),
    ("unknown_backtest_key", "backtest"),
    ("invalid_backtest_type", "quantity"),
    ("unknown_template_parameter", "entry_description"),
    ("reserved_help", "help"),
    ("reserved_h", "h"),
    ("duplicate_report_filename", "scan_filename"),
])
def test_registry_validation_identifies_the_invalid_setting_and_source_path(tmp_path, fault, field):
    from stock_trade_analyser.strategy.registry import load_registry

    settings = json.loads(REGISTRY_PATH.read_text())
    definition = settings["strategies"]["rsi_bollinger"]
    if fault == "default":
        settings["default_strategy"] = "unknown_strategy"
    elif fault == "parameter_type":
        definition["parameters"]["rsi_period"] = "unregistered_type"
    elif fault == "filename":
        definition["scan_filename"] = "../other.csv"
    elif fault == "missing_label":
        del definition["label"]
    elif fault == "reserved_scan_field":
        definition["scan_fields"] = ["ticker"]
    elif fault == "duplicate_scan_field":
        definition["scan_fields"] = ["rsi", "rsi"]
    elif fault == "unknown_backtest_key":
        definition["backtest"] = {"unknown_quantity": 10}
    elif fault == "invalid_backtest_type":
        settings["backtest_defaults"]["quantity"] = True
    elif fault == "unknown_template_parameter":
        definition["entry_description"] = "Buy when {unconfigured_rule} occurs."
    elif fault in {"reserved_help", "reserved_h"}:
        definition["parameters"][fault.removeprefix("reserved_")] = "int"
    elif fault == "duplicate_report_filename":
        settings["strategies"]["supertrend"]["scan_filename"] = definition["scan_filename"]
    path = tmp_path / "invalid_strategies.json"
    path.write_text(json.dumps(settings), encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        load_registry(path)
    assert str(path) in str(caught.value)
    assert field in str(caught.value)


def test_registry_default_strategy_selects_the_parser_and_runtime(configured_project, monkeypatch):
    monkeypatch.setattr(cli, "REGISTRY", replace(cli.REGISTRY, default_strategy="supertrend"))
    assert "SuperTrend" in cli.build_parser().description
    assert cli.main([]) == 0
    predictions = configured_project / "predictions"
    assert sorted(path.name for path in predictions.iterdir()) == ["supertrend.csv"]


def test_cli_import_reports_invalid_registry_without_traceback():
    code = "\n".join([
        "import importlib",
        "import sys",
        f"sys.path.insert(0, {str(ROOT / 'src')!r})",
        "from stock_trade_analyser.strategy import registry",
        "def invalid_registry():",
        "    raise ValueError('Config /temporary/strategies.json: invalid metadata')",
        "registry.load_registry = invalid_registry",
        "importlib.import_module('stock_trade_analyser.strategy.cli')",
    ])
    result = subprocess.run(
        [sys.executable, "-S", "-c", code], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert result.stderr.strip() == "Config /temporary/strategies.json: invalid metadata"


@pytest.mark.parametrize("strategy,target_suffix", [
    ("rsi_bollinger", "rsi-bollinger"),
    ("supertrend", "supertrend"),
    ("wyckoff", "wyckoff"),
])
@pytest.mark.parametrize("target_prefix,module", [("run", "strategy"), ("backtest", "backtest")])
def test_named_make_targets_select_their_strategy_explicitly(strategy, target_suffix, target_prefix, module):
    result = subprocess.run(
        ["make", "--no-print-directory", "-n", f"{target_prefix}-{target_suffix}",
         "PYTHON=/test/python", "ARGS=--ticker-file example.csv"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert shlex.split(result.stdout) == [
        "/test/python", "-m", f"src.stock_trade_analyser.{module}",
        "--strategy", strategy, "--ticker-file", "example.csv",
    ]
