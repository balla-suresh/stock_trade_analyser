"""Load strategy metadata without importing numerical or download libraries."""

from __future__ import annotations

import argparse
import importlib
import json
import keyword
import math
from dataclasses import dataclass
from pathlib import Path
from string import Formatter
from typing import Any


_PARAMETER_TYPES = {"int": int, "float": float, "str": str, "bool": bool, "str_list": list}
_BACKTEST_TYPES = {
    "initial_capital": float,
    "quantity": int,
    "position_sizing": str,
    "capital_allocation_pct": float,
    "commission_bps": float,
    "slippage_bps": float,
    "close_at_end": bool,
}
_SCAN_GENERIC_FIELDS = {"ticker", "signal", "close", "buy_signals", "exit_signals", "message"}
_CLI_RESERVED_NAMES = {
    "help", "h", "strategy", "ticker_file", "data_type", "data_dir", "output_dir", "report_dir",
    "download", "period", "interval", "keep_open", *_BACKTEST_TYPES,
}


@dataclass(frozen=True)
class StrategyDefinition:
    """Configurable presentation, parameter schema, and lazy implementation."""

    name: str
    label: str
    report_label: str
    implementation: str
    parameters: dict[str, str]
    scan_filename: str
    backtest_filename: str
    scan_fields: tuple[str, ...]
    entry_description: str
    exit_description: str
    no_buys_message: str
    minimum_description: str
    backtest: dict[str, Any]

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return tuple(self.parameters)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        """Expose only the selected strategy's parameters as CLI overrides."""
        for name, kind in self.parameters.items():
            options: dict[str, Any] = {
                "default": None,
                "help": f"Defaults to {self.name}.{name} in the selected config.",
            }
            if kind == "bool":
                options["action"] = argparse.BooleanOptionalAction
            elif kind == "str_list":
                options.update(type=str, nargs="+", metavar=name.upper())
            else:
                options["type"] = _PARAMETER_TYPES[kind]
            parser.add_argument("--" + name.replace("_", "-"), **options)

    def create(self, parameters: dict[str, Any]) -> Any:
        """Import the selected strategy only when an execution needs it."""
        module_name, class_name = self.implementation.split(":")
        try:
            implementation = getattr(importlib.import_module(module_name, package=__package__), class_name)
        except (ImportError, AttributeError) as error:
            raise ValueError(f"Cannot load strategy {self.name} ({self.implementation}): {error}") from error
        return implementation(**parameters)

    def describe(self, template: str, strategy: Any) -> str:
        """Fill metadata templates with validated strategy parameters."""
        values = {}
        for _, name, _, _ in Formatter().parse(template):
            if name is not None:
                value = getattr(strategy, name)
                values[name] = ", ".join(str(item) for item in value) if isinstance(value, (list, tuple)) else value
        return template.format_map(values)


@dataclass(frozen=True)
class StrategyRegistry:
    default_strategy: str
    strategies: dict[str, StrategyDefinition]
    backtest_defaults: dict[str, Any]
    backtest_columns: tuple[str, ...]


def _nonempty_string(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{location} must be a nonempty string")
    return value.strip()


def _strings(value: Any, location: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"{location} must be {'an' if allow_empty else 'a nonempty'} array of strings")
    values = tuple(_nonempty_string(item, location) for item in value)
    if len(set(values)) != len(values):
        raise ValueError(f"{location} must not contain duplicate names")
    return values


def _backtest_settings(value: Any, location: str, *, require_all: bool = False) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be an object")
    unknown = value.keys() - _BACKTEST_TYPES.keys()
    if unknown:
        raise ValueError(f"{location} contains unknown settings: {', '.join(sorted(unknown))}")
    if require_all:
        missing = _BACKTEST_TYPES.keys() - value.keys()
        if missing:
            raise ValueError(f"{location} is missing settings: {', '.join(sorted(missing))}")
    for name, setting in value.items():
        expected = _BACKTEST_TYPES[name]
        valid = type(setting) is expected
        if expected is float:
            valid = type(setting) in (int, float) and math.isfinite(setting)
        if not valid:
            expected_description = {
                float: "a finite number", int: "an integer", bool: "a boolean", str: "a string",
            }[expected]
            raise ValueError(f"{location}.{name} must be {expected_description}")
        if name == "position_sizing" and setting not in {"capital", "fixed"}:
            raise ValueError(f"{location}.{name} must be 'capital' or 'fixed'")
    return value.copy()


def _definition(name: str, settings: Any) -> StrategyDefinition:
    location = f"strategies.{name}"
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"Invalid strategy name: {name!r}")
    if not isinstance(settings, dict):
        raise ValueError(f"{location} must be an object")
    text_fields = (
        "label", "report_label", "implementation", "scan_filename", "backtest_filename",
        "entry_description", "exit_description", "no_buys_message", "minimum_description",
    )
    text = {field: _nonempty_string(settings.get(field), f"{location}.{field}") for field in text_fields}
    implementation_parts = text["implementation"].split(":")
    if len(implementation_parts) != 2 or not all(implementation_parts):
        raise ValueError(f"{location}.implementation must be module:ClassName")
    for field in ("scan_filename", "backtest_filename"):
        filename = text[field]
        if "/" in filename or "\\" in filename or Path(filename).name != filename or not filename.endswith(".csv"):
            raise ValueError(f"{location}.{field} must be a simple CSV filename")
    if text["scan_filename"] == text["backtest_filename"]:
        raise ValueError(f"{location} scan and backtest filenames must differ")
    parameters = settings.get("parameters")
    if not isinstance(parameters, dict) or not parameters:
        raise ValueError(f"{location}.parameters must be a nonempty object")
    for parameter, kind in parameters.items():
        if not parameter.isidentifier() or keyword.iskeyword(parameter) or parameter in _CLI_RESERVED_NAMES:
            raise ValueError(f"{location}.parameters contains invalid or reserved name: {parameter!r}")
        if not isinstance(kind, str) or kind not in _PARAMETER_TYPES:
            raise ValueError(f"{location}.parameters.{parameter} has unknown type: {kind!r}")
    scan_fields = _strings(settings.get("scan_fields"), f"{location}.scan_fields", allow_empty=True)
    clashes = set(scan_fields) & _SCAN_GENERIC_FIELDS
    if clashes:
        raise ValueError(f"{location}.scan_fields repeats generic fields: {', '.join(sorted(clashes))}")
    for field in ("entry_description", "exit_description", "no_buys_message", "minimum_description"):
        try:
            placeholders = [part[1] for part in Formatter().parse(text[field]) if part[1] is not None]
        except ValueError as error:
            raise ValueError(f"{location}.{field} has an invalid template: {error}") from error
        if any(placeholder not in parameters for placeholder in placeholders):
            raise ValueError(f"{location}.{field} placeholders must name configured parameters")
    return StrategyDefinition(
        name=name, parameters=parameters.copy(), scan_fields=scan_fields,
        backtest=_backtest_settings(settings.get("backtest", {}), f"{location}.backtest"),
        **text,
    )


def load_registry(path: str | Path | None = None) -> StrategyRegistry:
    """Read registry metadata from the package, independently of run configs."""
    path = Path(path) if path is not None else Path(__file__).resolve().parents[1] / "config" / "strategies.json"
    try:
        with path.open(encoding="utf-8") as handle:
            config = json.load(handle)
        if not isinstance(config, dict):
            raise ValueError("registry must be an object")
        raw_strategies = config.get("strategies")
        if not isinstance(raw_strategies, dict) or not raw_strategies:
            raise ValueError("strategies must be a nonempty object")
        strategies = {name: _definition(name, settings) for name, settings in raw_strategies.items()}
        report_files: dict[str, str] = {}
        for name, definition in strategies.items():
            for field in ("scan_filename", "backtest_filename"):
                filename = getattr(definition, field)
                if filename in report_files:
                    raise ValueError(
                        f"strategies.{name}.{field} repeats report filename {filename!r} "
                        f"already used by {report_files[filename]}"
                    )
                report_files[filename] = f"strategies.{name}.{field}"
        default_strategy = _nonempty_string(config.get("default_strategy"), "default_strategy")
        if default_strategy not in strategies:
            raise ValueError("default_strategy must name a configured strategy")
        backtest_defaults = _backtest_settings(config.get("backtest_defaults"), "backtest_defaults", require_all=True)
        backtest_columns = _strings(config.get("backtest_columns"), "backtest_columns")
        return StrategyRegistry(default_strategy, strategies, backtest_defaults, backtest_columns)
    except (OSError, ValueError) as error:
        raise ValueError(f"Config {path}: {error}") from error
