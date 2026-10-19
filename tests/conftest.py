"""Explicit execution settings for fixtures that verify fixed-quantity results."""

from copy import deepcopy

import pytest


@pytest.fixture
def fixed_position_registry(monkeypatch):
    """Keep historical one-share examples independent of CLI sizing defaults."""
    from stock_trade_analyser.strategy import cli

    registry = deepcopy(cli.REGISTRY)
    registry.backtest_defaults["position_sizing"] = "fixed"
    monkeypatch.setattr(cli, "REGISTRY", registry)
