"""Run the long-only RSI/Bollinger backtest."""

from ..strategy.cli import main


if __name__ == "__main__":
    raise SystemExit(main(mode="backtest"))
