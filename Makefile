PYTHON_VERSION ?= python3.12
# Respect an explicit interpreter or active environment, then use setup's trade
# environment when available. A bare system python3 may not have dependencies.
TRADE_PYTHON := $(if $(WORKON_HOME),$(WORKON_HOME),$(HOME)/.virtualenvs)/trade/bin/python
PYTHON ?= $(if $(VIRTUAL_ENV),$(VIRTUAL_ENV)/bin/python,$(if $(wildcard $(TRADE_PYTHON)),$(TRADE_PYTHON),python3))
ARGS ?=

.PHONY: setup clean distclean run-rsi-bollinger backtest-rsi-bollinger run-supertrend backtest-supertrend run-wyckoff backtest-wyckoff run-sma-cash-exit backtest-sma-cash-exit test

setup:
	@echo "Creating virtual environment 'trade' using $(PYTHON_VERSION)..."
	@zsh -i -c "mkvirtualenv -p $(PYTHON_VERSION) trade && pip install -r requirements/base.txt"
	mkdir -p output/day output/intraday
	mkdir -p predictions/day predictions/intraday predictions/result
	mkdir -p logs backtests

clean:
	@echo "Cleaning up python cache files..."
	rm -rf output predictions logs backtests
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info

run-rsi-bollinger:
	@"$(PYTHON)" -m src.stock_trade_analyser.strategy --strategy rsi_bollinger $(ARGS)

backtest-rsi-bollinger:
	@"$(PYTHON)" -m src.stock_trade_analyser.backtest --strategy rsi_bollinger $(ARGS)

run-supertrend:
	@"$(PYTHON)" -m src.stock_trade_analyser.strategy --strategy supertrend $(ARGS)

backtest-supertrend:
	@"$(PYTHON)" -m src.stock_trade_analyser.backtest --strategy supertrend $(ARGS)

run-wyckoff:
	@"$(PYTHON)" -m src.stock_trade_analyser.strategy --strategy wyckoff $(ARGS)

backtest-wyckoff:
	@"$(PYTHON)" -m src.stock_trade_analyser.backtest --strategy wyckoff $(ARGS)

run-sma-cash-exit:
	@"$(PYTHON)" -m src.stock_trade_analyser.strategy --strategy sma_cash_exit $(ARGS)

backtest-sma-cash-exit:
	@"$(PYTHON)" -m src.stock_trade_analyser.backtest --strategy sma_cash_exit $(ARGS)

test:
	"$(PYTHON)" -m pytest

distclean:
	@echo "Removing virtual environment 'trade' and cleaning up python cache files..."
	@zsh -i -c "rmvirtualenv trade"
	rm -rf output predictions logs backtests
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info
