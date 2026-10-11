PYTHON_VERSION ?= python3.12
ARGS ?=

.PHONY: setup run-heikin-ashi run-fbb run-support-resistance run-stock-predictor run-machine-learning run-seasonal clean distclean

setup:
	@echo "Creating virtual environment 'trade' using $(PYTHON_VERSION)..."
	@zsh -i -c "mkvirtualenv -p $(PYTHON_VERSION) trade && pip install -r requirements/base.txt"
	mkdir -p output/day output/intraday
	mkdir -p predictions/day predictions/intraday predictions/result
	mkdir -p logs

run-heikin-ashi:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.heikin_ashi_supertrend $(ARGS)"

run-fbb:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.FBB $(ARGS)"

run-support-resistance:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.support_resistance $(ARGS)"

run-stock-predictor:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.stock_predictor $(ARGS)"

run-machine-learning:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.machine_learning $(ARGS)"

run-seasonal:
	@zsh -i -c "workon trade && python -m src.stock_trade_analyser.modules.seasonal $(ARGS)"

clean:
	@echo "Cleaning up python cache files..."
	rm -rf output predictions logs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info
distclean:
	@echo "Removing virtual environment 'trade' and cleaning up python cache files..."
	@zsh -i -c "rmvirtualenv trade"
	rm -rf output predictions logs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf *.egg-info