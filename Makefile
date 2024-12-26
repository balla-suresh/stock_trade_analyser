.ONESHELL:
PYTHON = python3

.PHONY = help setup test run clean

help:
	@echo "---------------HELP-----------------"

setup:
	@echo "Checking if project files are generated..."
	mkdir -p output/day output/intraday
	mkdir -p predictions/day predictions/intraday predictions/result
	mkdir -p logs
	${PYTHON} -m pip install -r requirements/base.txt

test:
	python3 -m src.main.heikin_ashi_supertrend

venv:
	${PYTHON} -m pip install --upgrade pip
	${PYTHON} -m venv trade
	. trade/bin/activate

run: setup
	python3 -m src.main.heikin_ashi_supertrend

clean:
	rm -rf output predictions logs

deepclean: clean
	${PYTHON} -m pip uninstall -r requirements/base.txt
	find . -name '*.pyc' -delete

.PHONY: clean setup run

