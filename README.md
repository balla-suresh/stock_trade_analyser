# Stock trade analyser

Prerequisites
-------------
* Python 3.6+
* Internet access to download data

Setup
-----
To create a virtual environment to work, run the following command
```shell
make venv
```
To install required packages, run below command.
```shell
pip install -f requirements/base.txt
```

Strategies
----------
* heikin ashi
```shell
python3 -m src.main.test_heikin_ashi 
```
* supertrend
```shell
python3 -m src.main.test_supertrend
```
* heikin ashi and supertrend
```shell
python -m src.main.heikin_ashi_supertrend
```
* stock predictor using LSTM
```shell
python3 -m src.main.stock_predictor
```
* stock prediction using LSTM or GRU or MLP
```shell
python3 -m src.main.machine_learning
```


