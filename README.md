# Stock Trade Analyser

A comprehensive stock market analysis tool that implements various trading strategies and machine learning models for stock prediction.

## Prerequisites
* Python 3.11+ (recommended for TensorFlow compatibility)
* Internet access to download stock data
* For Apple Silicon Macs: TensorFlow with Metal acceleration support

## Setup

### 1. Create Virtual Environment
```shell
make venv
```

### 2. Install Dependencies
```shell
pip install -r requirements/base.txt
```

**Note for Apple Silicon Mac users:** If you encounter TensorFlow installation issues with Python 3.13, consider using Python 3.11 or 3.12 for better compatibility.

## Features

### Trading Strategies
The project includes several technical analysis strategies:

* **Heikin Ashi**: Candlestick charting technique
```shell
python3 -m src.stock_trade_analyser.modules.heikin_ashi_supertrend
```

* **Supertrend**: Trend-following indicator
```shell
python3 -m src.stock_trade_analyser.modules.stock_predictor
```

* **Machine Learning Models**: LSTM, GRU, and MLP for stock prediction
```shell
python3 -m src.stock_trade_analyser.modules.machine_learning
```

* **Seasonal**: For each stock, computes the % price increase of each calendar quarter (Q1..Q4) in every historical year, averages those per-quarter returns across years (excluding the last bar’s calendar quarter so an in-progress quarter does not bias the averages), then ranks the four quarters 1..4 where **1 = worst** and **4 = best**. The output CSV lists all four quarter ratings plus `current_quarter_rating`, sorted ascending on that column. Set `seasonal.clean_output` to `true` in `config/day.json` if you want the same directory clean as other modules. Results go to `predictions/day/seasonal.csv`.
```shell
python3 -m src.stock_trade_analyser.modules.seasonal
```

### Machine Learning
The project leverages TensorFlow for deep learning models including:
- Long Short-Term Memory (LSTM) networks
- Gated Recurrent Units (GRU)
- Multi-Layer Perceptrons (MLP)

### Data Sources
- Yahoo Finance (`yfinance`)
- TradingView data feeds
- Technical analysis indicators via `pandas-ta`

## Project Structure
```
src/stock_trade_analyser/
├── config/          # Configuration files
├── models/          # ML model definitions
├── modules/         # Core trading modules
└── tools/           # Utility functions
```


