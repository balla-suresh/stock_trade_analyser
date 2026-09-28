# Stock Trade Analyser

A comprehensive stock market analysis tool that implements various trading strategies and machine learning models for stock prediction.

## Prerequisites
* Python 3.12+ (required for pandas-ta and TensorFlow compatibility)
* Internet access to download stock data
* For Apple Silicon Macs: TensorFlow with Metal acceleration support
* `virtualenvwrapper` installed and configured (for `mkvirtualenv` and `workon` commands)

## Setup

### 1. Create Virtual Environment and Install Dependencies
```shell
make setup
```

**Note for Apple Silicon Mac users:** If you encounter TensorFlow installation issues with Python 3.13, consider using Python 3.12 for better compatibility. The `Makefile` uses `python3.12` by default.

## Features

### Trading Strategies
The project includes several technical analysis strategies:

* **Heikin Ashi + Supertrend**: Heikin Ashi candlestick smoothing combined with the Supertrend trend-following indicator.
```shell
make run-heikin-ashi
```

* **Fibonacci Bollinger Bands (FBB)**: Bollinger Bands plotted at Fibonacci retracement levels, used to flag buy/sell/partial signals.
```shell
make run-fbb
```

* **Support / Resistance**: Detects key support and resistance levels per stock.
```shell
make run-support-resistance
```

* **Stock Predictor**: LSTM-based next-day price forecaster (runs each ticker in parallel via `multiprocessing`).
```shell
make run-stock-predictor
```

* **Machine Learning Models**: LSTM, GRU, and MLP for stock prediction
```shell
make run-machine-learning
```

* **Seasonal**: For each stock, computes the % price increase of each calendar quarter (Q1..Q4) in every historical year, averages those per-quarter returns across years (excluding the last bar’s calendar quarter so an in-progress quarter does not bias the averages), then ranks the four quarters 1..4 where **1 = worst** and **4 = best**. The output CSV lists all four quarter ratings plus `current_quarter_rating`, sorted ascending on that column. Set `seasonal.clean_output` to `true` in `config/day.json` if you want the same directory clean as other modules. Results go to `predictions/day/seasonal.csv`.
```shell
make run-seasonal
```

### Selecting a Ticker Universe (`--ticker-file`)

Every module under `src/stock_trade_analyser/modules/` accepts a `--ticker-file` flag so you can target a specific subset of stocks (Nifty 50, a single sector, an ad-hoc list, etc.) without editing code. You can pass arguments to the `make` commands using the `ARGS` variable.

The value accepts three shapes:

| Value | Resolved to |
|---|---|
| _omitted_ | `data/tickers.csv` (full universe, default) |
| Bare filename, e.g. `tickers_50.csv` | `data/tickers_50.csv` |
| Repo-relative path, e.g. `data/tickers_power.csv` | `data/tickers_power.csv` |
| Absolute path, e.g. `/abs/path/my.csv` | as-is |

A missing file raises a clear `FileNotFoundError` up front, and each module logs the resolved path at startup so you can verify which universe a run used.

Examples:
```shell
# Default — runs against the full ~480-symbol list in data/tickers.csv
make run-fbb

# Run FBB against the Nifty 50 only
make run-fbb ARGS="--ticker-file tickers_50.csv"

# Run Heikin Ashi + Supertrend on just the IT sector
make run-heikin-ashi ARGS="--ticker-file tickers_information_technology.csv"

# Run the seasonal pipeline on a custom CSV anywhere on disk
make run-seasonal ARGS="--ticker-file /tmp/my_watchlist.csv"
```

### Ticker Files

Pre-built ticker subsets live in `data/`. Each file is a one-symbol-per-line CSV using Yahoo Finance's `.NS` suffix for NSE-listed stocks, ready to be passed straight to `--ticker-file`.

| File | Description |
|---|---|
| `tickers.csv` | Full universe (~480 NSE stocks + a couple of US tickers and the `^NSEI` index) |
| `tickers_50.csv` | Current **Nifty 50** constituents |
| `tickers_<sector>.csv` | Sector subsets (see below) |

Sector files are generated from `tickers.csv` using NSE Indices' official 4-tier industry classification, merged from `ind_niftytotalmarket_list.csv`, Nifty 500, Microcap 250, Smallcap 250 and Midcap 150 (with manual fall-throughs for the long tail). Each NSE-listed symbol in `tickers.csv` belongs to exactly one sector file.

Available sector files (counts in parentheses):

| Sector file | # |   | Sector file | # |
|---|---:|---|---|---:|
| `tickers_financial_services.csv` | 81 |   | `tickers_services.csv` | 17 |
| `tickers_capital_goods.csv` | 46 |   | `tickers_oil_gas_and_consumable_fuels.csv` | 16 |
| `tickers_healthcare.csv` | 43 |   | `tickers_metals_and_mining.csv` | 13 |
| `tickers_chemicals.csv` | 42 |   | `tickers_construction_materials.csv` | 13 |
| `tickers_consumer_durables.csv` | 31 |   | `tickers_construction.csv` | 13 |
| `tickers_automobile_and_auto_components.csv` | 31 |   | `tickers_realty.csv` | 12 |
| `tickers_fast_moving_consumer_goods.csv` | 30 |   | `tickers_power.csv` | 11 |
| `tickers_consumer_services.csv` | 28 |   | `tickers_telecommunication.csv` | 10 |
| `tickers_information_technology.csv` | 24 |   | `tickers_textiles.csv` | 9 |
| `tickers_media_entertainment_and_publication.csv` | 6 |   | `tickers_diversified.csv` | 3 |
| `tickers_forest_materials.csv` | 2 |   |  |  |

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
stock_trade_analyser/
├── data/                          # Ticker CSVs (full universe + Nifty 50 + per-sector)
└── src/stock_trade_analyser/
    ├── config/                    # Configuration files (day.json, intraday.json)
    ├── models/                    # ML model definitions
    ├── modules/                   # Core trading modules (each is a `python -m` entry point)
    └── tools/                     # Utility functions (FileUtils, Downloader, logging)
```
