# Stock Trade Analyser

Python stock data utilities with long-only adaptations of ChartArt's
**Bollinger + RSI, Double Strategy v1.1** and KivancOzbilgic's
**SuperTrend STRATEGY**, with scanners and historical backtests.

## Setup

Use Python 3.12 or newer. With `virtualenvwrapper` configured, run:

```shell
make setup
workon trade
```

`make setup` creates the `trade` virtual environment, installs
`requirements/base.txt` and creates the data/report directories. To install the
package into an existing environment instead:

```shell
python -m pip install -e .
```

The `make` run and test commands use an active virtual environment, or the
existing `trade` environment under `$WORKON_HOME` (default `~/.virtualenvs`).
You can choose an interpreter explicitly with `make PYTHON=/path/to/python ...`.
Direct `python3 -m ...` commands use the Python selected by your shell; run
`workon trade` first to ensure pandas and numpy are available.

Cached daily and intraday OHLCV CSVs live in `output/day/` and `output/intraday/`.
Each CSV needs a date column (or saved date index) and open, high, low and close
prices. Column names may use either case; volume is optional. Prices must be
positive and finite, and timestamps must be unique. Rows are sorted by date.

`make clean` removes generated output, predictions, logs, backtests and Python
caches. `make distclean` also removes the `trade` virtual environment.

## RSI + Bollinger strategy

The supplied ChartArt Pine script is adapted with **long entries only**. Its
original short-entry condition closes an existing long and never opens a short.

Strategy settings are read from the `rsi_bollinger` section of
`src/stock_trade_analyser/config/day.json`, or `intraday.json` when using
`--data-type intraday`:

```json
"rsi_bollinger": {
    "rsi_period": 6,
    "bb_period": 200,
    "bb_multiplier": 2.0,
    "rsi_threshold": 50.0
}
```

Edit these values to configure both scanning and backtesting. Optional
`--rsi-period`, `--bb-period`, `--bb-multiplier` and `--rsi-threshold` arguments
override the selected file's values for one run. All four settings are required
in the config; the strategy constructor takes them explicitly.

| Setting | Default |
|---|---:|
| RSI period | 6 |
| RSI entry and exit threshold | 50 |
| Bollinger SMA period | 200 |
| Bollinger standard deviation multiplier | 2 |

RSI uses Wilder's smoothing, seeded by the average of the first `rsi_period`
close changes. Bands use the rolling close mean and population standard deviation
(`ddof=0`). The strategy requires both crossings on the **same completed candle**:

- **BUY:** RSI crosses above the configured threshold and close crosses above the lower band. Each
  previous value must be at or below its corresponding threshold or band.
- **EXIT:** RSI crosses below the configured threshold and close crosses below the upper band. Each
  previous value must be at or above its corresponding threshold or band.
- **HOLD:** neither event occurs. An EXIT while flat is ignored, and repeated
  buys while holding a position do not add shares.

The bundled configs preserve the script's threshold of 50; merely being below a band or
having a low RSI does not generate a buy. This implementation represents a
completely flat RSI seed as undefined, so it does not generate crossings; that
edge case has not been compared with live TradingView output. At least 201 candles are needed to
compare two fully formed default bands. Long periods with no simultaneous
crossings can legitimately produce no trades.

Scan using the daily configuration:

```shell
make run-rsi-bollinger ARGS="--ticker-file tickers_50.csv"
```

Both scanner and backtest read download settings from
`src/stock_trade_analyser/config/day.json`. The current daily defaults enable
downloading daily Yahoo Finance history. `download.is_download` controls
downloading, `download.period` sets the history length, and `download.interval`
sets the candle interval.
With downloading enabled, the command downloads the full ticker list in one
threaded batch using those settings. It then takes each ticker's prices from
the returned in-memory object and runs the strategy and backtest directly.
It also saves validated prices under `output/day/` for later cached runs;
the downloaded data is not read back from CSV during the current run. No
download flag is needed. Symbols missing from a partial batch are reported
individually in the command's summary CSV while the remaining symbols continue, without
additional downloads. Commands print no routine progress or status messages.
Results and per-ticker error details are saved in the report files.

`--data-type intraday` selects `src/stock_trade_analyser/config/intraday.json`
and the `output/intraday/` cache. Edit the selected config to change the defaults.
Optional `--period` and `--interval` arguments override its values for one run;
`--download` or `--no-download` overrides its download switch. For cached data
only, use `--no-download` or set `download.is_download` to `false` in the config.

The scanner writes only `predictions/rsi_bolinger.csv`. Indicators and signal
history are calculated in memory, without intermediate signal files or per-ticker
report directories. The summary's signal describes the latest completed candle; it is an event, not a
record of a live brokerage position. Nothing places real orders.

Scanner CSV columns: `ticker,signal,close,rsi,bb_lower,bb_upper,buy_signals,exit_signals,message`.

## Backtest

```shell
make backtest-rsi-bollinger ARGS="--ticker-file tickers_50.csv"
```

This downloads the whole ticker list using the daily config before backtesting,
including when cached CSVs do not exist yet. Use `--data-dir PATH` for another
CSV directory. The final summary is `predictions/rsi_bolinger_backtest.csv`; detailed
backtest reports are under `backtests/rsi_bollinger/day/`. `--output-dir PATH`
overrides the directory for both the final `rsi_bolinger_backtest.csv` and detailed reports.

The backtest summary shows the input capital and output equity for each ticker,
along with executed trade counts and performance:

| Columns | Meaning |
|---|---|
| `initial_capital`, `final_equity` | Starting account value and ending cash plus the value of any open position |
| `total_trades` | Completed buy-and-sell round trips |
| `total_buys`, `total_sells` | Executed entry and closing orders; a final-candle liquidation counts as a sell |
| `winning_trades`, `losing_trades`, `breakeven_trades` | Completed trades with positive, negative or zero profit after fees |
| `net_profit`, `total_return_pct` | Change in account value, in money and as a percentage of initial capital |
| `win_rate_pct` | Winning completed trades divided by all completed trades, multiplied by 100 |
| `max_drawdown_pct` | Largest percentage decline from a prior closing-equity peak |
| `message` | Missing data, insufficient history or other processing details |

Buy and sell counts include only filled orders. Skipped, rejected and unfilled
orders are excluded. With `--keep-open`, buys can exceed sells and final equity
includes unrealized profit. Historical signal counts remain in each ticker's
`metrics.json` for diagnostics.

Signals are confirmed at candle close and orders fill at the **next candle's
open**. In the source, `stop=BBlower` is an entry stop, not a protective stop-loss.
The buy signal's close is already above that stop, so the stop activates
immediately and the market order fills at the next open, including a gap below
the band. This follows [TradingView's order execution rules](https://www.tradingview.com/pine-script-docs/concepts/strategies/#stop-and-stop-limit-orders).

Each ticker has an independent cash account starting at **100,000**, with a
fixed **one-share** order size, no leverage, and zero commission or slippage by
default. These capital and cost settings are configurable:

```shell
make backtest-rsi-bollinger ARGS="--ticker-file tickers_50.csv --quantity 10 --initial-capital 100000 --commission-bps 10 --slippage-bps 5"
```

Orders are rejected if cash cannot cover the full quantity and entry fee. Fees
apply to both entry and exit value; 10 basis points means 0.1% per fill.
Slippage increases the buy price and decreases the exit price. Each closed
trade's net profit is quantity times the price change, minus both fees.

Daily equity is cash plus shares times that candle's close. Total return is
`(final_equity / initial_capital - 1) * 100`. Maximum drawdown measures the largest
percentage fall from the highest closing equity, including starting capital.
The engine does not use intrabar highs/lows to compute drawdown and does not
credit dividends. It uses the CSV's close prices; Yahoo downloads request
unadjusted OHLC. Match data and account settings when comparing with TradingView.

By default, a remaining position is liquidated at the final close, with costs,
and labeled `END_OF_DATA`. This is a reporting convention, separate from the
strategy's exit signal. Pass `--keep-open` to leave it open and include its
unrealized profit in final equity. Last-candle signals cannot fill without a
following candle; they appear in `orders.csv` as `NO_NEXT_BAR`.

The backtest writes:

| File | Purpose |
|---|---|
| `predictions/rsi_bolinger_backtest.csv` | Starting/ending capital, executed buys/sells, completed trades, wins/losses, profit, return, win rate and drawdown |
| `config.json` | Indicator, account and resolved download settings, including the source config path |
| `SYMBOL/signals.csv` | Prices, RSI, bands and BUY/EXIT/HOLD events |
| `SYMBOL/trades.csv` | Completed longs with fills, fees, profit and exit reason |
| `SYMBOL/equity.csv` | Cash, quantity, equity and drawdown for every candle |
| `SYMBOL/orders.csv` | Filled, skipped, rejected and unfilled orders with reasons |
| `SYMBOL/metrics.json` | Detailed results, including open positions and unrealized profit |

The files other than the final summary are relative to the detailed report
directory. Scans update `predictions/rsi_bolinger.csv`; backtests update
`predictions/rsi_bolinger_backtest.csv`, preserving the scanner's summary.

The `message` column describes missing files, invalid data, insufficient history,
no buy signals and signals without completed trades. Win rate is blank when
there are no completed trades. Missing or malformed data retains the configured
input capital, leaves output/performance fields blank and makes the command
return a nonzero exit code.

View all options or run the offline checks:

```shell
make backtest-rsi-bollinger ARGS="--help"
make test
```

Equivalent module commands are `python -m src.stock_trade_analyser.strategy`
and `python -m src.stock_trade_analyser.backtest` from the repository root. After
installing the package, omit `src.` from those module names.

## SuperTrend

The supplied Pine v4 SuperTrend strategy by **© KivancOzbilgic** is implemented
in `src/stock_trade_analyser/strategy/supertrend.py`, retaining its
[Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/) notice.

```shell
make run-supertrend ARGS="--ticker-file tickers_50.csv"
make backtest-supertrend ARGS="--ticker-file tickers_50.csv"
```

These commands use the same config-driven batch download and in-memory
processing as RSI/Bollinger. They run quietly and write separate reports:

| Command | Final CSV |
|---|---|
| `run-supertrend` | `predictions/supertrend.csv` |
| `backtest-supertrend` | `predictions/supertrend_backtest.csv` |

The scanner creates only its final CSV. Detailed backtest files are written to
`backtests/supertrend/day/SYMBOL/`; the backtest summary includes input capital,
final equity, executed buys/sells, completed trades, wins/losses/breakevens,
profit, return, win rate and drawdown. `--output-dir` overrides the report
location, and `--no-download` uses cached prices.

Settings come from the selected daily or intraday config:

```json
"supertrend": {
    "atr_period": 10,
    "atr_multiplier": 3.0,
    "source": "hl2",
    "change_atr": true,
    "from_date": "2018-09-01",
    "to_date": "9999-01-01",
    "timezone": "Asia/Kolkata"
}
```

All seven settings are required. Optional CLI overrides are `--atr-period`,
`--atr-multiplier`, `--source`, `--change-atr` / `--no-change-atr`,
`--from-date`, `--to-date` and `--timezone`. Supported price sources are
`open`, `high`, `low`, `close`, `hl2`, `hlc3` and `ohlc4`.

`change_atr: true` uses Wilder's ATR, seeded with the average of the first
`atr_period` true ranges. `false` uses the source script's simple average of
raw true range. The first candle contributes high minus low to Wilder's ATR,
while raw true range is undefined without a previous close, so the simple
average starts one candle later. This follows
[TradingView's true-range behavior](https://www.tradingview.com/pine-script-docs/language/built-ins/).

The carried bands and trend reversals follow the supplied recurrence. A change
from downtrend to uptrend generates **BUY**; a change from uptrend to downtrend
generates **EXIT**, closing an existing long. The initial uptrend state does
not generate a buy. Positions are long-only, and orders fill at the next
candle's open. The displayed SuperTrend band is not an intrabar stop-loss.

Indicators use all available history for warmup. Buy and exit events are
allowed only inside the configured window, from `from_date` at 00:00 through
`to_date` at 23:59 in the configured timezone. Input timestamps normalized to
UTC by the shared loader are converted to that timezone for the comparison.
The year 9999 end date is supported. Passing the window's end does not itself
generate an exit; the shared final-data liquidation convention and
`--keep-open` option still apply.

The scanner columns are
`ticker,signal,close,trend,supertrend,atr,buy_signals,exit_signals,message`.
Equivalent module commands use `--strategy supertrend` with
`python -m src.stock_trade_analyser.strategy` or
`python -m src.stock_trade_analyser.backtest`.

## Ticker files

Pre-built ticker subsets live in `data/`. Each file is a one-symbol-per-line CSV using Yahoo Finance's `.NS` suffix for NSE-listed stocks.

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

## Project structure

```text
stock_trade_analyser/
├── data/                           # Ticker universes
├── requirements/                   # Python dependencies
├── output/                         # Cached OHLCV data (generated)
├── predictions/rsi_bolinger.csv     # Scanner summary (generated)
├── predictions/rsi_bolinger_backtest.csv # Backtest summary (generated)
├── predictions/supertrend.csv       # SuperTrend scanner summary (generated)
├── predictions/supertrend_backtest.csv # SuperTrend backtest summary (generated)
├── backtests/rsi_bollinger/         # Backtest reports (generated)
├── backtests/supertrend/            # SuperTrend backtest reports (generated)
├── tests/                          # Offline indicator, execution and CLI checks
└── src/stock_trade_analyser/
    ├── strategy/                   # RSI/Bollinger and SuperTrend indicators and scanner
    ├── backtest/                   # Long-only execution and backtest runner
    ├── config/                     # Data and legacy experiment configuration
    └── tools/                      # Downloader, file, data and logging helpers
```
