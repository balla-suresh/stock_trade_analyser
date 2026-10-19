# Stock Trade Analyser

Python stock data utilities with long-only adaptations of ChartArt's
**Bollinger + RSI, Double Strategy v1.1**, KivancOzbilgic's
**SuperTrend STRATEGY**, Kingshuk Ghosh's **Wyckoff Schematic**, and adolgov's
**SMA crossover with cash SL/TP**, with scanners and historical backtests.

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
prices. Column names may use either case. Wyckoff requires volume; volume is
optional for RSI/Bollinger and SuperTrend. Prices must be positive and finite,
and timestamps must be unique. Rows are sorted by date.

`make clean` removes generated output, predictions, logs, backtests and Python
caches. `make distclean` also removes the `trade` virtual environment.

## Strategy configuration

The CLI uses `src/stock_trade_analyser/config/strategies.json` to select a
strategy, generate its arguments and choose its reports. Each entry in
`strategies` defines its implementation, label, parameter names/types, scan
fields, summary filenames and entry/exit descriptions. `default_strategy`
selects the strategy for module commands without `--strategy`. Named Makefile
targets select their named strategy explicitly.

Indicator values remain in each strategy's section of `day.json` or
`intraday.json`. Only the selected strategy's flags are accepted and shown in
help. For example, SuperTrend exposes `--atr-period` and `--no-change-atr`,
while Wyckoff exposes `--lookback`, `--buy-events` and overrides for every other
Wyckoff parameter. Argument types in `strategies.json` are `int`, `float`,
`str`, `bool` and `str_list`.

Backtest account defaults live in `strategies.json` under `backtest_defaults`.
An individual strategy's `backtest` object in that file can override them.
For daily/intraday-specific settings, add a `backtest` object inside the
existing strategy section of `day.json` or `intraday.json`. For example, add
this property to the existing `supertrend` object:

```json
"backtest": {
    "initial_capital": 100000,
    "position_sizing": "fixed",
    "quantity": 10,
    "commission_bps": 10,
    "slippage_bps": 5,
    "close_at_end": true
}
```

This changes only SuperTrend in that daily/intraday config. Effective backtest
settings are merged in this order: shared defaults, strategy registry
overrides, daily/intraday strategy overrides, then explicit CLI arguments.
Use `--keep-open` or `--close-at-end` to override final-position handling for
one run. Existing commands and default report filenames are preserved.

Default position sizing is `capital`, with `capital_allocation_pct` set to
100. Each buy uses the available cash at its actual execution price, including
entry commission and slippage. To reserve cash, set a lower allocation percentage
inside the strategy's `backtest` object or pass `--capital-allocation-pct 50`.
Use `position_sizing: "fixed"` with `quantity` for a fixed share count. A
strategy-specific `quantity` setting or explicit `--quantity` also selects fixed
sizing when that override does not specify a sizing mode. Explicit
`--quantity` and `--position-sizing capital` cannot be combined.

Strategies own their minimum history requirements and data diagnostics. To
add another implementation, implement the `TradingStrategy` interface,
register its metadata in `strategies.json` and add its parameter values to
the daily/intraday configs. The shared CLI needs no strategy-specific branch.

The CLI delegates shared I/O and data preparation to the existing `tools/`
modules. `downloader.py` owns Yahoo requests and cache initialization;
`file_utils.py` owns ticker-file resolution, ticker parsing and CSV/JSON reports;
`data_utils.py` owns price CSV loading, batch extraction and OHLCV validation.
All strategies use the same data validation functions. Imports stay lightweight
so help and cached-data runs do not load Yahoo, TradingView or training
dependencies unnecessarily. Strategy downloads use one threaded batch and
process the returned object in memory; the legacy `Downloader` class retains
its incremental and TradingView modes through the same provider helper.

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

Each ticker has an independent cash account starting at **100,000**. By default,
each entry buys the maximum whole number of shares affordable with **100% of
that account's current cash**, including entry costs. There is no leverage,
and commission and slippage default to zero. For example, with 100,000 cash and
a fill price of 250, it buys 400 shares before costs. Quantity is recalculated
for every entry, so realized gains or losses change the next trade's budget.

```text
budget = current_cash * capital_allocation_pct / 100
quantity = floor(budget / (fill_price * point_value * (1 + commission_bps / 10000)))
```

The fill price includes configured buy slippage. Point value is 1 for ordinary
share accounting and follows the instrument setting for cash-exit strategies.
Whole-share rounding leaves unused cash in the account. An entry is rejected
if even one share cannot be afforded. These capital and cost settings are
configurable; explicitly setting `--quantity` uses fixed sizing:

```shell
make backtest-rsi-bollinger ARGS="--ticker-file tickers_50.csv --quantity 10 --initial-capital 100000 --commission-bps 10 --slippage-bps 5"
```

Fixed-size orders are rejected if cash cannot cover the full quantity and entry fee. Fees
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

## Wyckoff Schematic

The supplied Pine v5 indicator **Wyckoff Schematic by Kingshuk Ghosh**,
**© kingshukghosh71**, is adapted in
`src/stock_trade_analyser/strategy/wyckoff.py`, retaining its
[Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/) notice.

The source detects events and phases without defining trade orders. This
adaptation adds the following configurable long-only rules:

| Signal | Default events | Action |
|---|---|---|
| BUY | Spring, Sign of Strength (SOS), Last Point of Support (LPS) | Open a long at the next candle's open when flat |
| EXIT | Buying Climax (BC), Upthrust After Distribution (UTAD), Last Point of Supply (LPSY) | Close an existing long at the next candle's open |

If both sets match on the same candle, EXIT takes priority. Repeated buys do
not add to an existing position, and exits while flat do not open shorts.
There is no protective stop or profit target in these rules. Account size,
quantity, fees, slippage, final-data liquidation and `--keep-open` follow the
shared backtest behavior described above.

All 14 source events are retained: PS, SC, AR, ST, SPRING, TEST, SOS, LPS, PSY,
BC, AR_DIST, ST_DIST, UTAD and LPSY. The accumulation and distribution phase
updates run in the source order and appear in the reports. Trades use the raw
event conditions, independently of the displayed phase and label spacing.
Chart labels, boxes, zigzag and miniature daily wave drawings do not affect
these conditions and are not used for trading. The source's `autoDetect` input
is unused in its detection logic.

For example, Spring requires a low below the previous `sweep_period` lows,
a bullish candle, volume above its average, and a close more than
`spring_recovery_pct` above that candle's low. It does not additionally require
the close to regain the previous support. SOS requires a rising close, volume
above `sos_volume_multiplier` times average volume, a bullish body larger than
average range, and a close above the range midpoint. LPS requires a higher low
than the latest Spring/ST, a bullish candle, below-average volume and a close
above the midpoint. These preserve the supplied indicator's conditions.

The `wyckoff` sections in both daily and intraday configs hold all detection
settings and the event-to-order mapping:

```json
"wyckoff": {
    "lookback": 100,
    "volume_multiplier": 1.5,
    "climax_volume_addition": 0.5,
    "range_period": 20,
    "narrow_range_multiplier": 0.6,
    "preliminary_period": 50,
    "climax_period": 30,
    "sweep_period": 20,
    "rally_change_pct": 0.01,
    "secondary_test_tolerance_pct": 0.005,
    "spring_recovery_pct": 0.01,
    "test_range_tolerance": 0.02,
    "test_volume_multiplier": 0.8,
    "sos_volume_multiplier": 1.3,
    "buy_events": ["SPRING", "SOS", "LPS"],
    "exit_events": ["BC", "UTAD", "LPSY"]
}
```

Percent settings use fractions: `0.01` means 1%, and `0.005` means 0.5%.
`lookback` sets the volume SMA and support/resistance windows;
`range_period` sets the candle-range SMA. `preliminary_period` and
`climax_period` are the extreme-price windows for preliminary/climax events
and the automatic rally/reaction; `sweep_period` is the prior-bar window used
by Spring and UTAD. Climax volume is greater than average volume times
`volume_multiplier + climax_volume_addition`. The remaining multipliers
control narrow candles, secondary tests, low-volume tests and SOS.

Run the scanner and then the historical backtest:

```shell
make run-wyckoff
make backtest-wyckoff
```

Both commands use the configured ticker file, download period and interval,
and fetch all tickers in one batch before processing the returned data.
Use `ARGS="--no-download"` to reuse cached prices. Optional `--ticker-file`,
`--lookback`, `--volume-multiplier`, `--buy-events` and `--exit-events` arguments
override their configured values. For example, to enter only on Spring:

```shell
make backtest-wyckoff ARGS="--ticker-file tickers_50.csv --buy-events SPRING"
```

The scanner writes only `predictions/wyckoff.csv`, with columns
`ticker,signal,close,phase,pattern,events,support,resistance,buy_signals,exit_signals,message`.
`events` lists all raw events on the latest candle, including events that do
not trigger an order. Signal counts cover the supplied history; backtest
buy/sell counts cover executed fills.

The backtest summary is `predictions/wyckoff_backtest.csv`, with starting and
ending capital, executed buys/sells, completed trades, wins/losses, profit,
return, win rate and maximum drawdown. Detailed reports live in
`backtests/wyckoff/day/` or `backtests/wyckoff/intraday/`: `config.json` records
the effective settings, and each ticker has `signals.csv`, `orders.csv`,
`trades.csv`, `equity.csv` and `metrics.json`. Signals include every raw event,
volume/range averages and the phase/pattern history.

Volume must be present, finite and nonnegative. An instrument with zero volume
throughout its history produces no events or trades, with an explanation in
the summary's `message` column. This can occur with index feeds. These are
backtests of the added order rules; the source indicator itself has no
TradingView strategy results to reproduce.

## SMA crossover with cash exits

The supplied Pine v4 example **Stop loss and Take Profit in $$ example**,
**© adolgov**, is implemented in `strategy/sma_cash_exit.py` with its
[Mozilla Public License 2.0](https://mozilla.org/MPL/2.0/) notice. The existing
long-only requirement applies: an upward fast/slow SMA crossover opens a long,
and a downward crossover closes it. A stop-loss or take-profit can also close
the position. An exit never opens a short, and there is no pyramiding.

Both daily and intraday configs contain:

```json
"sma_cash_exit": {
    "fast_period": 14,
    "slow_period": 28,
    "take_profit_cash": 100.0,
    "stop_loss_cash": 20.0,
    "point_value": 1.0,
    "tick_size": 0.01
}
```

All six settings have CLI overrides generated from the strategy registry.
The account's share quantity, initial capital and costs use the same configurable
`backtest` settings as the other strategies.

Capital-based sizing changes the number of shares; the source script's cash
targets remain fixed for the entire position. A 100 cash take-profit still
targets 100 gross profit regardless of how much capital is invested. Its price
distance becomes smaller as quantity increases. The entry-bar protection delay
described below still applies.

The cash amounts apply to the **whole position**, before commissions and
slippage, in the instrument's quote currency. They are not percentages and
do not always mean US dollars. Price distances are calculated from the actual
entry fill and executed quantity:

```text
take_profit_price = entry_price + take_profit_cash / (point_value * quantity)
stop_loss_price   = entry_price - stop_loss_cash / (point_value * quantity)
profit_ticks     = take_profit_cash / (point_value * quantity * tick_size)
loss_ticks       = stop_loss_cash / (point_value * quantity * tick_size)
```

For 10 shares with point value 1, the defaults put the target 10 price units
above the entry and the stop 2 units below it. The original formula accepts
fractional tick distances; this implementation preserves them without rounding
prices to a broker's tick grid. Set point value and tick size for the instrument.
Point value also scales the full cash cost, proceeds, equity and profit; the
backtest requires enough capital for the full position value.

The script's default execution settings matter: crossover orders fill at the
next open, but the position-dependent exit orders are first created at the
**entry candle's close**. They can fill from the following candle. Earlier
entry-candle highs/lows do not trigger those orders. A stop already crossed at
creation becomes a pending stop exit for the following open, even if that open
recovers above the stop. See [TradingView's explanation of entry-dependent exits](https://www.tradingview.com/pine-script-docs/faq/strategies/#how-can-i-set-stop-loss-and-take-profit-levels-as-a-percentage-from-my-entry-point).

Once active, a bracket fills at the open when price gaps through a level.
Otherwise the backtest follows open-high-low-close when the open is nearer
the high, and open-low-high-close when nearer the low. If both levels are
touched, the first on that path wins. Equal distances use the low first as an
explicit tie convention. Pending market exits execute before bracket checks at
the same open. These assumptions follow the [TradingView broker emulator](https://www.tradingview.com/pine-script-docs/concepts/strategies/#broker-emulator).

Stops and market exits include configured adverse slippage. Take-profit limits
fill at their limit or better without adverse slippage; both fills incur
commission. Gaps, fees and slippage can make realized net profit differ from
the configured cash amounts. Final-data liquidation and `--keep-open` retain
their existing behavior.

```shell
make run-sma-cash-exit
make backtest-sma-cash-exit
make backtest-sma-cash-exit ARGS="--quantity 10 --take-profit-cash 100 --stop-loss-cash 20"
```

The scanner writes only `predictions/sma_cash_exit.csv`, with the latest
BUY/EXIT/HOLD, close, fast/slow SMA values and historical crossover counts.
Its `exit_signals` counts downward SMA crossings. The backtest's executed sell
count also includes protective exits and final-data liquidation.

The separate backtest summary is `predictions/sma_cash_exit_backtest.csv`.
Detailed reports live under `backtests/sma_cash_exit/day/` or `intraday/`.
`trades.csv` identifies `STOP_LOSS`, `TAKE_PROFIT`, `EXIT_SIGNAL` or `END_OF_DATA`;
`equity.csv` includes the position's stop/target levels and activation state.

## Ticker files

The default ticker list is configured by the top-level `ticker_file` setting
in `src/stock_trade_analyser/config/day.json` and `intraday.json`:

```json
"ticker_file": "data/tickers_test.csv"
```

When `--ticker-file` is omitted, all scan and backtest commands use that value
from the selected daily or intraday config. Change it to choose another default,
or pass `--ticker-file tickers.csv` to override it for one run. Bare filenames
resolve under `data/`, repo-relative paths resolve from the repository root,
and absolute paths are used directly.

Pre-built ticker subsets live in `data/`. Each file is a one-symbol-per-line CSV using Yahoo Finance's `.NS` suffix for NSE-listed stocks.

| File | Description |
|---|---|
| `tickers_test.csv` | Default test ticker list |
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
├── predictions/wyckoff.csv          # Wyckoff scanner summary (generated)
├── predictions/wyckoff_backtest.csv # Wyckoff backtest summary (generated)
├── predictions/sma_cash_exit.csv    # SMA cash-exit scanner summary (generated)
├── predictions/sma_cash_exit_backtest.csv # SMA cash-exit backtest summary (generated)
├── backtests/rsi_bollinger/         # Backtest reports (generated)
├── backtests/supertrend/            # SuperTrend backtest reports (generated)
├── backtests/wyckoff/               # Wyckoff backtest reports (generated)
├── backtests/sma_cash_exit/         # SMA cash-exit backtest reports (generated)
├── tests/                          # Offline indicator, execution and CLI checks
└── src/stock_trade_analyser/
    ├── strategy/                   # Indicator implementations and shared CLI/registry
    ├── backtest/                   # Long-only execution and backtest runner
    ├── config/                     # Strategy registry, run settings and legacy configuration
    └── tools/                      # Downloader, file, data and logging helpers
```
