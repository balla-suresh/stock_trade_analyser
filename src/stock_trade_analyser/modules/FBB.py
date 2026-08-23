import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import FibonacciBollingerBands
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.file_utils import FileUtils, parse_ticker_file_arg
import datetime

ticker_file_arg = parse_ticker_file_arg(
    prog="fbb",
    description="Run the Fibonacci Bollinger Bands strategy.",
)

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json'), 'r') as f:
    config = json.load(f)

logger = LoggerUtils("fbb").get_logger()
logger.info("Started FBB")

file_utils = FileUtils(
    data_type=config["download"]["data_type"],
    ticker_file=ticker_file_arg,
)
logger.info(f"Using ticker file: {file_utils.ticker_file}")
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

fbb = FibonacciBollingerBands(
    length=config["fbb"]["length"],
    multiplier=config["fbb"]["multiplier"],
    use_vwma=config["fbb"]["use_vwma"]
)

df = pd.DataFrame(ticker_list, columns=['symbol'])  # type: ignore
df = df.set_index('symbol')

rows = []

if ticker_list is not None:
    for each_ticker in ticker_list:
        if isinstance(each_ticker, dict):
            each_ticker = each_ticker['symbol']

        current_data = file_utils.import_csv(each_ticker)
        current_data = current_data.dropna()
        current_data = current_data.rename(columns=str.lower)

        # Calculate FBB
        fbb_data = fbb.setup(current_data)

        # Get signals
        strategy = fbb.get_signal(each_ticker, fbb_data)

        # Generate CSV for stock if intermediate is enabled
        if config["fbb"]["intermediate"]:
            file_utils.result_csv(strategy, sub_dir=file_utils.get_data_type(), ticker=each_ticker)

        last = strategy.iloc[-1]
        rows.append({
            'symbol': each_ticker,
            'signal': last['fbb_signal'],
            'close': last['close'],
            'band_position': last.get('band_position'),
            'pos_trend': last.get('pos_trend'),
            'pct_to_up6': last.get('pct_to_up6'),
            'pct_to_low6': last.get('pct_to_low6'),
            'bars_since_up6': last.get('bars_since_up6'),
        })

df = pd.DataFrame(rows).set_index('symbol') if rows else df

# Strongest signals first. Within the momentum tiers the most extended names
# come first (highest band_position); the sort is stable so buy tiers keep
# their natural ordering too.
df = df.sort_values(by=['signal', 'band_position'], ascending=[False, False])

# Save results
# Buys
file_utils.result_csv(df[df['signal'] == 1.0], sub_dir=file_utils.get_data_type(), ticker='fbb_buy')
file_utils.result_csv(df[df['signal'] == 0.5], sub_dir=file_utils.get_data_type(), ticker='fbb_weak_buy')
file_utils.result_csv(df[df['signal'] == 0.25], sub_dir=file_utils.get_data_type(), ticker='fbb_buy_watch')
# Momentum / upper band
file_utils.result_csv(df[df['signal'] == -0.5], sub_dir=file_utils.get_data_type(), ticker='fbb_momentum')
file_utils.result_csv(df[df['signal'] == -0.25], sub_dir=file_utils.get_data_type(), ticker='fbb_momentum_watch')
# No signal -- mid-range. Sorted by band_position so the extremes of the
# neutral zone are visible at either end of the file.
wait = df[df['signal'] == 0.0].sort_values(by='band_position', ascending=False)
file_utils.result_csv(wait, sub_dir=file_utils.get_data_type(), ticker='fbb_wait')

logger.info("Completed FBB")

