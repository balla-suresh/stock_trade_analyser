import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import FibonacciBollingerBands
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.backtest import BackTest
from stock_trade_analyser.tools.file_utils import FileUtils
import datetime

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json'), 'r') as f:
    config = json.load(f)

logger = LoggerUtils("fbb").get_logger()
logger.info("Started FBB")

file_utils = FileUtils(data_type=config["download"]["data_type"])
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
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])  # type: ignore
df = df.set_index('symbol')

signals = []
close_prices = []
differences = []
directions = []

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
        
        # Calculate difference from FBB bands or target price
        diff = None
        target_price = strategy.iloc[-1].get('target_price')
        direction = strategy.iloc[-1].get('direction', '')
        signal = strategy.iloc[-1]['fbb_signal']
        
        if target_price and not pd.isna(target_price):
            # Use target price for difference calculation
            diff = target_price - strategy.iloc[-1]['close']
        elif signal == 0.5:  # Partial buy - use fbb_low5
            diff = strategy.iloc[-1]['close'] - strategy.iloc[-1]['fbb_low5']
        elif signal == -0.5:  # Partial sell - use fbb_up6
            diff = strategy.iloc[-1]['fbb_up6'] - strategy.iloc[-1]['close']
        elif direction == 'up':
            # Upward direction: difference from upper band
            diff = strategy.iloc[-1]['fbb_up6'] - strategy.iloc[-1]['close']
        elif direction == 'down':
            # Downward direction: difference from lower band
            diff = strategy.iloc[-1]['close'] - strategy.iloc[-1]['fbb_low6']
        else:
            # No direction: difference from middle band
            diff = strategy.iloc[-1]['close'] - strategy.iloc[-1]['fbb_mid']
        
        diff = round((diff / strategy.iloc[-1]['close']) * 100, 2)
        
        signals.append(signal)
        close_prices.append(strategy.iloc[-1]['close'])
        differences.append(diff)
        directions.append(direction)
df['signal'] = signals
df['close'] = close_prices
df['diff'] = differences
df['direction'] = directions
# Sort by signal and difference
df = df.sort_values(by=['signal', 'direction', 'diff'], ascending=[False, False, True])

# Save results
# Full signals
file_utils.result_csv(df[df['signal'] == 1], sub_dir=file_utils.get_data_type(), ticker='fbb_buy')
file_utils.result_csv(df[df['signal'] == -1], sub_dir=file_utils.get_data_type(), ticker='fbb_sell')
# Partial signals
file_utils.result_csv(df[df['signal'] == 0.5], sub_dir=file_utils.get_data_type(), ticker='fbb_partial_buy')
file_utils.result_csv(df[df['signal'] == -0.5], sub_dir=file_utils.get_data_type(), ticker='fbb_partial_sell')
# No signal
file_utils.result_csv(df[df['signal'] == 0], sub_dir=file_utils.get_data_type(), ticker='fbb_wait')

logger.info("Completed FBB")

