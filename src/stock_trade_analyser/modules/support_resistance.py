import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import SupportResistance
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.backtest import BackTest
from stock_trade_analyser.tools.file_utils import FileUtils, parse_ticker_file_arg
import datetime

import json

ticker_file_arg = parse_ticker_file_arg(
    prog="support_resistance",
    description="Run the support/resistance analysis.",
)

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json'), 'r') as f:
    config = json.load(f)

logger = LoggerUtils("support_resistance").get_logger()
logger.info("Started testing")

file_utils = FileUtils(ticker_file=ticker_file_arg)
logger.info(f"Using ticker file: {file_utils.ticker_file}")
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

support_resistance = SupportResistance()
merged_outputs = {}

if ticker_list is not None:
    for each_ticker in ticker_list:
        if isinstance(each_ticker, dict):
            each_ticker = each_ticker['symbol']
            
        try:
            current_data = file_utils.import_csv(each_ticker)
            if current_data is None or current_data.empty:
                continue
            current_data = current_data.dropna()
            current_data = current_data.rename(columns=str.lower)
            low_centers, high_centers = support_resistance.setup(current_data)
            
            # Convert numpy arrays to flat lists of floats for JSON serialization
            low_list = [float(x[0]) for x in low_centers] if len(low_centers) > 0 else []
            high_list = [float(x[0]) for x in high_centers] if len(high_centers) > 0 else []
            
            merged_outputs[each_ticker] = {
                "support_levels": low_list,
                "resistance_levels": high_list
            }
        except Exception as e:
            logger.error(f"Error processing {each_ticker}: {e}")

# Write directly to predictions folder
predictions_dir = file_utils.get_predictions()
os.makedirs(predictions_dir, exist_ok=True)
path = os.path.join(predictions_dir, 'support_resistance.json')

with open(path, 'w') as f:
    json.dump(merged_outputs, f, indent=4)

logger.info(f"Saved support and resistance levels to {path}")
logger.info("Completed testing")


def test_support_resistance():
    assert True