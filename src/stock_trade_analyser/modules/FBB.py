import sys
import os
import json
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import FibonacciBollingerBands
from stock_trade_analyser.tools.log_utils import LoggerUtils
from stock_trade_analyser.tools.file_utils import FileUtils, parse_ticker_file_arg

def main():
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
    
    loader = Downloader(
        period=config["download"]["period"], 
        interval=config["download"]["interval"], 
        is_download=config["download"]["is_download"], 
        file_utils=file_utils
    )
    loader.download()
    ticker_list = loader.get_ticker_list()

    fbb = FibonacciBollingerBands(
        length=config["fbb"]["length"],
        multiplier=config["fbb"]["multiplier"],
        use_vwma=config["fbb"]["use_vwma"]
    )

    rows = []

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

                if len(current_data) < config["fbb"]["length"]:
                    continue

                # Calculate FBB
                fbb_data = fbb.setup(current_data)
                
                # Get metrics
                metrics = fbb.get_metrics(fbb_data)
                metrics['symbol'] = each_ticker

                rows.append(metrics)
            except Exception as e:
                logger.error(f"Error processing {each_ticker}: {e}")

    if rows:
        df = pd.DataFrame(rows)
        df = df.set_index('symbol')
        
        # Save separate files based on signal
        signals = df['signal'].unique()
        for signal in signals:
            signal_df = df[df['signal'] == signal]
            file_name = f"fbb_{signal.lower()}"
            file_utils.result_csv(signal_df, sub_dir=file_utils.get_data_type(), ticker=file_name)
            logger.info(f"Saved {len(signal_df)} {signal} signals to {file_name}.csv")
    else:
        logger.warning("No data processed.")

    logger.info("Completed FBB")

if __name__ == "__main__":
    main()


