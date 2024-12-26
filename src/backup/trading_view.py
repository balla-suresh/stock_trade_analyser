import pandas as pd
import pandas_ta as ta
from src.lib.tools.log_utils import LoggerUtils
from src.lib.tools.file_utils import FileUtils
from src.lib.tools.downloader import Downloader
from tvDatafeed import TvDatafeed, Interval

# tv = TvDatafeed("sureshballa", "Sunis*1234567890123")

# # index
# nifty_index_data = tv.get_hist(symbol='NIFTY',exchange='NSE',interval=Interval.in_1_hour,n_bars=1000)

# print(nifty_index_data)

config = {
    "download": {
        "interval": "5m",
        "period": "10d",
        "is_download": False
    },
    "supertrend": {
        "lookback": 10,
        "multiplier": 3,
        "intermediate": True
    },
    "heikin_ashi": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 3,
        "lstm_size": 64,
        "dropout": 0.2,
    }
}

logger = None
logger = LoggerUtils("trading_view").get_logger()
logger.info("Started testing")

file_utils = FileUtils(data_type="intraday")
file_utils.clean()

loader = Downloader(interval=Interval.in_15_minute, n_bars=1000, is_download=config["download"]["is_download"], file_utils=file_utils)

ticker_list = loader.get_ticker_list()
data = loader.tv_download()

for each_ticker in ticker_list:
    
    each_ticker=each_ticker['symbol']
    # current_data = data[each_ticker]
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)

    # Calculate Returns and append to the df DataFrame
    current_data.ta.supertrend(
        append=True, multiplier=config["supertrend"]["multiplier"], length=config["supertrend"]["lookback"])
    # New Columns with results
    # print(current_data.columns)
    
    if config["supertrend"]["intermediate"]:
        file_utils.result_csv(current_data, ticker=each_ticker)

logger.info("Completed testing")
