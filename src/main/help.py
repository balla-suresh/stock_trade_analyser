import pandas as pd
import pandas_ta as ta
from src.lib.tools.log_utils import LoggerUtils
from src.lib.tools.file_utils import FileUtils
from src.lib.tools.downloader import Downloader

config = {
    "download": {
        "interval": "5m",
        "period": "10d",
        "is_download": True
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

ohlcv_dict = {'open': 'first',
              'high': 'max',
              'low': 'min',
              'close': 'last',
              'volume': 'sum'
              }

logger = None
logger = LoggerUtils("new_super_trend").get_logger()
logger.info("Started testing")

file_utils = FileUtils(data_type="intraday" )
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()


for each_ticker in ticker_list:
    # current_data = data[each_ticker]
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    print(current_data.columns)
    # Calculate Returns and append to the df DataFrame
    current_data.ta.supertrend(
        append=True, multiplier=config["supertrend"]["multiplier"], length=config["supertrend"]["lookback"])

    # New Columns with results
    print(current_data.columns)

    # Take a peek
    print(current_data.tail())

    

    # print(current_data.groupby('Date').mean({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}))
    # print(current_data.dt.hour)
    new_data = current_data.resample('15T').agg(ohlcv_dict).dropna()
    new_data.ta.supertrend(
        append=True, multiplier=config["supertrend"]["multiplier"], length=config["supertrend"]["lookback"])
    print(new_data.tail())
    
    if config["supertrend"]["intermediate"]:
        file_utils.result_csv(new_data, ticker=each_ticker)
