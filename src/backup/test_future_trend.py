from src.lib.tools.downloader import Downloader
from src.lib.models.ta import FutureTrend
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.backtest import BackTest
from src.lib.tools.file_utils import FileUtils
import numpy as np

config = {
    "download": {
        "interval": "1d",
        "period": "1y",
        "is_download": True,
        "data_type": "day"
    },
    "future_trend": {
        "length": 150,
        "multi": 2,
        "extend": 20,
        "period": 14
    }
}

logger = None
logger = LoggerUtils("future_trend").get_logger()
logger.info("Started testing")

file_utils = FileUtils(data_type=config["download"]["data_type"])
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

future_trend = FutureTrend(config["future_trend"]["length"], config["future_trend"]["multi"],)
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')

for each_ticker in ticker_list:
    if isinstance(each_ticker, dict):
        each_ticker = each_ticker['symbol']
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    result = future_trend.get_future_price(current_data)
    print(f"{each_ticker} {result}")

logger.info("Completed testing")