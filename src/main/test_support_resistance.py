from src.lib.tools.downloader import Downloader
from src.lib.models.ta import SupportResistance
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.backtest import BackTest
from src.lib.tools.file_utils import FileUtils
import datetime

supertrend_list = [
    {
        'lookback': 10,
        'multiplier': 3
    }
]
config = {
    "download": {
        "interval": "15m",
        "period": "10d",
        "is_download": False
    }
}

logger = None
logger = LoggerUtils("support_resistance").get_logger()
logger.info("Started testing")

file_utils = FileUtils()
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

support_resistance = SupportResistance()
for each_ticker in ticker_list:
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    low_centers, high_centers = support_resistance.setup(current_data)
    print(f"support levels for ticker {each_ticker}:{low_centers}")
    print(f"resistance levels for ticker {each_ticker}:{high_centers}")

logger.info("Completed testing")