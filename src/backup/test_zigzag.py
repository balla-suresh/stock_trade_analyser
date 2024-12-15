from src.lib.tools.downloader import Downloader
from src.lib.models.ta import ZigZag
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
    "zigzag": {
        "zigzag_period": 10,
        "show_projection": True
    }
}

logger = None
logger = LoggerUtils("zig_zag").get_logger()
logger.info("Started testing")

file_utils = FileUtils(data_type=config["download"]["data_type"])
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

zigzag = ZigZag(config["zigzag"]["zigzag_period"], config["zigzag"]["show_projection"])
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')

for each_ticker in ticker_list:
    if isinstance(each_ticker, dict):
        each_ticker = each_ticker['symbol']
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    result = zigzag.zigzag_with_projection(current_data, config["zigzag"]["zigzag_period"],)
    print(f"{each_ticker} {result}")

logger.info("Completed testing")