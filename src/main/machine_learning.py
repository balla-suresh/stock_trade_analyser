from src.lib.tools.downloader import Downloader
from src.lib.models.ml import *
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.file_utils import FileUtils

config = {
    "download": {
        "interval": "1m",
        "period": "5d",
        "is_download": False
    }
}

logger = None
logger = LoggerUtils("machine_learning").get_logger()
logger.info("Started testing")

file_utils = FileUtils()
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

# model = MLP()
model = LSTMmodel()
# model = GRUmodel()

for each_ticker in ticker_list:
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    y_test, y_pred = model.evalute(df=current_data)
    print(f"accuracy_score for ticker {each_ticker}:{accuracy_score(y_test, y_pred)}")

logger.info("Completed testing")
