import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ml import *
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.file_utils import FileUtils
import json

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'intraday.json'), 'r') as f:
    config = json.load(f)

logger = None
logger = LoggerUtils("machine_learning").get_logger()
logger.info("Started machine_learning")

file_utils = FileUtils(data_type=config["download"]["data_type"])
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

# model = MLP()
model = LSTMmodel()
# model = GRUmodel()

if ticker_list is not None:
    for each_ticker in ticker_list:
        current_data = file_utils.import_csv(each_ticker)
        current_data = current_data.dropna()
        current_data = current_data.rename(columns=str.lower)
        y_test, y_pred = model.evalute(df=current_data)
        print(f"accuracy_score for ticker {each_ticker}:{accuracy_score(y_test, y_pred)}")

logger.info("Completed machine_learning")
