import os
import shutil
import logging
import csv
import pandas as pd
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class FileUtils:
    def __init__(self, output: str = "output", predictions: str = "predictions", data_type: str = "day"):
        self.ROOT_DIR = Path(__file__).parent.parent.parent.parent
        self.output = f"{self.ROOT_DIR}/{output}"
        self.ticker_file = f"{self.ROOT_DIR}/data/tickers.csv"
        self.tv_ticker_file = f"{self.ROOT_DIR}/data/tv_tickers.csv"
        self.predictions = f"{self.ROOT_DIR}/{predictions}"
        self.data_type = f"{data_type}"

    def get_output(self):
        return self.output

    def get_predictions(self):
        return self.predictions

    def get_data_type(self):
        return self.data_type

    def clean(self):
        # shutil.rmtree(os.path.realpath(self.predictions), ignore_errors=True)
        os.makedirs(os.path.realpath(self.predictions + '/' + "day"), exist_ok=True)
        os.makedirs(os.path.realpath(self.predictions + '/' + "intraday"), exist_ok=True)
        os.makedirs(os.path.realpath(self.output + '/' + "day"), exist_ok=True)
        os.makedirs(os.path.realpath(self.output + '/' + "intraday"), exist_ok=True)

    def read_ticker(self):
        ticker_list = []
        with open(self.ticker_file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            name = None
            for row in reader:
                if row and row[0]:
                    ticker_list.append(row[0])
        return ticker_list

    def read_tv_ticker(self):
        ticker_list = []
        with open(self.tv_ticker_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            name = None
            for row in reader:
                if row:
                    ticker_list.append(row)
        return ticker_list

    def create_all_csv(self, data, ticker_list: list = None, output: str = None):
        if not output:
            output = self.output + '/' + self.data_type
        logger.info("Started CSV creation for all")
        if not ticker_list:
            logger.info("Skipping CSV creation for all")
            return ticker_list
        new_ticker_list = ticker_list.copy()
        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            ticker_data = data.dropna()
            if len(ticker_data) == 0:
                logger.info("Skipping CSV creation for " + ticker)
                new_ticker_list.remove(ticker)
            else:
                self.export_csv(ticker_data, ticker, output, 'Date')
        else:
            for ticker in ticker_list:
                ticker_data = data[ticker].dropna()
                if len(ticker_data) == 0:
                    logger.info("Skipping CSV creation for " + ticker)
                    new_ticker_list.remove(ticker)
                else:
                    self.export_csv(ticker_data, ticker, output, 'Date')
        logger.info("Finished CSV creation for all")
        return new_ticker_list

    def result_csv(self, data, ticker, output: str = None, index: str = None):
        if not output:
            output = self.predictions + '/' + self.data_type
            self.export_csv(data, ticker, output)

    def export_csv(self, data, ticker, output: str = None, index: str = None):
        if not output:
            output = self.output + '/' + self.data_type
        file = output + '/' + ticker + '.csv'
        logger.info(f"Started exporting for {ticker}")
        # data.dropna(inplace=True)
        if index:
            data.index.name = index
        data.to_csv(file, sep=',', encoding='utf-8')
        logger.info(f"Finished exporting for {ticker}")

    def import_csv(self, ticker, output: str = None, index: str = 'Date'):
        if not output:
            output = self.output + '/' + self.data_type
        file = output + '/' + ticker + '.csv'
        logger.info(f"Started importing for {ticker}")
        data = pd.read_csv(file, index_col=index, parse_dates=True)
        logger.info(f"Finished importing for {ticker}")
        return data

    def get_new_ticker_list(self, ticker_list):
        new_ticker_list = []
        for ticker in ticker_list:
            if isinstance(ticker, str):
                # check file and import
                file = self.output + '/' + ticker + '.csv'
                if os.path.exists(file):
                    logger.info(f"{ticker} already exists")
                else:
                    logger.info(f"{ticker} does not exists")
                    new_ticker_list.append(ticker)
            else:
                current_ticker = ticker['symbol']
                file = self.output + '/' + current_ticker + '.csv'
                if os.path.exists(file):
                    logger.info(f"{current_ticker} already exists")
                else:
                    logger.info(f"{current_ticker} does not exists")
                    new_ticker_list.append(ticker)
        return new_ticker_list

    def write_json(self, data, ticker, predictions: str = None):
        if not predictions:
            predictions = self.predictions
        file = predictions + '/' + ticker
        logger.info(f"Started writing for {file}")
        with open(file, "w") as write_file:
            json.dump(data, write_file, indent=4)
        logger.info(f"Finished writing for {file}")
