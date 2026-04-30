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
                if name != row[0]:
                    name = row[0]
                    ticker_list.append(name)
        return ticker_list

    def read_tv_ticker(self):
        ticker_list = []
        with open(self.tv_ticker_file, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            name = None
            for row in reader:
                ticker_list.append(row)

        new_ticker_list = [i for n, i in enumerate(ticker_list) if i not in ticker_list[n + 1:]]
        return new_ticker_list

    def create_all_csv(self, data, ticker_list: list | None = None, output: str | None = None):
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
                self.export_csv(ticker_data, ticker, output)
        else:
            for ticker in ticker_list:
                ticker_data = data[ticker].dropna()
                if len(ticker_data) == 0:
                    logger.info("Skipping CSV creation for " + ticker)
                    new_ticker_list.remove(ticker)
                else:
                    self.export_csv(ticker_data, ticker, output)
        logger.info("Finished CSV creation for all")
        return new_ticker_list


    def result_csv(self, data, ticker='result', sub_dir: str | None = None):
        if sub_dir is None:
            path = self.get_download_path(ticker)
        else:
            path = f"{self.predictions}/{sub_dir}/{ticker}.csv"
        data.to_csv(path)

    def export_csv(self, data, ticker, output: str | None = None):
        if not output:
            output = self.get_download_path(ticker)
        else:
            output = f"{output}/{ticker}.csv"
        logger.info(f"Started exporting for {ticker}")
        data.to_csv(output)
        logger.info(f"Finished exporting for {ticker}")

    def append_csv(self, data, ticker, output: str | None = None):
        """Append new rows to a ticker's CSV, deduplicating by date index.

        Only entries with index strictly greater than the existing file's last
        date are appended (no full rewrite). If the file does not yet exist,
        a fresh CSV is written. Returns the number of rows appended.
        """
        if data is None or len(data) == 0:
            logger.info(f"No data to append for {ticker}")
            return 0

        if output is None:
            path = self.get_download_path(ticker)
        else:
            path = f"{output}/{ticker}.csv"

        new_data = data.copy()
        if hasattr(new_data.index, 'tz') and new_data.index.tz is not None:
            new_data.index = new_data.index.tz_localize(None)

        if not os.path.exists(path):
            logger.info(f"Creating new file for {ticker} with {len(new_data)} rows")
            new_data.to_csv(path)
            return len(new_data)

        last_date = self.get_last_date(ticker, path=path)
        if last_date is None:
            logger.info(f"Existing file for {ticker} unreadable; rewriting")
            new_data.to_csv(path)
            return len(new_data)

        rows_to_append = new_data[new_data.index > last_date]
        if rows_to_append.empty:
            logger.info(f"No new entries to append for {ticker} (last={last_date.date()})")
            return 0

        rows_to_append.to_csv(path, mode='a', header=False)
        logger.info(
            f"Appended {len(rows_to_append)} rows to {ticker} (last was {last_date.date()})"
        )
        return len(rows_to_append)

    def append_all_csv(self, data, ticker_list: list | None = None, output: str | None = None):
        """Multi-ticker version of `append_csv`.

        Mirrors `create_all_csv` but uses `append_csv` so existing files are
        extended (deduped by date) rather than overwritten.
        """
        if not output:
            output = self.output + '/' + self.data_type
        logger.info("Started CSV append for all")
        if not ticker_list:
            logger.info("Skipping CSV append for all (empty ticker list)")
            return ticker_list
        appended_tickers = []
        if len(ticker_list) == 1:
            ticker = ticker_list[0]
            ticker_data = data.dropna()
            if len(ticker_data) == 0:
                logger.info(f"Skipping CSV append for {ticker} (no data)")
            else:
                if self.append_csv(ticker_data, ticker, output) > 0:
                    appended_tickers.append(ticker)
        else:
            for ticker in ticker_list:
                try:
                    ticker_data = data[ticker].dropna()
                except KeyError:
                    logger.info(f"Skipping CSV append for {ticker} (missing in download)")
                    continue
                if len(ticker_data) == 0:
                    logger.info(f"Skipping CSV append for {ticker} (no data)")
                    continue
                if self.append_csv(ticker_data, ticker, output) > 0:
                    appended_tickers.append(ticker)
        logger.info("Finished CSV append for all")
        return appended_tickers

    def import_csv(self, ticker):
        file = self.get_download_path(ticker)
        data = pd.read_csv(file, index_col=0, parse_dates=True)
        return data

    def get_last_date(self, ticker, path: str | None = None):
        """Return the latest date stored in a ticker's CSV, or None.

        Reads only the index column for efficiency. The result is a tz-naive
        `pd.Timestamp` so callers can compare it directly with downloaded data.
        """
        if path is None:
            if isinstance(ticker, dict):
                ticker = ticker.get('symbol')
            path = self.get_download_path(ticker)
        if not os.path.exists(path):
            return None
        try:
            df = pd.read_csv(path, usecols=[0], index_col=0, parse_dates=True)
        except (pd.errors.EmptyDataError, ValueError, IndexError) as exc:
            logger.warning(f"Failed to read last date from {path}: {exc}")
            return None
        if len(df.index) == 0:
            return None
        last_date = df.index.max()
        if hasattr(last_date, 'tz') and last_date.tz is not None:
            last_date = last_date.tz_localize(None)
        return last_date

    def get_last_dates(self, ticker_list):
        """Return a dict mapping ticker symbol -> last date for existing CSVs only."""
        last_dates = {}
        for ticker in ticker_list:
            symbol = ticker['symbol'] if isinstance(ticker, dict) else ticker
            last_date = self.get_last_date(symbol)
            if last_date is not None:
                last_dates[symbol] = last_date
        return last_dates

    def split_ticker_list(self, ticker_list):
        """Split tickers into (new, existing) based on the presence of a CSV file."""
        new_tickers = []
        existing_tickers = []
        for ticker in ticker_list:
            symbol = ticker['symbol'] if isinstance(ticker, dict) else ticker
            file = self.get_download_path(symbol)
            if os.path.exists(file):
                existing_tickers.append(ticker)
            else:
                new_tickers.append(ticker)
        return new_tickers, existing_tickers

    def get_new_ticker_list(self, ticker_list):
        logger.info("In get_new_ticker_list")
        new_ticker_list, _ = self.split_ticker_list(ticker_list)
        for ticker in new_ticker_list:
            symbol = ticker['symbol'] if isinstance(ticker, dict) else ticker
            logger.info(f"{symbol} does not exists")
        return new_ticker_list

    def current_ticker_list(self, ticker_list):
        new_ticker_list = []
        for ticker in ticker_list:
            if isinstance(ticker, str):
                file = self.get_download_path(ticker)
                if os.path.exists(file):
                    new_ticker_list.append(ticker)
        return new_ticker_list

    def write_json(self, data, ticker):
        path = self.get_download_path(ticker, "json")
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_download_path(self, ticker, extension="csv"):
        return f"{self.output}/{self.data_type}/{ticker}.{extension}"
