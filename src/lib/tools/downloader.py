import re
from enum import Enum

import yfinance as yf
import os
import time
import logging
import pandas as pd
from tvDatafeed import TvDatafeed, Interval
from src.lib.tools.file_utils import FileUtils

logger = logging.getLogger(__name__)


class Downloader:
    logger.info("Started Downloader")
    ticker_list = []
    is_download = True

    def __init__(self, downloader: str = "yahoo", period: str = '1y', interval: str = '1d', is_threads: bool = True, is_download: bool = True, n_bars: int = 0, file_utils: FileUtils = FileUtils()):
        self.period = '1y' if period is None else period
        self.interval = '1d' if interval is None else interval
        self.is_threads = is_threads
        self.output = 'output'
        self.is_download = is_download
        self.file_utils = file_utils
        self.n_bars = n_bars
        self.downloader = downloader
        if downloader == "tv":
            self.ticker_list = self.file_utils.read_tv_ticker()
            logger.info(type(self.interval))
            if not isinstance(self.interval, Enum):
                self.interval = Interval(self.interval)
            else:
                self.interval = self.interval
            if self.n_bars == 0:
                self.n_bars = 1000
        else:
            self.ticker_list = self.file_utils.read_ticker()
        # print(self.ticker_list)
        
    def download(self):
        if self.downloader == "tv":
            return self.tv_download()
        else:
            return self.yf_download()

    def yf_download(self):
        logger.info("In Download")
        data = pd.DataFrame()
        new_data = None
        new_ticker_list = []
        # print(self.is_download)
        if self.is_download:
            new_ticker_list = self.file_utils.get_new_ticker_list(
                self.ticker_list)
        else:
            new_ticker_list = self.ticker_list
        if new_ticker_list:
            logger.info("Started Downloading")
            new_data = yf.download(
                tickers=self.ticker_list,
                period=self.period,
                interval=self.interval,
                group_by='ticker',
                auto_adjust=False,
                prepost=False,
                threads=self.is_threads,
                proxy=None,
                ignore_tz=True
            )
            self.ticker_list = self.file_utils.create_all_csv(new_data, new_ticker_list)
            frames = [data, new_data]
            data = pd.concat(frames)

        logger.info("Finished Downloading")
        return data

    def tv_download(self):
        logger.info("In TV Download")
        tv = TvDatafeed(username=os.environ['TV_USERNAME'], password=os.environ['TV_PASSWORD'])
        data = pd.DataFrame()
        new_data = None
        new_ticker_list = []
        # print(self.is_download)
        if self.is_download:
            new_ticker_list = self.file_utils.get_new_ticker_list(
                self.ticker_list)
        else:
            new_ticker_list = self.ticker_list
        if new_ticker_list:
            logger.info("Started TV Downloading")
            tv = TvDatafeed()
            for each_ticker in new_ticker_list:
                new_data = tv.get_hist(
                    symbol=each_ticker['symbol'], exchange=each_ticker['exchange'], interval=self.interval, n_bars=self.n_bars)
                self.file_utils.export_csv(new_data, each_ticker['symbol'],index='Date')
        logger.info("Finished TV Downloading")
        return new_data

    def get_ticker_list(self):
        return self.ticker_list
