import os
import logging
from enum import Enum
from datetime import datetime, timedelta

import yfinance as yf
import pandas as pd
from tvDatafeed import TvDatafeed, Interval

from stock_trade_analyser.tools.file_utils import FileUtils

logger = logging.getLogger(__name__)


class Downloader:
    """Download OHLCV data, doing an *incremental* fetch when possible.

    For each ticker that already has an output CSV, only the bars after the
    file's last date are downloaded and appended (deduped) via FileUtils.
    Tickers without a CSV are backfilled with `period` worth of history.
    """

    logger.info("Started Downloader")
    ticker_list = []
    is_download = True

    def __init__(self, downloader: str = "yahoo", period: str = '1y', interval: str = '1d',
                 is_threads: bool = True, is_download: bool = True, n_bars: int = 0,
                 file_utils: FileUtils = FileUtils()):
        self.period = '1y' if period is None else period
        self.interval = '1d' if interval is None else interval
        self.is_threads = is_threads
        self.output = 'output'
        self.is_download = is_download
        self.file_utils = file_utils
        self.n_bars = n_bars
        self.downloader = downloader
        try:
            username = os.environ['TV_USERNAME']
        except KeyError:
            username = None
            logger.info("downloading from yahoo")
        if username is not None:
            self.downloader = "tv"
            self.ticker_list = self.file_utils.read_tv_ticker()
            logger.info(type(self.interval))
            if not isinstance(self.interval.upper(), Enum):
                self.interval = Interval(self.interval.upper())
            else:
                self.interval = self.interval
            if self.n_bars == 0:
                self.n_bars = 1000
        else:
            self.ticker_list = self.file_utils.read_ticker()

    def download(self):
        if self.downloader == "tv":
            return self.tv_download()
        else:
            return self.yf_download()

    def yf_download(self):
        """Yahoo Finance download with smart incremental updates.

        - Tickers without an existing CSV: fetched with the configured `period`.
        - Tickers with an existing CSV: fetched with `start = min(last_date)+1`
          and `end = today+1` (yfinance treats `end` as exclusive). Per-ticker
          dedup happens inside `FileUtils.append_csv`.
        """
        logger.info("In yf Download")
        assert isinstance(self.interval, str)
        data = pd.DataFrame()

        if not self.is_download:
            logger.info("is_download disabled; skipping Yahoo download")
            self.ticker_list = self.file_utils.current_ticker_list(self.ticker_list)
            return data

        new_tickers, existing_tickers = self.file_utils.split_ticker_list(self.ticker_list)

        backfill_data = self._yf_backfill(new_tickers)
        if backfill_data is not None and not backfill_data.empty:
            data = pd.concat([data, backfill_data])

        update_data = self._yf_incremental(existing_tickers)
        if update_data is not None and not update_data.empty:
            data = pd.concat([data, update_data])

        self.ticker_list = self.file_utils.current_ticker_list(self.ticker_list)
        logger.info("Finished yf Downloading")
        return data

    def _yf_backfill(self, new_tickers):
        if not new_tickers:
            logger.info("No new tickers to backfill")
            return None
        logger.info(f"Backfilling {len(new_tickers)} new tickers (period={self.period})")
        new_data = yf.download(
            tickers=new_tickers,
            period=self.period,
            interval=self.interval,
            group_by='ticker',
            auto_adjust=False,
            prepost=False,
            threads=self.is_threads,
        )
        if new_data is None or new_data.empty:
            logger.info("Backfill returned no data")
            return None
        self.file_utils.create_all_csv(new_data, new_tickers)
        return new_data

    def _yf_incremental(self, existing_tickers):
        if not existing_tickers:
            logger.info("No existing tickers to incrementally update")
            return None

        last_dates = self.file_utils.get_last_dates(existing_tickers)
        if not last_dates:
            logger.info("No readable last dates; skipping incremental update")
            return None

        min_last_date = min(last_dates.values())
        start_date = (min_last_date + timedelta(days=1)).date()
        today = datetime.now().date()
        end_date = today + timedelta(days=1)

        if start_date > today:
            logger.info(
                f"Existing tickers already current (last={min_last_date.date()}); "
                f"skipping incremental download"
            )
            return None

        logger.info(
            f"Incrementally updating {len(existing_tickers)} tickers "
            f"from {start_date} to {today}"
        )
        update_data = yf.download(
            tickers=existing_tickers,
            start=start_date.isoformat(),
            end=end_date.isoformat(),
            interval=self.interval,
            group_by='ticker',
            auto_adjust=False,
            prepost=False,
            threads=self.is_threads,
        )
        if update_data is None or update_data.empty:
            logger.info("Incremental download returned no data")
            return None
        self.file_utils.append_all_csv(update_data, existing_tickers)
        return update_data

    def tv_download(self):
        """TradingView download with per-ticker incremental append.

        TV's `get_hist` doesn't accept date ranges, so we still fetch `n_bars`
        bars per ticker; FileUtils.append_csv then dedupes by date so existing
        CSVs are extended instead of overwritten.
        """
        logger.info("In TV Download")
        assert isinstance(self.interval, Interval)
        new_data = None

        if not self.is_download:
            logger.info("is_download disabled; skipping TV download")
            return new_data

        tv = TvDatafeed(username=os.environ['TV_USERNAME'], password=os.environ['TV_PASSWORD'])
        for each_ticker in self.ticker_list:
            symbol = each_ticker['symbol']
            existing_last = self.file_utils.get_last_date(symbol)
            n_bars = self._tv_n_bars(existing_last)
            if n_bars == 0:
                logger.info(f"{symbol} already current (last={existing_last.date()})")
                continue
            logger.info(f"TV downloading {symbol} ({n_bars} bars)")
            new_data = tv.get_hist(
                symbol=symbol,
                exchange=each_ticker['exchange'],
                interval=self.interval,
                n_bars=n_bars,
            )
            if new_data is None or new_data.empty:
                logger.info(f"TV returned no data for {symbol}")
                continue
            if existing_last is None:
                self.file_utils.export_csv(new_data, symbol)
            else:
                self.file_utils.append_csv(new_data, symbol)
        logger.info("Finished TV Downloading")
        return new_data

    def _tv_n_bars(self, existing_last):
        """Pick a reasonable bar count: full backfill for new tickers, else
        just enough bars to cover the gap (with a small safety buffer)."""
        if existing_last is None:
            return self.n_bars
        gap_days = (datetime.now().date() - existing_last.date()).days
        if gap_days <= 0:
            return 0
        return min(self.n_bars, max(gap_days + 5, 10))

    def get_ticker_list(self):
        return self.ticker_list
