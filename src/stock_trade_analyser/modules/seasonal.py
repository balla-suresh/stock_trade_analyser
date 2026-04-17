import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import pandas as pd

from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.tools.file_utils import FileUtils
from stock_trade_analyser.tools.log_utils import LoggerUtils
from stock_trade_analyser.models.ta import Seasonal


def main() -> int:
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    logger = LoggerUtils("seasonal").get_logger()
    logger.info("Started seasonal")

    seasonal_cfg = config.get("seasonal", {})
    file_utils = FileUtils(data_type=config["download"]["data_type"])
    if seasonal_cfg.get("clean_output", False):
        file_utils.clean()

    loader = Downloader(
        period=config["download"]["period"],
        interval=config["download"]["interval"],
        is_download=config["download"]["is_download"],
        file_utils=file_utils,
    )
    loader.download()
    ticker_list = loader.get_ticker_list()

    seasonal = Seasonal()
    rows = []
    skipped = 0

    if ticker_list is not None:
        for each_ticker in ticker_list:
            if isinstance(each_ticker, dict):
                each_ticker = each_ticker['symbol']

            try:
                current_data = file_utils.import_csv(each_ticker)
            except FileNotFoundError:
                logger.debug("Skipping %s: data file missing", each_ticker)
                skipped += 1
                continue

            try:
                current_data = current_data.dropna()
                if current_data.empty:
                    logger.debug("Skipping %s: no data after dropna", each_ticker)
                    skipped += 1
                    continue

                current_data = current_data.rename(columns=str.lower)
                if 'close' not in current_data.columns:
                    logger.debug("Skipping %s: no close column", each_ticker)
                    skipped += 1
                    continue

                seasonal_data = seasonal.setup(each_ticker, current_data)

                if seasonal_cfg.get("intermediate", False):
                    file_utils.result_csv(
                        seasonal_data,
                        sub_dir=file_utils.get_data_type(),
                        ticker=f"seasonal_{each_ticker}",
                    )

                summary = seasonal.get_summary(each_ticker, current_data, seasonal_data)
                summary['symbol'] = each_ticker
                rows.append(summary)
            except Exception:
                logger.warning("Skipping %s: unexpected error", each_ticker, exc_info=True)
                skipped += 1
                continue

    if not rows:
        logger.info("No seasonal rows generated; exiting (skipped=%s)", skipped)
        return 0

    df = pd.DataFrame(rows).set_index('symbol')

    column_order = [
        'current_quarter_rating',
        'q1_rating',
        'q2_rating',
        'q3_rating',
        'q4_rating',
    ]
    df = df[[c for c in column_order if c in df.columns]]

    df = df.sort_values(
        by=['current_quarter_rating'],
        ascending=[True],
        na_position='last',
    )

    out_dir = os.path.join(file_utils.get_predictions(), file_utils.get_data_type())
    os.makedirs(os.path.realpath(out_dir), exist_ok=True)
    out_path = os.path.join(out_dir, 'seasonal.csv')
    file_utils.result_csv(df, sub_dir=file_utils.get_data_type(), ticker='seasonal')

    logger.info(
        "Completed seasonal: processed=%s skipped=%s output=%s",
        len(rows),
        skipped,
        out_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
