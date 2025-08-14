import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import HeikinAshi
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.backtest import BackTest
from stock_trade_analyser.tools.file_utils import FileUtils
# from detecta import detect_peaks


logger = None
interval = '1D'
period = '1y'
is_download = True
logger = LoggerUtils("heikin_ashi").get_logger()
logger.info("Started testing")

file_utils = FileUtils()
file_utils.clean()
loader = Downloader(period=period, interval=interval, is_download=is_download, file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()
heikin_ashi = HeikinAshi()
back_test = BackTest()


df = pd.DataFrame(ticker_list, columns=['symbol']) # type: ignore
df = df.set_index('symbol')
percentages = []
signals = []
date = []
for each_ticker in ticker_list:
    if isinstance(each_ticker, dict):
        each_ticker = each_ticker['symbol']
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)

    ha_data = heikin_ashi.setup(current_data)
    heikin_ashi.get_signal(ha_data)

    # generate CSV for stock
    ha_data.dropna(inplace=True)
    file_utils.result_csv(ha_data, sub_dir=file_utils.get_data_type(), ticker=each_ticker)

    profit_percentage = back_test.back_test(
        ticker=each_ticker, strategy=ha_data)
    logger.info(f"profit_percentage for {each_ticker}: {profit_percentage}")
    percentages.append(profit_percentage)
    signals.append(ha_data.iloc[-1]['position'])
    date.append(ha_data.index[-1])

df['percent'] = percentages
df['signal'] = signals
df['date'] = date
df = df.sort_values(by=['percent'], ascending=[False])
# print(df)
file_utils.result_csv(df, sub_dir=file_utils.get_data_type(), ticker='percentage')

logger.info("Completed testing")

def test_heikin_ashi():
    assert True
