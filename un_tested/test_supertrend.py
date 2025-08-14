import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import SuperTrend
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.backtest import BackTest
from stock_trade_analyser.tools.file_utils import FileUtils

supertrend_list = [
    {
        'lookback': 20,
        'multiplier': 3
    },
    {
        'lookback': 50,
        'multiplier': 3
    },
    {
        'lookback': 200,
        'multiplier': 3
    }
]
logger = LoggerUtils("super_trend").get_logger()
logger.info("Started testing")

file_utils = FileUtils()
file_utils.clean()
loader = Downloader(period='1y', file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

supertrend = SuperTrend(7, 3)
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol']) # type: ignore
df = df.set_index('symbol')
signal_list = []
for each_trend in supertrend_list:
    logger.info(
        f"Started testing for {each_trend['lookback']} and {each_trend['multiplier']} ")
    percentages = []
    signals = []
    signal_list.append(f"S_{each_trend['lookback']}")
    for each_ticker in ticker_list:
        # current_data = data[each_ticker]
        current_data = file_utils.import_csv(each_ticker)
        current_data = current_data.dropna()
        current_data = current_data.rename(columns=str.lower)

        current_data['st'], current_data['s_upt'], current_data['st_dt'], current_data['upper'], current_data['lower'] = supertrend.setup(each_ticker, current_data.loc[:, (
            'high')], current_data.loc[:, ('low')], current_data.loc[:, ('close')], lookback=each_trend['lookback'], multiplier=each_trend['multiplier'])
        current_data = current_data[1:]

        strategy = supertrend.get_signal(each_ticker, current_data)
        profit_percentage = back_test.back_test(
            ticker=each_ticker, strategy=strategy)
        
        # generate CSV for stock
        # print(strategy.iloc[-1]['position'])
        # file_utils.result_csv(strategy, ticker=each_ticker)
        
        # percentage = [each_ticker,profit_percentage]
        percentages.append(profit_percentage)
        signals.append(strategy.iloc[-1]['position'])
        # print(strategy)
    percent = f"P_{each_trend['lookback']}"
    signal = f"S_{each_trend['lookback']}"
    df[percent] = percentages
    df[signal] = signals

df = df.sort_values(by=["S_20","S_50","S_200","P_20"], ascending=[False, False, False, False])
# print(df)
file_utils.result_csv(df, sub_dir=file_utils.get_data_type(), ticker='percentage')

logger.info("Completed testing")


def test_supertrend():
    assert True
