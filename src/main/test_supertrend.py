from src.lib.tools.downloader import Dowloader
from src.lib.models.ta import SuperTrend
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.backtest import BackTest
from src.lib.tools.file_utils import FileUtils

supertrend_list = [
    {
        'lookback': 10,
        'multiplier': 3
    }
    ,
    {
        'lookback': 15,
        'multiplier': 3
    },
    {
        'lookback': 30,
        'multiplier': 3
    },
    {
        'lookback': 50,
        'multiplier': 3
    }
]
logger = None
logger = LoggerUtils("super_trend").get_logger()
logger.info("Started testing")

file_utils = FileUtils()
file_utils.clean()
loader = Dowloader(period='1y', file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

supertrend = SuperTrend(7, 3)
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')
for each_trend in supertrend_list:
    logger.info(
        f"Started testing for {each_trend['lookback']} and {each_trend['multiplier']} ")
    percentages = []
    signals = []
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
        file_utils.result_csv(strategy, ticker=each_ticker)
        
        # percentage = [each_ticker,profit_percentage]
        percentages.append(profit_percentage)
        signals.append(strategy.iloc[-1]['position'])
        # print(strategy)
    percent = f"P_{each_trend['lookback']}"
    signal = f"S_{each_trend['lookback']}"
    df[percent] = percentages
    df[signal] = signals

df = df.sort_values(by=['S_10', 'S_15', 'S_30', 'S_50'], ascending=[False, False, False, False])
# print(df)
file_utils.result_csv(df, ticker='percentage')

logger.info("Completed testing")
