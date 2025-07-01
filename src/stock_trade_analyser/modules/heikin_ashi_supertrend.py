import sys
import os
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from stock_trade_analyser.tools.downloader import Downloader
from stock_trade_analyser.models.ta import SuperTrend
from stock_trade_analyser.models.ta import HeikinAshi
from stock_trade_analyser.tools.log_utils import LoggerUtils
import pandas as pd
from stock_trade_analyser.tools.backtest import BackTest
from stock_trade_analyser.tools.file_utils import FileUtils
import datetime

with open(os.path.join(os.path.dirname(__file__), '..', 'config', 'day.json'), 'r') as f:
    config = json.load(f)

logger = None
logger = LoggerUtils("super_trend").get_logger()
logger.info("Started super_trend")

file_utils = FileUtils(data_type=config["download"]["data_type"])
file_utils.clean()
loader = Downloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

supertrend = SuperTrend(
    config["supertrend"]["lookback"], config["supertrend"]["multiplier"])
heikin_ashi = HeikinAshi()
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])  # type: ignore
df = df.set_index('symbol')

st = []
trade = []
close = []
upper = []
lower = []
differences = []
if ticker_list is not None:
    for each_ticker in ticker_list:
        # current_data = data[each_ticker]
        if isinstance(each_ticker, dict):
            each_ticker = each_ticker['symbol']
        current_data = file_utils.import_csv(each_ticker)
        current_data = current_data.dropna()
        current_data = current_data.rename(columns=str.lower)
        ha_data = heikin_ashi.setup(current_data)
        heikin_ashi.get_signal(ha_data)
        # current_data.index = pd.to_datetime(current_data.index) - datetime.timedelta(hours=6, minutes=30)

        current_data['st'], current_data['s_upt'], current_data['st_dt'], current_data['upper'], current_data['lower'] = supertrend.setup(each_ticker, ha_data.loc[:, (
            'high')], ha_data.loc[:, ('low')], ha_data.loc[:, ('close')], lookback=config["supertrend"]['lookback'], multiplier=config["supertrend"]['multiplier'])
        current_data = current_data[1:]
        current_data['close'] = ha_data['close']
        # print(current_data)

        strategy = supertrend.get_signal(each_ticker, current_data)
        profit_percentage = back_test.back_test(
            ticker=each_ticker, strategy=strategy)

        # generate CSV for stock
        # print(strategy.iloc[-1]['position'])
        if config["supertrend"]["intermediate"]:
            file_utils.result_csv(strategy, ticker=each_ticker)
        diff = None
        if strategy.iloc[-1]['position']:
            diff = strategy.iloc[-1]['close'] - strategy.iloc[-1]['st']
        else:
            diff = strategy.iloc[-1]['st'] - strategy.iloc[-1]['close']

        diff = round((diff / strategy.iloc[-1]['close'])*100, 2)
        
        if strategy.iloc[-1]['position'] and strategy.iloc[-1]['st_signal'] and ha_data.iloc[-1]['position']:
            trade.append(1)
        elif not strategy.iloc[-1]['position'] and not strategy.iloc[-1]['st_signal'] and not ha_data.iloc[-1]['position']:
            trade.append(-1)
        else:
            trade.append(0)
        
        # position.append(strategy.iloc[-1]['position'])
        st.append(strategy.iloc[-1]['st'])
        close.append(strategy.iloc[-1]['close'])
        upper.append(strategy.iloc[-1]['upper'])
        lower.append(strategy.iloc[-1]['lower'])
        differences.append(diff)

df['trade'] = trade
df['diff'] = differences
df['close'] = close
df['upper'] = upper
df['lower'] = lower
df['st'] = st

df = df.sort_values(by=['trade', 'diff'], ascending=[False, True])

file_utils.result_csv(df[df['trade'] == 1], ticker='supertrend_buy')
file_utils.result_csv(df[df['trade'] == -1], ticker='supertrend_sell')
file_utils.result_csv(df[df['trade'] == 0], ticker='supertrend_wait')

logger.info("Completed super_trend")