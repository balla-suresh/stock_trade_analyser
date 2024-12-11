from src.lib.tools.downloader import Dowloader
from src.lib.models.ta import SuperTrend
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.backtest import BackTest
from src.lib.tools.file_utils import FileUtils
import datetime

supertrend_list = [
    {
        'lookback': 10,
        'multiplier': 3
    }
]
config = {
    "download": {
        "interval": "15m",
        "period": "10d",
        "is_download": False
    },
    "supertrend": {
        "lookback": 10,
        "multiplier": 3,
        "intermediate": False
    },
    "helkin_ashi": {
        "input_size": 1,  # since we are only using 1 feature, close price
        "num_lstm_layers": 3,
        "lstm_size": 64,
        "dropout": 0.2,
    }
}

logger = None
logger = LoggerUtils("super_trend").get_logger()
logger.info("Started testing")

file_utils = FileUtils(data_type="intraday")
file_utils.clean()
loader = Dowloader(period=config["download"]["period"], interval=config["download"]
                   ["interval"], is_download=config["download"]["is_download"], file_utils=file_utils)
data = loader.download()
ticker_list = loader.get_ticker_list()

supertrend = SuperTrend(
    config["supertrend"]["lookback"], config["supertrend"]["multiplier"])
back_test = BackTest()

df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')

st = []
trade = []
close = []
upper = []
lower = []
differences = []
for each_ticker in ticker_list:
    # current_data = data[each_ticker]
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)
    # current_data.index = pd.to_datetime(current_data.index) - datetime.timedelta(hours=6, minutes=30)

    current_data['st'], current_data['s_upt'], current_data['st_dt'], current_data['upper'], current_data['lower'] = supertrend.setup(each_ticker, current_data.loc[:, (
        'high')], current_data.loc[:, ('low')], current_data.loc[:, ('close')], lookback=config["supertrend"]['lookback'], multiplier=config["supertrend"]['multiplier'])
    current_data = current_data[1:]
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
    
    if strategy.iloc[-1]['position'] and strategy.iloc[-1]['st_signal']:
        trade.append(1)
    elif not strategy.iloc[-1]['position'] and not strategy.iloc[-1]['st_signal']:
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

logger.info("Completed testing")