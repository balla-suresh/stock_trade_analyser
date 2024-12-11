from src.lib.tools.downloader import Downloader
from src.lib.models.ta import HeikinAshi
from src.lib.tools.log_utils import LoggerUtils
import pandas as pd
from src.lib.tools.backtest import BackTest
from src.lib.tools.file_utils import FileUtils
# from detecta import detect_peaks


logger = None
interval = '5m'
period = '5d'
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


df = pd.DataFrame(ticker_list, columns=['symbol'])
df = df.set_index('symbol')
percentages = []
signals = []
date = []
for each_ticker in ticker_list:
    current_data = file_utils.import_csv(each_ticker)
    current_data = current_data.dropna()
    current_data = current_data.rename(columns=str.lower)

    ha_data = heikin_ashi.setup(current_data)
    heikin_ashi.get_signal(ha_data)

    # generate CSV for stock
    ha_data.dropna(inplace=True)
    file_utils.result_csv(ha_data, ticker=each_ticker)

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
file_utils.result_csv(df, ticker='percentage')

logger.info("Completed testing")
