import pandas as pd
import numpy as np
from detecta import detect_peaks
import logging
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class HeikinAshi:
    def __init__(self):
        logger.info("Initializing HeikinAshi")

    def setup(self, data):
        logger.info(f"Starting Calculated Heikin Ashi")
        df_ha = data.copy()
        df_ha['old_close'] = df_ha['close']
        df_ha['old_open'] = df_ha['open']

        df_ha['close'] = (df_ha['old_open'] + df_ha['high'] + df_ha['low'] + df_ha['old_close']) / 4
        # df_ha.reset_index(inplace=True)
        ha_open = [(df_ha['old_open'][0] + df_ha['old_close'][0]) / 2]
        [ha_open.append((ha_open[i] + df_ha['close'].values[i]) / 2) \
         for i in range(0, len(df_ha) - 1)]
        df_ha['open'] = ha_open

        # df_ha.set_index('index', inplace=True)
        # df_ha['ha_high'] = df_ha[['ha_open', 'ha_close', 'high']].max(axis=1)
        # df_ha['ha_low'] = df_ha[['ha_open', 'ha_close', 'low']].min(axis=1)
        logger.info(f"Finished Calculated Heikin Ashi")
        return df_ha

    def get_signal(self, data):
        logger.info(f"Starting signals Heikin Ashi")
        ha_signals = []
        for i in range(len(data)):
            if data['open'][i] > data['close'][i]:
                ha_signals.append(0)
            else:
                ha_signals.append(1)
        data['position'] = ha_signals

        logger.info(f"Finished signals Heikin Ashi")
        return data


class SuperTrend:
    def __init__(self, lookback: int = 10, multiplier: int = 3):
        logger.info("Starting SuperTrend")
        self.lookback = lookback
        self.multiplier = multiplier

    def setup(self, ticker, high, low, close, lookback: int = 10, multiplier: int = 3):
        if lookback:
            self.lookback = lookback
        if multiplier:
            self.multiplier = multiplier
        logger.info(
            f"Calculating SuperTrend for {ticker} : {self.lookback} {self.multiplier}")

        # ATR

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
        atr = tr.ewm(self.lookback).mean()

        # H/L AVG AND BASIC UPPER & LOWER BAND

        hl_avg = (high + low) / 2
        upper_band = (hl_avg + self.multiplier * atr).dropna()
        lower_band = (hl_avg - self.multiplier * atr).dropna()

        # FINAL UPPER BAND

        final_bands = pd.DataFrame(columns=['upper', 'lower'])
        final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
        final_bands.iloc[:, 1] = final_bands.iloc[:, 0]

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 0] = 0
            else:
                if (upper_band[i] < final_bands.iloc[i-1, 0]) | (close[i-1] > final_bands.iloc[i-1, 0]):
                    final_bands.iloc[i, 0] = upper_band[i]
                else:
                    final_bands.iloc[i, 0] = final_bands.iloc[i-1, 0]

        # FINAL LOWER BAND

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 1] = 0
            else:
                if (lower_band[i] > final_bands.iloc[i-1, 1]) | (close[i-1] < final_bands.iloc[i-1, 1]):
                    final_bands.iloc[i, 1] = lower_band[i]
                else:
                    final_bands.iloc[i, 1] = final_bands.iloc[i-1, 1]

        # SUPERTREND
        supertrend = pd.DataFrame(columns=[f'supertrend_{self.lookback}'])
        supertrend.iloc[:, 0] = [
            x for x in final_bands['upper'] - final_bands['upper']]

        for i in range(len(supertrend)):
            if i == 0:
                supertrend.iloc[i, 0] = 0
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] < final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close[i] > final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] > final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close[i] < final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

        supertrend = supertrend.set_index(upper_band.index)
        supertrend = supertrend.dropna()[1:]

        # ST UPTREND/DOWNTREND

        upt = []
        dt = []
        close = close.iloc[len(close) - len(supertrend):]

        for i in range(len(supertrend)):
            if close[i] > supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close[i] < supertrend.iloc[i, 0]:
                upt.append(np.nan)
                dt.append(supertrend.iloc[i, 0])
            else:
                upt.append(np.nan)
                dt.append(np.nan)

        st, upt, dt, upper, lower = pd.Series(
            supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt), pd.Series(final_bands['upper']), pd.Series(final_bands['lower'])
        upper = upper.iloc[1:]
        lower = lower.iloc[1:]
        new_row = pd.DataFrame()
        pd.concat([new_row,upper.loc[:]]).reset_index(drop=True) 
        
        upt.index, dt.index, upper.index, lower.index = supertrend.index, supertrend.index, supertrend.index, supertrend.index
        logger.info(f"Finished calculation of Supertrend for {ticker}")
        return st, upt, dt, upper, lower

    def implement_st_strategy(self, ticker, prices, st):
        logger.info(f"Starting Strategy for {ticker}")
        buy_price = []
        sell_price = []
        st_signal = []
        signal = 0

        for i in range(len(st)):
            if st[i-1] > prices[i-1] and st[i] < prices[i]:
                if signal != 1:
                    buy_price.append(prices[i])
                    sell_price.append(np.nan)
                    signal = 1
                    st_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    st_signal.append(0)
            elif st[i-1] < prices[i-1] and st[i] > prices[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices[i])
                    signal = -1
                    st_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    st_signal.append(0)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                st_signal.append(0)

        self.st_signal = st_signal
        # return buy_price, sell_price, st_signal
        logger.info(f"Finished Strategy for {ticker}")
        return st_signal

    def get_signal(self, ticker, data):
        logger.info(f"Starting Position for {ticker}")
        self.implement_st_strategy(ticker, data['close'], data['st'])
        position = []
        for i in range(len(self.st_signal)):
            if self.st_signal[i] > 1:
                position.append(0)
            else:
                position.append(1)

        for i in range(len(data['close'])):
            if self.st_signal[i] == 1:
                position[i] = 1
            elif self.st_signal[i] == -1:
                position[i] = 0
            else:
                position[i] = position[i-1]

        close_price = data['close']
        st = data['st']
        self.st_signal = pd.DataFrame(self.st_signal).rename(
            columns={0: 'st_signal'}).set_index(data.index)
        position = pd.DataFrame(position).rename(
            columns={0: 'position'}).set_index(data.index)

        # frames = [close_price, st, data['s_upt'], data['st_dt'], data['upper'], data['lower'],
        #           self.st_signal, position]
        frames = [close_price, st, data['upper'], data['lower'],
                  self.st_signal, position]
        strategy = pd.concat(frames, join='inner', axis=1)

        # strategy.head()
        # print(strategy[20:25])
        logger.info(f"Finishing Position for {ticker}")
        return strategy


class SupportResistance:
    def get_optimum_clusters(self, df, saturation_point=0.01):
        wcss = []
        k_models = []
        dates = []

        size = min(11, len(df.index))
        for i in range(1, size):
            kmeans = KMeans(n_clusters=i, init='k-means++',
                            max_iter=300, n_init=10, random_state=0)
            kmeans.fit(df)
            wcss.append(kmeans.inertia_)
            k_models.append(kmeans)

        # Compare differences in inertias until it's no more than saturation_point
        optimum_k = len(wcss)-1
        for i in range(0, len(wcss)-1):
            diff = abs(wcss[i+1] - wcss[i])
            if diff < saturation_point:
                optimum_k = i
                break
        # print("Optimum K is " + str(optimum_k + 1))
        optimum_clusters = k_models[optimum_k]

        return optimum_clusters

    def setup(self, data):
        lows = pd.DataFrame(data=data, index=data.index, columns=["low"])
        highs = pd.DataFrame(data=data, index=data.index, columns=["high"])
        low_clusters = self.get_optimum_clusters(lows)

        low_centers = low_clusters.cluster_centers_
        low_centers = np.sort(low_centers, axis=0)

        high_clusters = self.get_optimum_clusters(highs)
        high_centers = high_clusters.cluster_centers_
        high_centers = np.sort(high_centers, axis=0)

        return low_centers, high_centers

    def get_signal(self, ticker, data):
        print("nothing")

class ZigZag:
    def __init__(self, zigzag_period: int = 10, show_projection: bool = True):
        self.zigzag_period = zigzag_period
        self.show_projection = show_projection

    def zigzag_with_projection(self, data, zigzag_period=20, show_projection=True):
        high = data['high']
        low = data['low']
        bar_index = np.arange(len(data))

        # Zigzag variables
        ph = high.rolling(zigzag_period).max()
        pl = low.rolling(zigzag_period).min()

        dir = 0
        zz_points = []

        for i in range(len(data)):
            if high[i] == ph[i]:
                dir = 1
                zz_points.append((bar_index[i], high[i]))
            elif low[i] == pl[i]:
                dir = -1
                zz_points.append((bar_index[i], low[i]))

        zz_points = np.array(zz_points)

        # Projection logic
        if show_projection and len(zz_points) >= 4:
            last_direction = 1 if zz_points[-1, 1] > zz_points[-2, 1] else -1
            last_length = abs(zz_points[-1, 1] - zz_points[-2, 1])

            avg_bullish_length = np.mean([abs(zz_points[i, 1] - zz_points[i + 1, 1]) for i in range(0, len(zz_points) - 1, 2)])
            avg_bearish_length = np.mean([abs(zz_points[i, 1] - zz_points[i + 1, 1]) for i in range(1, len(zz_points) - 1, 2)])

            if last_direction == 1:
                proj_length = avg_bullish_length - last_length if avg_bullish_length > last_length else 0
            else:
                proj_length = avg_bearish_length - last_length if avg_bearish_length > last_length else 0

            if proj_length > 0:
                start_x, start_y = zz_points[-1]
                end_x = start_x + proj_length
                end_y = start_y + proj_length * last_direction
                return f"{last_direction}:{start_y}:{end_y}"
                # plt.plot([start_x, end_x], [start_y, end_y], linestyle='dotted', color='red', label='Projection')


class FutureTrend:
    def __init__(self, length: int = 10, multi: int = 2, extend: int = 0, period: int = 5):
        self.length = length
        self.multi = multi
        self.extend = extend
        self.period = period

    def calculate_atr(self, high, low, close):
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=self.period).mean()
        return atr

    def future_price(self, x1, x2, y1, y2, index):
        slope = (y2 - y1) / (x2 - x1)
        return y1 + slope * (index - x1)

    def trend_detection(self, close, atr):
        sma = close.rolling(window=self.length).mean()
        upper = sma + atr
        lower = sma - atr
        trend = np.zeros_like(close)
        trend[close > upper] = 1
        trend[close < lower] = -1
        return trend

    def get_future_price(self, data):
        atr = self.calculate_atr(data['high'], data['low'], data['close'])
        trend = self.trend_detection(data['close'], atr)
        global proj_price
        close = data['close']
        high = data['high']
        low = data['low']
        bar_index = np.arange(len(data))

        mid = close.rolling(window=self.length).mean()
        upper = mid + atr * self.multi
        lower = mid - atr * self.multi

        # Future projection
        for i in range(1, len(trend)):
            # if trend[i] != trend[i - 1]:
            x1, x2 = bar_index[i - 1], bar_index[i]
            y1, y2 = mid[i - 1], mid[i]
            proj_index = bar_index[-1] + self.extend
            proj_price = self.future_price(x1, x2, y1, y2, proj_index)
        return proj_price