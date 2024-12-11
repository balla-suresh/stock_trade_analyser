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
        for i in range(data.shape[0]):
            if i > 0:
                df_ha.loc[data.index[i], 'open'] = (
                    data['open'][i-1] + data['close'][i-1])/2

            df_ha.loc[data.index[i], 'close'] = (
                data['open'][i] + data['close'][i] + data['low'][i] + data['high'][i])/4
        df_ha['old_open'] = data['open']
        df_ha['old_close'] = data['close']
        df_ha = df_ha.iloc[1:, :]
        logger.info(f"Finished Calculated Heikin Ashi")
        return df_ha

    def get_signal(self, data):
        logger.info(f"Starting signals Heikin Ashi")
        ha_signals = []
        for i in range(len(data)):
            if data['open'][i] > data['close'][i]:
                ha_signals.append(data['low'][i])
            else:
                ha_signals.append(data['high'][i])
        data['ha'] = ha_signals
        valley = detect_peaks(data['ha'], mpd=5, valley=True)
        peak = detect_peaks(data['ha'], mpd=5, valley=False)

        logger.info("Calculating HA values")
        peaks = peak.tolist() + valley.tolist()
        peaks.sort()
        previous = 0
        next = 0
        current = 0
        new_data = []
        length = len(peaks)
        for i in peaks:
            index = peaks.index(i)
            current = data['ha'][i]
            if index == 0:
                previous = current
                next = current

            if previous >= current and current <= next:
                # print(f"first {index} {i} {previous} {current} {next} ")
                new_data.append(current)
            elif previous < current and current > next:
                # print(f"second {index} {i} {previous} {current} {next} ")
                new_data.append(current)
            else:
                # print(f"last {index} {i} {previous} {current} {next} ")
                new_data.append(np.nan)
            previous = data['ha'][i]

            if index >= length-2:
                next = data['ha'][i]
            else:
                next = data['ha'][peaks[index+2]]

        logger.info("Calculating HA sinals")
        # print(new_data)
        new_final_data = []
        for i in range(len(data)):
            if i in peaks:
                position = peaks.index(i)
                value = new_data[position]
                new_final_data.append(value)
            elif i == len(data)-1:
                new_final_data.append(data['ha'][i])
            else:
                new_final_data.append(np.nan)
        data['ha_signal'] = new_final_data
        data.dropna(inplace=True)
        logger.info("Calculating HA position")
        position = []
        for i in range(len(data)):
            # if i == 0:
            #     position.append(1)
            if not i % 2:
                position.append(1)
            else:
                position.append(0)
        data['position'] = position
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
