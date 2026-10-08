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
        # ha_open is a sequential recursion (each value depends on the prior
        # one), so the loop stays -- but over NumPy scalars rather than
        # `.iloc`/list-comprehension side effects. Values are unchanged.
        ha_close = df_ha['close'].to_numpy(dtype='float64', copy=False)
        n_ha = len(df_ha)
        ha_open = np.empty(n_ha, dtype='float64')
        if n_ha:
            ha_open[0] = (
                df_ha['old_open'].to_numpy(dtype='float64', copy=False)[0]
                + df_ha['old_close'].to_numpy(dtype='float64', copy=False)[0]
            ) / 2
            for i in range(n_ha - 1):
                ha_open[i + 1] = (ha_open[i] + ha_close[i]) / 2
        df_ha['open'] = ha_open

        # df_ha.set_index('index', inplace=True)
        # df_ha['ha_high'] = df_ha[['ha_open', 'ha_close', 'high']].max(axis=1)
        # df_ha['ha_low'] = df_ha[['ha_open', 'ha_close', 'low']].min(axis=1)
        logger.info(f"Finished Calculated Heikin Ashi")
        return df_ha

    def get_signal(self, data):
        logger.info(f"Starting signals Heikin Ashi")
        # Vectorized: bearish candle (open > close) -> 0, otherwise 1.
        data['position'] = np.where(
            data['open'].to_numpy() > data['close'].to_numpy(), 0, 1
        )

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

        # FINAL UPPER / LOWER BAND
        #
        # These recursions are inherently sequential (bar i depends on i-1), so
        # the loop stays, but it now runs over plain NumPy arrays instead of
        # `DataFrame.iloc` scalar access. That removes the per-element Series
        # boxing that dominated the runtime while computing identical values.
        ub = upper_band.to_numpy(dtype='float64', copy=False)
        lb = lower_band.to_numpy(dtype='float64', copy=False)
        # `close` is aligned to the band index (bands were produced by dropna()).
        cl = close.reindex(upper_band.index).to_numpy(dtype='float64', copy=False)
        n = len(ub)

        fb_upper = np.zeros(n, dtype='float64')
        fb_lower = np.zeros(n, dtype='float64')

        for i in range(1, n):
            if (ub[i] < fb_upper[i-1]) or (cl[i-1] > fb_upper[i-1]):
                fb_upper[i] = ub[i]
            else:
                fb_upper[i] = fb_upper[i-1]

            if (lb[i] > fb_lower[i-1]) or (cl[i-1] < fb_lower[i-1]):
                fb_lower[i] = lb[i]
            else:
                fb_lower[i] = fb_lower[i-1]

        final_bands = pd.DataFrame(
            {'upper': fb_upper, 'lower': fb_lower}, index=upper_band.index
        )

        # SUPERTREND
        st_vals = np.zeros(n, dtype='float64')
        for i in range(1, n):
            prev = st_vals[i-1]
            if prev == fb_upper[i-1]:
                st_vals[i] = fb_upper[i] if cl[i] < fb_upper[i] else (
                    fb_lower[i] if cl[i] > fb_upper[i] else prev)
            elif prev == fb_lower[i-1]:
                st_vals[i] = fb_lower[i] if cl[i] > fb_lower[i] else (
                    fb_upper[i] if cl[i] < fb_lower[i] else prev)
            else:
                st_vals[i] = prev

        supertrend = pd.DataFrame(
            {f'supertrend_{self.lookback}': st_vals}, index=upper_band.index
        )
        supertrend = supertrend.dropna()[1:]

        # ST UPTREND/DOWNTREND (vectorized)
        close = close.iloc[len(close) - len(supertrend):]
        st_arr = supertrend.iloc[:, 0].to_numpy(dtype='float64', copy=False)
        cl_arr = close.to_numpy(dtype='float64', copy=False)

        upt_arr = np.where(cl_arr > st_arr, st_arr, np.nan)
        dt_arr = np.where(cl_arr < st_arr, st_arr, np.nan)

        st = pd.Series(supertrend.iloc[:, 0])
        upt = pd.Series(upt_arr, index=supertrend.index)
        dt = pd.Series(dt_arr, index=supertrend.index)
        upper = pd.Series(final_bands['upper']).iloc[1:]
        lower = pd.Series(final_bands['lower']).iloc[1:]

        upper.index, lower.index = supertrend.index, supertrend.index
        logger.info(f"Finished calculation of Supertrend for {ticker}")
        return st, upt, dt, upper, lower

    def implement_st_strategy(self, ticker, prices, st):
        logger.info(f"Starting Strategy for {ticker}")

        px = prices.to_numpy(dtype='float64', copy=False)
        stv = st.to_numpy(dtype='float64', copy=False)
        n = len(stv)

        # Crossover conditions are elementwise, so compute them in one pass.
        # `np.roll` reproduces the original loop's `iloc[i-1]` indexing, which
        # at i == 0 wraps around to the last bar. That wrap is preserved here
        # deliberately so signals stay identical to the previous behaviour.
        prev_st = np.roll(stv, 1)
        prev_px = np.roll(px, 1)
        cross_up = (prev_st > prev_px) & (stv < px)
        cross_dn = (prev_st < prev_px) & (stv > px)

        # Only the buy/sell alternation is sequential; iterate just the bars
        # where a crossover actually fired instead of every bar.
        st_signal = [0] * n
        signal = 0
        for idx in np.nonzero(cross_up | cross_dn)[0]:
            if cross_up[idx]:
                if signal != 1:
                    signal = 1
                    st_signal[idx] = 1
            else:
                if signal != -1:
                    signal = -1
                    st_signal[idx] = -1

        self.st_signal = st_signal
        logger.info(f"Finished Strategy for {ticker}")
        return st_signal

    def get_signal(self, ticker, data):
        logger.info(f"Starting Position for {ticker}")
        self.implement_st_strategy(ticker, data['close'], data['st'])
        # Position is a forward-fill of the buy/sell signals: 1 after a buy,
        # 0 after a sell, carry the previous value otherwise.
        #
        # Two quirks of the original loops are preserved intentionally so
        # output does not change: the pre-fill defaults every bar to 1 (its
        # `st_signal[i] > 1` test can never be true, since signals are only
        # -1/0/1), and bar 0 in the "carry" case reads `position[-1]`, i.e. the
        # last element, which is still the default 1 at that point.
        sig = np.asarray(self.st_signal)
        n_sig = len(sig)
        position = np.ones(n_sig, dtype='int64')
        if n_sig:
            carry = 1  # position[-1] as seen by i == 0 (pre-filled default)
            for i in range(n_sig):
                if sig[i] == 1:
                    carry = 1
                elif sig[i] == -1:
                    carry = 0
                position[i] = carry
        position = position.tolist()

        close_price = data['close']
        st = data['st']
        self.st_signal = pd.DataFrame(self.st_signal).rename(
            columns={0: 'st_signal'}).set_index(data.index)  # type: ignore
        position = pd.DataFrame(position).rename(
            columns={0: 'position'}).set_index(data.index)  # type: ignore

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
                            max_iter=300, n_init=10, random_state=0)  # type: ignore
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


class FibonacciBollingerBands:
    def __init__(self, length: int = 200, multiplier: float = 3.0, use_vwma: bool = True):
        """
        Initialize Fibonacci Bollinger Bands
        
        Parameters:
        -----------
        length : int
            Period for calculation (default: 200)
        multiplier : float
            Multiplier for standard deviation (default: 3.0)
        use_vwma : bool
            If True, use Volume Weighted Moving Average
            If False, use Simple Moving Average
        """
        logger.info("Initializing FibonacciBollingerBands")
        self.length = length
        self.multiplier = multiplier
        self.use_vwma = use_vwma

    def _vwma(self, src, volume, length):
        """
        Volume Weighted Moving Average (VWMA)
        VWMA = Sum(Price * Volume) / Sum(Volume) over the period
        """
        return (src * volume).rolling(window=length).sum() / volume.rolling(window=length).sum()

    def setup(self, data):
        """
        Calculate Fibonacci Bollinger Bands
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame with OHLCV data (columns should be lowercase)
        
        Returns:
        --------
        pd.DataFrame with FBB columns added
        """
        logger.info(f"Calculating Fibonacci Bollinger Bands: length={self.length}, multiplier={self.multiplier}, use_vwma={self.use_vwma}")
        
        df = data.copy()
        
        # Ensure column names are lowercase
        df.columns = df.columns.str.lower()
        
        # Calculate typical price (hlc3)
        tp = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate basis (moving average)
        if self.use_vwma:
            # Check if Volume column exists
            if 'volume' not in df.columns:
                logger.warning("Volume column not found. Falling back to Simple Moving Average.")
                basis = tp.rolling(self.length).mean()
            else:
                basis = self._vwma(tp, df['volume'], self.length)
        else:
            # Simple Moving Average
            basis = tp.rolling(self.length).mean()
        
        # Calculate standard deviation of the source (tp/hlc3)
        dev = self.multiplier * tp.rolling(self.length).std()
        
        # Calculate Fibonacci Bollinger Bands
        df['fbb_mid'] = basis
        df['fbb_up1'] = basis + (0.236 * dev)
        df['fbb_up2'] = basis + (0.382 * dev)
        df['fbb_up3'] = basis + (0.5 * dev)
        df['fbb_up4'] = basis + (0.618 * dev)
        df['fbb_up5'] = basis + (0.764 * dev)
        df['fbb_up6'] = basis + (1 * dev)
        df['fbb_low1'] = basis - (0.236 * dev)
        df['fbb_low2'] = basis - (0.382 * dev)
        df['fbb_low3'] = basis - (0.5 * dev)
        df['fbb_low4'] = basis - (0.618 * dev)
        df['fbb_low5'] = basis - (0.764 * dev)
        df['fbb_low6'] = basis - (1 * dev)
        
        logger.info("Finished calculating Fibonacci Bollinger Bands")
        return df
        
    def get_band(self, price, row):
        if pd.isna(price) or pd.isna(row['fbb_mid']):
            return "unknown"
        
        if price >= row['fbb_up6']: return "above_up6"
        if price >= row['fbb_up5']: return "up5_up6"
        if price >= row['fbb_up4']: return "up4_up5"
        if price >= row['fbb_up3']: return "up3_up4"
        if price >= row['fbb_up2']: return "up2_up3"
        if price >= row['fbb_up1']: return "up1_up2"
        if price >= row['fbb_mid']: return "mid_up1"
        
        if price >= row['fbb_low1']: return "low1_mid"
        if price >= row['fbb_low2']: return "low2_low1"
        if price >= row['fbb_low3']: return "low3_low2"
        if price >= row['fbb_low4']: return "low4_low3"
        if price >= row['fbb_low5']: return "low5_low4"
        if price >= row['fbb_low6']: return "low6_low5"
        return "below_low6"

    def get_signal(self, trend, band):
        if band == "above_up6":
            return "SELL"
        elif band == "up5_up6":
            return "BUY_WATCH" if trend == 1 else "SELL"
        elif band in ["up4_up5", "up3_up4", "up2_up3", "up1_up2"]:
            return "BUY_WATCH" if trend == 1 else "WAIT"
        elif band == "mid_up1":
            return "BUY" if trend == 1 else "WAIT"
        elif band == "low1_mid":
            return "WAIT" if trend == 1 else "SELL"
        elif band in ["low2_low1", "low3_low2", "low4_low3", "low5_low4"]:
            return "WAIT" if trend == 1 else "SELL_WATCH"
        elif band == "low6_low5":
            return "BUY" if trend == 1 else "SELL_WATCH"
        elif band == "below_low6":
            return "BUY"
        return "WAIT"

    def get_metrics(self, df):
        """
        Calculates trend, band, signal, target price, and estimated days to target
        based on the calculated FBB data.
        """
        # Add a 20-day SMA to determine the trend
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Get the last two rows
        last_row = df.iloc[-1]
        prev_row = df.iloc[-2] if len(df) > 1 else last_row
        
        current_price = last_row['close']
        
        # Trend: 1 for up (20-day SMA rising), 0 for down (20-day SMA falling)
        trend = 1 if last_row['sma_20'] > prev_row['sma_20'] else 0
        
        band = self.get_band(current_price, last_row)
        signal = self.get_signal(trend, band)

        # Calculate target and estimated days
        fbb_mid = last_row['fbb_mid']
        if trend == 1:
            target_price = last_row['fbb_up6'] if current_price >= fbb_mid else fbb_mid
        else:
            target_price = last_row['fbb_low6'] if current_price <= fbb_mid else fbb_mid

        pct_to_target = 0
        est_days = -1
        if pd.notna(target_price) and current_price > 0:
            pct_to_target = (target_price - current_price) / current_price
            
            # Calculate historical drift (mean daily return)
            daily_returns = df['close'].pct_change().dropna()
            drift = daily_returns.mean()
            
            if drift != 0:
                est_days = int(round(abs(pct_to_target) / abs(drift)))

        return {
            'close': round(current_price, 2),
            'trend': trend,
            'band': band,
            'signal': signal,
            'target_price': round(target_price, 2) if pd.notna(target_price) else None,
            'pct_to_target': round(pct_to_target * 100, 2),
            'est_days': est_days
        }



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


class Seasonal:
    """Per-quarter historical performance ratings.

    For each ticker the class computes the % price increase for each
    calendar quarter (Q1..Q4) in every historical year, averages those
    across years to get one number per quarter, then ranks the four
    quarters 1..4 where 1 = worst and 4 = best.

    The (year, quarter) of the last bar in the series is excluded from
    the per-year averages when aggregating by quarter, so a still-open
    calendar quarter does not bias historical quarter comparisons.
    """

    QUARTERS = [1, 2, 3, 4]

    def __init__(self):
        logger.debug("Initializing Seasonal")

    @staticmethod
    def _current_quarter(index: pd.DatetimeIndex) -> int:
        return int(index[-1].quarter)

    def setup(self, ticker: str, data: pd.DataFrame) -> pd.DataFrame:
        """Compute per-quarter average % increase and a 1..4 rating.

        Returns a DataFrame indexed by quarter (1..4) with columns:
          - avg_return_pct: mean of (start_close -> end_close) % return
            across every historical year of that quarter
          - years_covered: number of years contributing to the average
          - rating: 1 = worst avg_return_pct, 4 = best
        Quarters with no historical data get avg_return_pct = NaN and
        rating = None.
        """
        logger.debug(f"Starting Seasonal setup for {ticker}")
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        df = df.assign(_year=df.index.year, _quarter=df.index.quarter)

        per_year = df.groupby(['_year', '_quarter']).agg(
            start_close=('close', 'first'),
            end_close=('close', 'last'),
        ).reset_index()
        per_year['return_pct'] = (
            (per_year['end_close'] - per_year['start_close']) / per_year['start_close']
        ) * 100

        last_ts = df.index[-1]
        per_year_fit = per_year[
            ~((per_year['_year'] == last_ts.year) & (per_year['_quarter'] == last_ts.quarter))
        ]
        per_quarter = per_year_fit.groupby('_quarter').agg(
            avg_return_pct=('return_pct', 'mean'),
            years_covered=('_year', 'nunique'),
        )
        per_quarter = per_quarter.reindex(self.QUARTERS)
        per_quarter['years_covered'] = per_quarter['years_covered'].fillna(0).astype(int)
        per_quarter['avg_return_pct'] = per_quarter['avg_return_pct'].round(4)

        ranks = per_quarter['avg_return_pct'].rank(method='min', ascending=True, na_option='keep')
        per_quarter['rating'] = ranks.astype('Int64')

        logger.debug(f"Finished Seasonal setup for {ticker}")
        return per_quarter

    def get_summary(self, ticker: str, data: pd.DataFrame, seasonal_data: pd.DataFrame) -> dict:
        """Return a flat dict with the 4 quarter ratings plus the current
        quarter and its rating."""
        logger.debug(f"Starting Seasonal summary for {ticker}")
        current_q = self._current_quarter(data.index)

        out = {
            'current_quarter': current_q,
            'current_quarter_rating': None,
        }
        for q in self.QUARTERS:
            if q in seasonal_data.index:
                rating = seasonal_data.loc[q, 'rating']
                out[f'q{q}_rating'] = int(rating) if pd.notna(rating) else None
            else:
                out[f'q{q}_rating'] = None

        out['current_quarter_rating'] = out.get(f'q{current_q}_rating')
        logger.debug(f"Finished Seasonal summary for {ticker}")
        return out