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

    def _calculate_reversal_probability(self, df, target_level, current_idx, lookback=50):
        """
        Calculate probability of reversal after touching a FBB level based on historical patterns.
        """
        if target_level is None or current_idx < lookback:
            return 0.5  # Default probability
        
        # Vectorized touch/reversal scan over the lookback window.
        level_values = df[target_level].iloc[max(0, current_idx-lookback):current_idx]
        prices = df['close'].iloc[max(0, current_idx-lookback):current_idx]

        px = prices.to_numpy(dtype='float64', copy=False)
        lv = level_values.to_numpy(dtype='float64', copy=False)
        n = len(px)
        if n < 2:
            touch_count = 0
            reversal_count = 0
        else:
            # Crossings evaluated at i (1..n-1) against the level at i,
            # mirroring the original loop's index alignment.
            prev_px = px[:-1]
            curr_px = px[1:]
            lvl = lv[1:]
            if target_level.startswith('fbb_up'):
                touch = (prev_px <= lvl) & (curr_px > lvl)
            else:
                touch = (prev_px >= lvl) & (curr_px < lvl)

            idx = np.nonzero(touch)[0] + 1  # positions i in the window
            touch_count = int(idx.size)

            # A touch only counts as a reversal when a full 6-bar forward
            # window exists (`i + 5 < n`), matching the original guard; touches
            # too close to the window end stay in touch_count only.
            eligible = idx[idx + 5 < n]
            if eligible.size:
                # rows of forward windows prices[i:i+6]
                windows = px[eligible[:, None] + np.arange(6)]
                lvl_at = lv[eligible]
                if target_level.startswith('fbb_up'):
                    reversal_count = int((windows.min(axis=1) < lvl_at).sum())
                else:
                    reversal_count = int((windows.max(axis=1) > lvl_at).sum())
            else:
                reversal_count = 0

        if touch_count > 0:
            return reversal_count / touch_count
        else:
            # If no historical touches, use level-based probability
            # Extreme levels (up6, low6) have higher reversal probability
            if 'up6' in target_level or 'low6' in target_level:
                return 0.75
            elif 'up5' in target_level or 'low5' in target_level:
                return 0.65
            elif 'up4' in target_level or 'low4' in target_level:
                return 0.55
            else:
                return 0.45

    def _calculate_dynamic_direction(self, df, current_idx, current_price, fbb_levels):
        """
        Calculate direction dynamically using multiple timeframes and recent price action.
        
        Returns:
        --------
        tuple: (direction, velocity, optimal_lookback)
        """
        min_lookback = 3
        max_lookback = min(30, len(df) - 1)
        
        # Check for recent band touches and reversals (last 5-10 days)
        recent_reversal_detected = False
        recent_touch_direction = None
        
        # Look back up to 10 days for recent band touches
        lookback_recent = min(10, current_idx)
        for i in range(max(1, current_idx - lookback_recent), current_idx):
            close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-1]
            
            # Get historical FBB levels for that period
            if 'fbb_up6' in df.columns and 'fbb_low6' in df.columns:
                up6_hist = df['fbb_up6'].iloc[i]
                low6_hist = df['fbb_low6'].iloc[i]
                
                if not pd.isna(up6_hist) and not pd.isna(low6_hist):
                    # Check if price touched upper band and reversed
                    if prev_close >= up6_hist and close < up6_hist:
                        # Touched upper band and reversed down
                        recent_reversal_detected = True
                        recent_touch_direction = 'down'
                        break
                    
                    # Check if price touched lower band and reversed
                    if prev_close <= low6_hist and close > low6_hist:
                        # Touched lower band and reversed up
                        recent_reversal_detected = True
                        recent_touch_direction = 'up'
                        break
        
        # Calculate momentum using multiple timeframes with weights
        velocities = []
        weights = []
        lookbacks = [3, 5, 7, 10, 15, 20]  # Multiple timeframes
        
        for lookback in lookbacks:
            if current_idx < lookback:
                continue
                
            recent_prices = df['close'].iloc[-lookback:].values
            if len(recent_prices) < 2:
                continue
            
            # Use linear regression for velocity
            recent_dates = np.arange(len(recent_prices))
            slope = np.polyfit(recent_dates, recent_prices, 1)[0]
            velocities.append(slope)
            
            # Weight: more weight on shorter timeframes (recent momentum is more important)
            weight = 1.0 / lookback
            weights.append(weight)
        
        if not velocities:
            # Fallback: use simple 3-day momentum
            if current_idx >= 3:
                recent_prices = df['close'].iloc[-3:].values
                velocity = np.mean(np.diff(recent_prices))
            else:
                velocity = 0
            return ('up' if velocity > 0 else 'down', velocity, 3)
        
        # Weighted average velocity
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        velocity = np.average(velocities, weights=weights)
        
        # Determine direction
        direction = 'up' if velocity > 0 else 'down'
        
        # If recent reversal detected, adjust direction
        if recent_reversal_detected:
            # Recent reversal takes precedence if it's strong
            # Check if the reversal momentum is stronger than overall trend
            reversal_lookback = min(5, current_idx)
            if reversal_lookback >= 2:
                reversal_prices = df['close'].iloc[-reversal_lookback:].values
                reversal_dates = np.arange(len(reversal_prices))
                reversal_velocity = np.polyfit(reversal_dates, reversal_prices, 1)[0]
                
                # If reversal momentum is significant (at least 50% of overall velocity)
                if abs(reversal_velocity) > abs(velocity) * 0.5:
                    direction = recent_touch_direction
                    velocity = reversal_velocity
                    # Ensure velocity sign matches the overridden direction
                    if direction == 'up' and velocity < 0:
                        velocity = abs(velocity)
                    elif direction == 'down' and velocity > 0:
                        velocity = -abs(velocity)
        
        # Use optimal lookback based on which timeframe has strongest momentum
        optimal_lookback = lookbacks[np.argmax(np.abs(velocities))]
        
        return direction, velocity, optimal_lookback

    def predict_fbb_touch_and_reversal(self, df, lookback_period=None, max_days_ahead=60):
        """
        Predict which Fibonacci Bollinger Band level the price will touch and when,
        before it reverses.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLC data and FBB columns (lowercase column names)
        lookback_period : int, optional
            Number of days to look back for velocity calculation (if None, calculated dynamically)
        max_days_ahead : int
            Maximum number of days to project forward
        
        Returns:
        --------
        dict : Prediction results with:
            - target_level: Which FBB level will be touched (e.g., 'fbb_up6', 'fbb_low6')
            - target_price: Price level to be touched
            - predicted_date: Estimated date when level will be touched
            - days_to_touch: Number of days until touch
            - current_price: Current closing price
            - direction: 'up' or 'down'
            - reversal_probability: Probability of reversal after touch (0-1)
        """
        if len(df) < 3:
            return None
        
        # Get current values
        current_idx = len(df) - 1
        current_price = df['close'].iloc[current_idx]
        
        # Handle date index
        if isinstance(df.index, pd.DatetimeIndex):
            current_date = df.index[current_idx]
        else:
            try:
                current_date = pd.to_datetime(df.index[current_idx])
            except:
                current_date = pd.Timestamp.now()
        
        # Get latest FBB levels (skip NaN values)
        fbb_levels = {}
        for level in ['fbb_up6', 'fbb_up5', 'fbb_up4', 'fbb_up3', 'fbb_up2', 'fbb_up1', 
                      'fbb_mid', 'fbb_low1', 'fbb_low2', 'fbb_low3', 'fbb_low4', 'fbb_low5', 'fbb_low6']:
            if level in df.columns:
                value = df[level].iloc[current_idx]
                if not pd.isna(value):
                    fbb_levels[level] = value
        
        # Calculate direction dynamically
        direction, velocity, optimal_lookback = self._calculate_dynamic_direction(
            df, current_idx, current_price, fbb_levels
        )
        
        # Use optimal lookback for velocity calculation if not provided
        if lookback_period is None:
            lookback_period = optimal_lookback
        
        # Find which level will be touched first
        # Use all bands sorted by distance from current price in the direction of travel
        target_level = None
        target_price = None
        days_to_touch = None
        
        all_level_names = [
            'fbb_up6', 'fbb_up5', 'fbb_up4', 'fbb_up3', 'fbb_up2', 'fbb_up1',
            'fbb_mid',
            'fbb_low1', 'fbb_low2', 'fbb_low3', 'fbb_low4', 'fbb_low5', 'fbb_low6'
        ]
        
        if direction == 'up':
            # Collect all bands above current price, sorted closest first
            candidates = []
            for level in all_level_names:
                if level in fbb_levels:
                    level_price = fbb_levels[level]
                    if level_price > current_price:
                        candidates.append((level, level_price))
            candidates.sort(key=lambda x: x[1])
            
            for level, level_price in candidates:
                distance = level_price - current_price
                if velocity > 0:
                    days_needed = distance / velocity
                    if days_needed > 0 and days_needed <= max_days_ahead:
                        target_level = level
                        target_price = level_price
                        days_to_touch = int(np.ceil(days_needed))
                        break
            
            if target_level is None:
                # Price is already above all bands — predict reversal back
                # to the nearest band below current price
                reversal_candidates = []
                for level in all_level_names:
                    if level in fbb_levels:
                        level_price = fbb_levels[level]
                        if level_price <= current_price:
                            reversal_candidates.append((level, level_price))
                reversal_candidates.sort(key=lambda x: x[1], reverse=True)
                
                for level, level_price in reversal_candidates:
                    distance = current_price - level_price
                    abs_velocity = abs(velocity) if velocity != 0 else 1
                    days_needed = distance / abs_velocity
                    if days_needed > 0 and days_needed <= max_days_ahead:
                        target_level = level
                        target_price = level_price
                        days_to_touch = int(np.ceil(days_needed))
                        direction = 'reversal_down'
                        break
        else:
            # Collect all bands below current price, sorted closest first
            candidates = []
            for level in all_level_names:
                if level in fbb_levels:
                    level_price = fbb_levels[level]
                    if level_price < current_price:
                        candidates.append((level, level_price))
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            for level, level_price in candidates:
                distance = current_price - level_price
                if velocity < 0:
                    days_needed = distance / abs(velocity)
                    if days_needed > 0 and days_needed <= max_days_ahead:
                        target_level = level
                        target_price = level_price
                        days_to_touch = int(np.ceil(days_needed))
                        break
            
            if target_level is None:
                # Price is already below all bands — predict reversal back
                # to the nearest band above current price
                reversal_candidates = []
                for level in all_level_names:
                    if level in fbb_levels:
                        level_price = fbb_levels[level]
                        if level_price >= current_price:
                            reversal_candidates.append((level, level_price))
                reversal_candidates.sort(key=lambda x: x[1])
                
                for level, level_price in reversal_candidates:
                    distance = level_price - current_price
                    abs_velocity = abs(velocity) if velocity != 0 else 1
                    days_needed = distance / abs_velocity
                    if days_needed > 0 and days_needed <= max_days_ahead:
                        target_level = level
                        target_price = level_price
                        days_to_touch = int(np.ceil(days_needed))
                        direction = 'reversal_up'
                        break
        
        # Calculate reversal probability based on historical patterns
        reversal_probability = self._calculate_reversal_probability(df, target_level, current_idx)
        
        # Price beyond all bands has high reversal probability
        if direction in ('reversal_up', 'reversal_down'):
            reversal_probability = max(reversal_probability, 0.75)
        
        # Calculate predicted date
        if days_to_touch:
            try:
                if isinstance(current_date, pd.Timestamp):
                    predicted_date = current_date + pd.Timedelta(days=days_to_touch)
                else:
                    predicted_date = pd.Timestamp.now() + pd.Timedelta(days=days_to_touch)
            except:
                predicted_date = None
        else:
            predicted_date = None
        
        logger.info("Predicted date: %s", predicted_date)
        logger.info("Days to touch: %s", days_to_touch)
        logger.info("Direction: %s", direction)
        logger.info("Velocity: %s", velocity)
        logger.info("Reversal probability: %s", reversal_probability)

        return {
            'target_level': target_level,
            'target_price': target_price,
            'predicted_date': predicted_date,
            'days_to_touch': days_to_touch,
            'current_price': current_price,
            'current_date': current_date,
            'direction': direction,
            'velocity': velocity,
            'reversal_probability': reversal_probability,
            'all_levels': fbb_levels
        }

    def get_signal(self, ticker, data, lookback_period=None, max_days_ahead=60):
        """
        Get trading signals based on Fibonacci Bollinger Bands with predictions
        
        Parameters:
        -----------
        ticker : str
            Ticker symbol
        data : pd.DataFrame
            DataFrame with FBB columns (from setup method)
        lookback_period : int, optional
            Number of days to look back for velocity calculation in prediction.
            If None, calculated dynamically based on recent price action.
        max_days_ahead : int
            Maximum number of days to project forward in prediction
        
        Returns:
        --------
        pd.DataFrame with signal columns added:
            - fbb_signal: Trading signal (0: no signal, 0.5: partial buy, 1: full buy, -0.5: partial sell, -1: full sell)
            - target_price: Predicted price level to be touched
            - days_to_touch: Number of days until target level is touched
            - reversal_probability: Probability of reversal after touch
            - direction: Price direction ('up' or 'down')
        """
        logger.info(f"Getting FBB signals for {ticker}")
        
        df = data.copy()
        
        # Get prediction for the latest data point
        prediction = self.predict_fbb_touch_and_reversal(df, lookback_period, max_days_ahead)
        
        # Initialize signal columns
        # Signal values: 0: no signal, 0.5: partial buy (fbb_low5), 1: full buy, -0.5: partial sell (fbb_up6), -1: full sell
        df['fbb_signal'] = 0.0  # Initialize as float to support 0.5 and -0.5 values
        df['target_price'] = None
        df['days_to_touch'] = None
        df['reversal_probability'] = None
        df['direction'] = None
        
        # Signal logic (vectorized):
        # - Full buy (1.0) when price touches or crosses fbb_low6 (extreme lower band)
        # - Partial buy (0.5) when price touches or crosses fbb_low5 (but not fbb_low6)
        # - Partial sell (-0.5) when price touches or crosses fbb_up6
        # Priority: full buy > partial buy, so fbb_low6 is applied last and wins.
        #
        # The original row loop also OR-ed in a "crossed from above" term
        # (`prev_close > level_prev and close <= level`); that is a subset of
        # `close <= level`, so `A or (B and A)` collapses to `A` and the
        # elementwise comparisons below are equivalent. Bar 0 is excluded to
        # match the loop's `range(1, len(df))` start.
        close = df['close']
        active = np.zeros(len(df), dtype=bool)
        active[1:] = True

        if 'fbb_up6' in df.columns:
            up6 = df['fbb_up6']
            df.loc[active & up6.notna() & (close >= up6), 'fbb_signal'] = -0.5

        if 'fbb_low5' in df.columns:
            low5 = df['fbb_low5']
            df.loc[active & low5.notna() & (close <= low5), 'fbb_signal'] = 0.5

        if 'fbb_low6' in df.columns:
            low6 = df['fbb_low6']
            df.loc[active & low6.notna() & (close <= low6), 'fbb_signal'] = 1.0

        # Add prediction data to the last row
        if prediction:
            last_idx = len(df) - 1
            df.loc[df.index[last_idx], 'target_price'] = prediction.get('target_price')
            df.loc[df.index[last_idx], 'days_to_touch'] = prediction.get('days_to_touch')
            df.loc[df.index[last_idx], 'reversal_probability'] = prediction.get('reversal_probability')
            df.loc[df.index[last_idx], 'direction'] = prediction.get('direction')
        
        logger.info(f"Finished getting FBB signals for {ticker}")
        return df


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