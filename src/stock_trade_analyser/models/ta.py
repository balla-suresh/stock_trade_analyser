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
        ha_open = [(df_ha['old_open'].iloc[0] + df_ha['old_close'].iloc[0]) / 2]
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
            if data['open'].iloc[i] > data['close'].iloc[i]:
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

        final_bands = pd.DataFrame(index=upper_band.index, columns=['upper', 'lower'])  # type: ignore
        final_bands.iloc[:, 0] = [x for x in upper_band - upper_band]
        final_bands.iloc[:, 1] = final_bands.iloc[:, 0]

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 0] = 0
            else:
                if (upper_band.iloc[i] < final_bands.iloc[i-1, 0]) | (close.iloc[i-1] > final_bands.iloc[i-1, 0]):
                    final_bands.iloc[i, 0] = upper_band.iloc[i]
                else:
                    final_bands.iloc[i, 0] = final_bands.iloc[i-1, 0]

        # FINAL LOWER BAND

        for i in range(len(final_bands)):
            if i == 0:
                final_bands.iloc[i, 1] = 0
            else:
                if (lower_band.iloc[i] > final_bands.iloc[i-1, 1]) | (close.iloc[i-1] < final_bands.iloc[i-1, 1]):
                    final_bands.iloc[i, 1] = lower_band.iloc[i]
                else:
                    final_bands.iloc[i, 1] = final_bands.iloc[i-1, 1]

        # SUPERTREND
        supertrend = pd.DataFrame(index=final_bands.index, columns=[f'supertrend_{self.lookback}'])  # type: ignore
        supertrend.iloc[:, 0] = [
            x for x in final_bands['upper'] - final_bands['upper']]

        for i in range(len(supertrend)):
            if i == 0:
                supertrend.iloc[i, 0] = 0
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close.iloc[i] < final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 0] and close.iloc[i] > final_bands.iloc[i, 0]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close.iloc[i] > final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 1]
            elif supertrend.iloc[i-1, 0] == final_bands.iloc[i-1, 1] and close.iloc[i] < final_bands.iloc[i, 1]:
                supertrend.iloc[i, 0] = final_bands.iloc[i, 0]

        supertrend = supertrend.set_index(upper_band.index)
        supertrend = supertrend.dropna()[1:]

        # ST UPTREND/DOWNTREND

        upt = []
        dt = []
        close = close.iloc[len(close) - len(supertrend):]

        for i in range(len(supertrend)):
            if close.iloc[i] > supertrend.iloc[i, 0]:
                upt.append(supertrend.iloc[i, 0])
                dt.append(np.nan)
            elif close.iloc[i] < supertrend.iloc[i, 0]:
                upt.append(np.nan)
                dt.append(supertrend.iloc[i, 0])
            else:
                upt.append(np.nan)
                dt.append(np.nan)

        st, upt, dt, upper, lower = pd.Series(
            supertrend.iloc[:, 0]), pd.Series(upt), pd.Series(dt), pd.Series(final_bands['upper']), pd.Series(final_bands['lower'])
        upper = upper.iloc[1:]
        lower = lower.iloc[1:]
        
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
            if st.iloc[i-1] > prices.iloc[i-1] and st.iloc[i] < prices.iloc[i]:
                if signal != 1:
                    buy_price.append(prices.iloc[i])
                    sell_price.append(np.nan)
                    signal = 1
                    st_signal.append(signal)
                else:
                    buy_price.append(np.nan)
                    sell_price.append(np.nan)
                    st_signal.append(0)
            elif st.iloc[i-1] < prices.iloc[i-1] and st.iloc[i] > prices.iloc[i]:
                if signal != -1:
                    buy_price.append(np.nan)
                    sell_price.append(prices.iloc[i])
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
        
        # Look at historical touches of this level
        reversal_count = 0
        touch_count = 0
        
        # Get the level values
        level_values = df[target_level].iloc[max(0, current_idx-lookback):current_idx]
        prices = df['close'].iloc[max(0, current_idx-lookback):current_idx]
        
        # Check if price touched the level and then reversed
        for i in range(1, len(prices)):
            prev_price = prices.iloc[i-1]
            curr_price = prices.iloc[i]
            level_price = level_values.iloc[i]
            
            # Check if price crossed the level
            if target_level.startswith('fbb_up'):
                # For upper levels, check if price went above and then below
                if prev_price <= level_price and curr_price > level_price:
                    touch_count += 1
                    # Check if it reversed (went back down) within next 5 days
                    if i + 5 < len(prices):
                        future_prices = prices.iloc[i:i+6]
                        if future_prices.min() < level_price:
                            reversal_count += 1
            else:
                # For lower levels, check if price went below and then above
                if prev_price >= level_price and curr_price < level_price:
                    touch_count += 1
                    # Check if it reversed (went back up) within next 5 days
                    if i + 5 < len(prices):
                        future_prices = prices.iloc[i:i+6]
                        if future_prices.max() > level_price:
                            reversal_count += 1
        
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
        target_level = None
        target_price = None
        days_to_touch = None
        
        if direction == 'up':
            # Check upper levels (in order from closest to farthest)
            upper_levels = ['fbb_up1', 'fbb_up2', 'fbb_up3', 'fbb_up4', 'fbb_up5', 'fbb_up6']
            for level in upper_levels:
                if level in fbb_levels:
                    level_price = fbb_levels[level]
                    if level_price > current_price:
                        distance = level_price - current_price
                        if velocity > 0:
                            days_needed = distance / velocity
                            if days_needed > 0 and days_needed <= max_days_ahead:
                                target_level = level
                                target_price = level_price
                                days_to_touch = int(np.ceil(days_needed))
                                break
            
            if target_level is None:
                # Price is already above all upper bands — predict reversal back
                # to the nearest upper band below current price (closest first)
                for level in reversed(upper_levels):
                    if level in fbb_levels:
                        level_price = fbb_levels[level]
                        if level_price <= current_price:
                            distance = current_price - level_price
                            if velocity > 0:
                                days_needed = distance / velocity
                            else:
                                days_needed = distance / abs(velocity) if velocity != 0 else max_days_ahead
                            if days_needed > 0 and days_needed <= max_days_ahead:
                                target_level = level
                                target_price = level_price
                                days_to_touch = int(np.ceil(days_needed))
                                direction = 'reversal_down'
                                break
        else:
            # Check lower levels (in order from closest to farthest)
            lower_levels = ['fbb_low1', 'fbb_low2', 'fbb_low3', 'fbb_low4', 'fbb_low5', 'fbb_low6']
            for level in lower_levels:
                if level in fbb_levels:
                    level_price = fbb_levels[level]
                    if level_price < current_price:
                        distance = current_price - level_price
                        if velocity < 0:
                            days_needed = abs(distance / velocity)
                            if days_needed > 0 and days_needed <= max_days_ahead:
                                target_level = level
                                target_price = level_price
                                days_to_touch = int(np.ceil(days_needed))
                                break
            
            if target_level is None:
                # Price is already below all lower bands — predict reversal back
                # to the nearest lower band above current price (closest first)
                for level in reversed(lower_levels):
                    if level in fbb_levels:
                        level_price = fbb_levels[level]
                        if level_price >= current_price:
                            distance = level_price - current_price
                            if velocity < 0:
                                days_needed = distance / abs(velocity)
                            else:
                                days_needed = distance / velocity if velocity != 0 else max_days_ahead
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
        
        # Signal logic: 
        # - Full buy (1.0) when price touches or crosses fbb_low6 (extreme lower band)
        # - Partial buy (0.5) when price touches or crosses fbb_low5 (but not fbb_low6)
        # - Partial sell (-0.5) when price touches or crosses fbb_up6
        # Priority: Full buy > Partial buy, so check fbb_low6 first
        for i in range(1, len(df)):
            close = df['close'].iloc[i]
            prev_close = df['close'].iloc[i-1]
            
            # Check if price touched or crossed fbb_low6 (full buy signal - highest priority)
            if 'fbb_low6' in df.columns:
                low6 = df['fbb_low6'].iloc[i]
                low6_prev = df['fbb_low6'].iloc[i-1] if i > 0 else low6
                if not pd.isna(low6):
                    # Price is at or below fbb_low6, or crossed from above
                    if close <= low6 or (prev_close > low6_prev and close <= low6):
                        df.loc[df.index[i], 'fbb_signal'] = 1.0  # Full buy
                        continue  # Skip other checks if full buy signal
            
            # Check if price touched or crossed fbb_low5 (partial buy signal)
            # Only set if not already a full buy
            if 'fbb_low5' in df.columns:
                low5 = df['fbb_low5'].iloc[i]
                low5_prev = df['fbb_low5'].iloc[i-1] if i > 0 else low5
                if not pd.isna(low5):
                    # Price is at or below fbb_low5, or crossed from above
                    if close <= low5 or (prev_close > low5_prev and close <= low5):
                        # Only set partial buy if not already a full buy
                        if df.loc[df.index[i], 'fbb_signal'] == 0.0:
                            df.loc[df.index[i], 'fbb_signal'] = 0.5  # Partial buy
            
            # Check if price touched or crossed fbb_up6 (partial sell signal)
            # This check is independent - price can't be at both levels simultaneously
            if 'fbb_up6' in df.columns:
                up6 = df['fbb_up6'].iloc[i]
                up6_prev = df['fbb_up6'].iloc[i-1] if i > 0 else up6
                if not pd.isna(up6):
                    # Price is at or above fbb_up6, or crossed from below
                    if close >= up6 or (prev_close < up6_prev and close >= up6):
                        df.loc[df.index[i], 'fbb_signal'] = -0.5  # Partial sell
        
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