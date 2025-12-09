import pandas as pd
import plotly.offline as py_offline
import plotly.graph_objs as go
import pandas_datareader as web
import datetime as datetime
import yfinance as yf
import numpy as np

def fibonacci_bollinger_bands(df, n=20, m=3):
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    ma = tp.rolling(n).mean()
    sd = m * tp.rolling(n).std()
    df['FBB_mid'] = ma
    df['FBB_up1'] = ma + (0.236 * sd)
    df['FBB_up2'] = ma + (0.382 * sd)
    df['FBB_up3'] = ma + (0.5 * sd)
    df['FBB_up4'] = ma + (0.618 * sd)
    df['FBB_up5'] = ma + (0.764 * sd)
    df['FBB_up6'] = ma + (1 * sd)
    df['FBB_low1'] = ma - (0.236 * sd)
    df['FBB_low2'] = ma - (0.382 * sd)
    df['FBB_low3'] = ma - (0.5 * sd)
    df['FBB_low4'] = ma - (0.618 * sd)
    df['FBB_low5'] = ma - (0.764 * sd)
    df['FBB_low6'] = ma - (1 * sd)
    return df


def predict_fbb_touch_and_reversal(df, lookback_period=20, max_days_ahead=60):
    """
    Predict which Fibonacci Bollinger Band level the price will touch and when,
    before it reverses.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with OHLC data and FBB columns
    lookback_period : int
        Number of days to look back for velocity calculation
    max_days_ahead : int
        Maximum number of days to project forward
    
    Returns:
    --------
    dict : Prediction results with:
        - target_level: Which FBB level will be touched (e.g., 'FBB_up6', 'FBB_low6')
        - target_price: Price level to be touched
        - predicted_date: Estimated date when level will be touched
        - days_to_touch: Number of days until touch
        - current_price: Current closing price
        - direction: 'up' or 'down'
        - reversal_probability: Probability of reversal after touch (0-1)
    """
    if len(df) < lookback_period:
        return None
    
    # Get current values
    current_idx = len(df) - 1
    current_price = df['Close'].iloc[current_idx]
    
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
    for level in ['FBB_up6', 'FBB_up5', 'FBB_up4', 'FBB_up3', 'FBB_up2', 'FBB_up1', 
                  'FBB_mid', 'FBB_low1', 'FBB_low2', 'FBB_low3', 'FBB_low4', 'FBB_low5', 'FBB_low6']:
        value = df[level].iloc[current_idx]
        if not pd.isna(value):
            fbb_levels[level] = value
    
    # Calculate price velocity (momentum)
    recent_prices = df['Close'].iloc[-lookback_period:].values
    recent_dates = np.arange(len(recent_prices))
    
    # Linear regression to get velocity (price change per day)
    if len(recent_prices) > 1:
        # Calculate average daily change
        price_changes = np.diff(recent_prices)
        velocity = np.mean(price_changes)  # Average price change per day
        
        # Alternative: Use linear regression for more accurate velocity
        if len(recent_prices) >= 2:
            slope = np.polyfit(recent_dates, recent_prices, 1)[0]
            velocity = slope
    else:
        velocity = 0
    
    # Determine direction
    direction = 'up' if velocity > 0 else 'down'
    
    # Find which level will be touched first
    target_level = None
    target_price = None
    days_to_touch = None
    
    if direction == 'up':
        # Check upper levels (in order from closest to farthest)
        upper_levels = ['FBB_up1', 'FBB_up2', 'FBB_up3', 'FBB_up4', 'FBB_up5', 'FBB_up6']
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
    else:
        # Check lower levels (in order from closest to farthest)
        lower_levels = ['FBB_low1', 'FBB_low2', 'FBB_low3', 'FBB_low4', 'FBB_low5', 'FBB_low6']
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
    
    # Calculate reversal probability based on historical patterns
    reversal_probability = calculate_reversal_probability(df, target_level, current_idx)
    
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


def calculate_reversal_probability(df, target_level, current_idx, lookback=50):
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
    prices = df['Close'].iloc[max(0, current_idx-lookback):current_idx]
    
    # Check if price touched the level and then reversed
    for i in range(1, len(prices)):
        prev_price = prices.iloc[i-1]
        curr_price = prices.iloc[i]
        level_price = level_values.iloc[i]
        
        # Check if price crossed the level
        if target_level.startswith('FBB_up'):
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


def plot_fbb(df, fname):
    index = list(range(len(df['Close'])))
    trace = go.Candlestick(
                           x=index,
                           open=df['Open'].values,
                           high=df['High'].values,
                           low=df['Low'].values,
                           close=df['Close'].values,
                           name="OHLC",
                           increasing_line_color='green',
                           decreasing_line_color='red')

    ls_up = dict(
        color='rgb(255, 0, 0, 0.5)'
    )
    ls_mid = dict(
        color='rgb(255,20,147, 0.5)'
    )
    ls_low = dict(
        color='rgb(34,139,34, 0.5)'
    )
    ls_fib = dict(
        color='rgb(169,169,169,0.5)',
        width=1
    )

    t_mid = go.Scatter(x=index, y=df.FBB_mid, line=ls_mid, name="Middle Band")
    t_up1 = go.Scatter(x=index, y=df.FBB_up1, line=ls_fib, showlegend=False, hoverinfo='none')
    t_up2 = go.Scatter(x=index, y=df.FBB_up2, line=ls_fib, showlegend=False, hoverinfo='none')
    t_up3 = go.Scatter(x=index, y=df.FBB_up3, line=ls_fib, showlegend=False, hoverinfo='none')
    t_up4 = go.Scatter(x=index, y=df.FBB_up4, line=ls_fib, showlegend=False, hoverinfo='none')
    t_up5 = go.Scatter(x=index, y=df.FBB_up5, line=ls_fib, showlegend=False, hoverinfo='none')
    t_up6 = go.Scatter(x=index, y=df.FBB_up6, line=ls_up, name="Upper Band")
    t_low1 = go.Scatter(x=index, y=df.FBB_low1, line=ls_fib, showlegend=False, hoverinfo='none')
    t_low2 = go.Scatter(x=index, y=df.FBB_low2, line=ls_fib, showlegend=False, hoverinfo='none')
    t_low3 = go.Scatter(x=index, y=df.FBB_low3, line=ls_fib, showlegend=False, hoverinfo='none')
    t_low4 = go.Scatter(x=index, y=df.FBB_low4, line=ls_fib, showlegend=False, hoverinfo='none')
    t_low5 = go.Scatter(x=index, y=df.FBB_low5, line=ls_fib, showlegend=False, hoverinfo='none')
    t_low6 = go.Scatter(x=index, y=df.FBB_low6, line=ls_low, name="Lower Band")

    layout = go.Layout(
        title='OHLC with Fibonacci Bands',
        xaxis=dict(
            tickmode='array',
            nticks=90,
            tickvals=index,
            ticktext=[str(d) for d in df.index]),
        yaxis=dict(
            title='Price'))

    # Put candlestick first to ensure it's visible
    data = [trace,
            t_mid, t_up1, t_up2, t_up3, t_up4, t_up5, t_up6,
            t_low1, t_low2, t_low3, t_low4, t_low5, t_low6]
    fig = go.Figure(data=data, layout=layout)
    py_offline.plot(fig, filename=fname+".html")


if __name__ == '__main__':
    today = datetime.datetime.now().date()
    print(today)
    # Subtract 365 days from today's date
    one_year_ago = today - datetime.timedelta(days=1000)
    print(one_year_ago)

    df = yf.download('TATAPOWER.NS', start=one_year_ago, end=today, auto_adjust=False)
    # Reset index if it's a DatetimeIndex to make it easier to work with
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.copy()

    # Handle yfinance MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    column_mapping = {}
    for col in df.columns:
        if isinstance(col, str):
            col_lower = col.lower()
            if col_lower == 'open':
                column_mapping[col] = 'Open'
            elif col_lower == 'high':
                column_mapping[col] = 'High'
            elif col_lower == 'low':
                column_mapping[col] = 'Low'
            elif col_lower == 'close':
                column_mapping[col] = 'Close'
    if column_mapping:
        df = df.rename(columns=column_mapping)

    fibonacci_bollinger_bands(df, 20, 3)

    # Predict FBB touch and reversal
    prediction = predict_fbb_touch_and_reversal(df, lookback_period=20, max_days_ahead=60)
    
    if prediction:
        print("\n" + "="*60)
        print("FIBONACCI BOLLINGER BAND PREDICTION")
        print("="*60)
        print(f"Current Price: ${prediction['current_price']:.2f}")
        print(f"Current Date: {prediction['current_date']}")
        print(f"Direction: {prediction['direction'].upper()}")
        print(f"Price Velocity: ${prediction['velocity']:.2f} per day")
        print("-"*60)
        
        if prediction['target_level']:
            print(f"Target Level: {prediction['target_level']}")
            print(f"Target Price: ${prediction['target_price']:.2f}")
            print(f"Days to Touch: {prediction['days_to_touch']} days")
            print(f"Predicted Date: {prediction['predicted_date']}")
            print(f"Reversal Probability: {prediction['reversal_probability']:.1%}")
        else:
            print("No FBB level will be touched within the prediction window.")
            print(f"Current price is between levels.")
        
        print("\nAll FBB Levels:")
        for level, price in sorted(prediction['all_levels'].items(), key=lambda x: x[1], reverse=True):
            if not pd.isna(price):
                distance = price - prediction['current_price']
                pct = (distance / prediction['current_price']) * 100
                marker = " <-- CURRENT" if abs(distance) < 0.01 else ""
                print(f"  {level:12s}: ${price:8.2f} ({pct:+.2f}%){marker}")
        
        print("="*60)
    else:
        print("Insufficient data for prediction.")

   # plot_fbb(df, "Test")