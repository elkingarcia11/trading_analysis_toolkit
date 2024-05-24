import pandas as pd
import numpy as np

def fib_levels_strategy(data):
    # Initialize variables
    profits = {'Fib_0.236': 0, 'Fib_0.382': 0, 'Fib_0.5': 0, 'Fib_0.618': 0, 'Fib_0.786': 0}
    positions = {level: None for level in profits.keys()}

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        for level in profits.keys():
            fib_level = row[level]

            # Buy signal
            if row['Close'] > fib_level and positions[level] != 'BUY':
                if positions[level] == 'SELL':
                    profits[level] += row['Close'] - entry_price
                positions[level] = 'BUY'
                entry_price = row['Close']

            # Sell signal
            elif row['Close'] < fib_level and positions[level] != 'SELL':
                if positions[level] == 'BUY':
                    profits[level] += entry_price - row['Close']
                positions[level] = 'SELL'
                entry_price = row['Close']

    # Calculate final profits
    final_profits = {level: 0 for level in profits.keys()}
    for level, position in positions.items():
        if position == 'BUY':
            final_profits[level] += data.iloc[-1]['Close'] - entry_price
        elif position == 'SELL':
            final_profits[level] += entry_price - data.iloc[-1]['Close']

    # Determine the most profitable Fib level
    most_profitable_level = max(final_profits, key=final_profits.get)

    return most_profitable_level, final_profits[most_profitable_level]

def pivot_point_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        pivot_point = row['Pivot_Point']
        support1 = row['Support1']
        resistance1 = row['Resistance1']

        # Buy signal: Closing price crosses above the pivot point
        if row['Close'] > pivot_point and position != 'BUY':
            if position == 'SELL':
                profit += row['Close'] - entry_price
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: Closing price crosses below the pivot point
        elif row['Close'] < pivot_point and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - row['Close']
            position = 'SELL'
            entry_price = row['Close']

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def support_resistance_strategy(data):
    # Initialize variables
    profits = {'Support1': 0, 'Resistance1': 0}
    positions = {level: None for level in profits.keys()}

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support1_level = row['Support1']
        resistance1_level = row['Resistance1']

        # Buy signal based on Support 1 level
        if row['Close'] > support1_level and positions['Support1'] != 'BUY':
            if positions['Support1'] == 'SELL':
                profits['Support1'] += row['Close'] - entry_price
            positions['Support1'] = 'BUY'
            entry_price = row['Close']

        # Sell signal based on Support 1 level
        elif row['Close'] < support1_level and positions['Support1'] != 'SELL':
            if positions['Support1'] == 'BUY':
                profits['Support1'] += entry_price - row['Close']
            positions['Support1'] = 'SELL'
            entry_price = row['Close']

        # Buy signal based on Resistance 1 level
        if row['Close'] > resistance1_level and positions['Resistance1'] != 'BUY':
            if positions['Resistance1'] == 'SELL':
                profits['Resistance1'] += row['Close'] - entry_price
            positions['Resistance1'] = 'BUY'
            entry_price = row['Close']

        # Sell signal based on Resistance 1 level
        elif row['Close'] < resistance1_level and positions['Resistance1'] != 'SELL':
            if positions['Resistance1'] == 'BUY':
                profits['Resistance1'] += entry_price - row['Close']
            positions['Resistance1'] = 'SELL'
            entry_price = row['Close']

    # Calculate final profit
    final_profit = sum(profits.values())

    return final_profit

def vwap_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        vwap = row['VWAP']

        # Buy signal: Closing price crosses below VWAP
        if row['Close'] < vwap and position != 'BUY':
            if position == 'SELL':
                profit += entry_price - row['Close']
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: Closing price crosses above VWAP
        elif row['Close'] > vwap and position != 'SELL':
            if position == 'BUY':
                profit += row['Close'] - entry_price
            position = 'SELL'
            entry_price = row['Close']

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def cmf_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        cmf = row['CMF']

        # Buy signal: CMF crosses above 0
        if cmf > 0 and position != 'BUY':
            if position == 'SELL':
                profit += entry_price - row['Close']
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: CMF crosses below 0
        elif cmf < 0 and position != 'SELL':
            if position == 'BUY':
                profit += row['Close'] - entry_price
            position = 'SELL'
            entry_price = row['Close']

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def obv_strategy(data):
    # Initialize variables
    profit = 0
    position = None
    obv_prev = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        obv = row['OBV']

        # Initial case
        if obv_prev is None:
            obv_prev = obv
            continue

        # Buy signal: OBV increases
        if obv > obv_prev and position != 'BUY':
            if position == 'SELL':
                profit += row['Close'] - entry_price
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: OBV decreases
        elif obv < obv_prev and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - row['Close']
            position = 'SELL'
            entry_price = row['Close']

        obv_prev = obv

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def atr_strategy(data, atr_multiplier=1):
    # Initialize variables
    profit = 0
    position = None
    atr_prev = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        atr = row['ATR']
        close = row['Close']

        # Initial case
        if atr_prev is None:
            atr_prev = atr
            continue

        # Buy signal: Closing price crosses above previous high + ATR
        if close > (row['High'] - atr_multiplier * atr) and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below previous low - ATR
        elif close < (row['Low'] + atr_multiplier * atr) and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

        atr_prev = atr

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def lower_bollinger_band_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Lower Bollinger Band
        if close > lower_band and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def upper_bollinger_band_strategy(data):
    # Initialize variables
    profit = 0
    position = None
    entry_price = 0  # Initialize entry price

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        upper_band = row['Upper Band']
        close = row['Close']

        # Handle initial position
        if position is None:
            if close > upper_band:
                position = 'BUY'
                entry_price = close
            elif close < upper_band:
                position = 'SELL'
                entry_price = close

        # Buy signal: Closing price crosses above Upper Bollinger Band
        elif close > upper_band and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Upper Bollinger Band
        elif close < upper_band and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit


def bollinger_bands_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        upper_band = row['Upper Band']
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Upper Bollinger Band
        if close > upper_band and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def psar_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        psar = row['PSAR']
        close = row['Close']

        # Buy signal: Closing price crosses above PSAR in an uptrend
        if close > psar and psar > 0 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below PSAR in a downtrend
        elif close < psar and psar < 0 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def adx_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        adx = row['ADX']
        adx_threshold = 25  # Adjust the threshold as needed
        
        # Check if ADX is above the threshold
        if adx > adx_threshold:
            # If ADX is above the threshold, consider taking trades based on other indicators
            # For example, you could use other indicators like moving averages or price action to trigger buy/sell signals
            
            # In this example, let's use a simple moving average crossover strategy
            # You can replace this with your preferred strategy
            
            # Calculate moving averages
            sma_20 = row['SMA_20']
            sma_50 = row['SMA_50']
            
            # Buy signal: 20-day SMA crosses above 50-day SMA
            if sma_20 > sma_50 and position != 'BUY':
                if position == 'SELL':
                    profit += row['Close'] - entry_price
                position = 'BUY'
                entry_price = row['Close']

            # Sell signal: 20-day SMA crosses below 50-day SMA
            elif sma_20 < sma_50 and position != 'SELL':
                if position == 'BUY':
                    profit += entry_price - row['Close']
                position = 'SELL'
                entry_price = row['Close']

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def macd_signal_line_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        macd_line = row['MACD']
        signal_line = row['Signal_Line']
        close = row['Close']

        # Buy signal: MACD line crosses above Signal Line
        if macd_line > signal_line and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: MACD line crosses below Signal Line
        elif macd_line < signal_line and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def macd_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        macd_line = row['MACD']
        close = row['Close']

        # Buy signal: MACD line crosses above 0
        if macd_line > 0 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: MACD line crosses below 0
        elif macd_line < 0 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def momentum_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        momentum = row['Momentum']
        close = row['Close']

        # Buy signal: Momentum is positive
        if momentum > 0 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Momentum is negative
        elif momentum < 0 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def roc_strategy(data, roc_period=14, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Rate of Change (ROC)
    data['ROC'] = (data['Close'].diff(roc_period) / data['Close'].shift(roc_period)) * 100

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        roc = row['ROC']
        close = row['Close']

        # Buy signal: ROC crosses above the buy threshold
        if roc > buy_threshold and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: ROC crosses below the sell threshold
        elif roc < sell_threshold and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit


def stochastic_d_strategy(data, k_period=14, d_period=3, buy_threshold=20, sell_threshold=80):
    # Initialize variables
    position = None
    profit = 0

    # Calculate %K and %D lines for Stochastic Oscillator
    data['%K'] = ((data['Close'] - data['Low'].rolling(window=k_period).min()) /
                  (data['High'].rolling(window=k_period).max() - data['Low'].rolling(window=k_period).min())) * 100
    data['%D'] = data['%K'].rolling(window=d_period).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        percent_d = row['%D']
        close = row['Close']

        # Buy signal: %D line crosses below buy threshold (oversold condition)
        if percent_d < buy_threshold and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: %D line crosses above sell threshold (overbought condition)
        elif percent_d > sell_threshold and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def stochastic_k_strategy(data, k_period=14, buy_threshold=20, sell_threshold=80):
    # Initialize variables
    position = None
    profit = 0

    # Calculate %K line for Stochastic Oscillator
    data['%K'] = ((data['Close'] - data['Low'].rolling(window=k_period).min()) /
                  (data['High'].rolling(window=k_period).max() - data['Low'].rolling(window=k_period).min())) * 100

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        percent_k = row['%K']
        close = row['Close']

        # Buy signal: %K line crosses above buy threshold
        if percent_k < buy_threshold and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: %K line crosses below sell threshold
        elif percent_k > sell_threshold and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def wma_200_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Weighted Moving Average (WMA) with a period of 200
    data['WMA_200'] = data['Close'].rolling(window=200).apply(lambda x: np.dot(x, np.arange(1,201)) / 201, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_200 = row['WMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_200
        if close > wma_200 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_200
        elif close < wma_200 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def ema_200_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Exponential Moving Average (EMA) with a period of 200
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_200 = row['EMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_200
        if close > ema_200 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_200
        elif close < ema_200 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def sma_200_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Simple Moving Average (SMA) with a period of 200
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_200 = row['SMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_200
        if close > sma_200 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_200
        elif close < sma_200 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def wma_50_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Weighted Moving Average (WMA) with a period of 50
    data['WMA_50'] = data['Close'].rolling(window=50).apply(lambda x: np.dot(x, np.arange(1, 51)) / 1275, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_50 = row['WMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_50
        if close > wma_50 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_50
        elif close < wma_50 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def ema_50_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Exponential Moving Average (EMA) with a period of 50
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_50 = row['EMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_50
        if close > ema_50 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_50
        elif close < ema_50 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def sma_50_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Simple Moving Average (SMA) with a period of 50
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_50 = row['SMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_50
        if close > sma_50 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_50
        elif close < sma_50 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def wma_20_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Weighted Moving Average (WMA) with a period of 20
    data['WMA_20'] = data['Close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21)) / 210, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_20 = row['WMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_20
        if close > wma_20 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_20
        elif close < wma_20 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def ema_20_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Exponential Moving Average (EMA) with a period of 20
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_20 = row['EMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_20
        if close > ema_20 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_20
        elif close < ema_20 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def sma_20_strategy(data, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Simple Moving Average (SMA) with a period of 20
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_20 = row['SMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_20
        if close > sma_20 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_20
        elif close < sma_20 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def rsi_strategy(data, buy_threshold=30, sell_threshold=70):
    # Initialize variables
    position = None
    profit = 0
    entry_price = 0

    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    RS = gain / loss
    RSI = 100 - (100 / (1 + RS))

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        rsi = RSI.loc[index]
        close = row['Close']

        # Buy signal: RSI crosses below the buy threshold
        if rsi < buy_threshold and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: RSI crosses above the sell threshold
        elif rsi > sell_threshold and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def lagging_span_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        lagging_span = row['Lagging_Span']
        close = row['Close']

        # Buy signal: Closing price crosses above Lagging Span
        if close > lagging_span and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lagging Span
        elif close < lagging_span and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def leading_span_b_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        leading_span_b = row['Leading_Span_B']
        close = row['Close']

        # Buy signal: Closing price crosses above Leading Span B
        if close > leading_span_b and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Leading Span B
        elif close < leading_span_b and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def base_line_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        base_line = row['Base_Line']
        close = row['Close']

        # Buy signal: Closing price crosses above Base Line
        if close > base_line and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Base Line
        elif close < base_line and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def conversion_line_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        conversion_line = row['Conversion_Line']
        close = row['Close']

        # Buy signal: Closing price crosses above Conversion Line
        if close > conversion_line and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Conversion Line
        elif close < conversion_line and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def support_resistance_3_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support3 = row['Support3']
        resistance3 = row['Resistance3']
        close = row['Close']

        # Buy signal: Closing price crosses above Support 3
        if close > support3 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Resistance 3
        elif close < resistance3 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def support_resistance_2_strategy(data):
    # Initialize variables
    position = None
    profit = 0

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support2 = row['Support2']
        resistance2 = row['Resistance2']
        close = row['Close']

        # Buy signal: Closing price crosses above Support 2
        if close > support2 and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Resistance 2
        elif close < resistance2 and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit
def lagging_span_sma_strategy(data, lagging_span_period=26):
    # Initialize variables
    position = None
    profit = 0

    # Calculate Lagging Span
    data['Lagging_Span'] = data['Close'].shift(-lagging_span_period)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        lagging_span = row['Lagging_Span']
        sma = row['SMA_20']
        close = row['Close']

        # Buy signal: Lagging Span crosses above SMA
        if lagging_span > sma and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Lagging Span crosses below SMA
        elif lagging_span < sma and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def bollinger_band_strategy(data):
    # Initialize variables
    profit = 0
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        upper_band = row['Upper Band']
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Upper Bollinger Band
        if close > upper_band and position != 'BUY':
            if position == 'SELL':
                profit += close - entry_price
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                profit += entry_price - close
            position = 'SELL'
            entry_price = close

    # If position is still open at the end, close it
    if position == 'BUY':
        profit += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        profit += entry_price - data.iloc[-1]['Close']

    return profit

def holdThenSell(data):
    # Get the prices on the first and last day
    first_day_price = data.iloc[0]['Close']
    last_day_price = data.iloc[-1]['Open']

    # Calculate profit
    profit = last_day_price - first_day_price

    return profit

def findBestIndicator(ticker_symbol):
    # Load your dataset
    data = pd.read_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv')

    max_profit = float('-inf')  # Initialize max profit to negative infinity

    most_profitable_level, profit = fib_levels_strategy(data)
    print("Most profitable Fib level:", most_profitable_level)
    print("Profit:", profit)
    max_profit=max(max_profit,profit)

    profit = pivot_point_strategy(data)
    print("Total profit using Pivot Point strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = support_resistance_strategy(data)
    print("Profit using Support 1 and Resistance 1 levels strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = vwap_strategy(data)
    print("Profit using VWAP strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = cmf_strategy(data)
    print("Profit using CMF strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = obv_strategy(data)
    print("Profit using OBV strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = atr_strategy(data, atr_multiplier=1.5)  # You can adjust the multiplier as needed
    print("Profit using ATR strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = lower_bollinger_band_strategy(data)
    print("Profit using Lower Bollinger Band strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = upper_bollinger_band_strategy(data)
    print("Profit using Upper Bollinger Band strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = bollinger_bands_strategy(data)
    print("Profit using Bollinger Bands strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = psar_strategy(data)
    print("Profit using PSAR strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = adx_strategy(data)
    print("Profit using ADX strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = macd_signal_line_strategy(data)
    print("Profit using MACD with Signal Line strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = macd_strategy(data)
    print("Profit using MACD strategy without Signal Line:", profit)
    max_profit=max(max_profit,profit)

    profit = momentum_strategy(data)
    print("Profit using Momentum strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = roc_strategy(data, roc_period=14, buy_threshold=2, sell_threshold=-2)
    print("Profit using ROC strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = stochastic_d_strategy(data, k_period=14, d_period=3, buy_threshold=20, sell_threshold=80)
    print("Profit using Stochastic %D and %K strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = stochastic_k_strategy(data, k_period=14, buy_threshold=20, sell_threshold=80)
    print("Profit using Stochastic strategy (just %K):", profit)
    max_profit=max(max_profit,profit)

    profit = wma_200_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using WMA_200 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = ema_200_strategy(data, buy_threshold=0, sell_threshold=0) 
    print("Profit using EMA_200 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = sma_200_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using SMA_200 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = wma_50_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using WMA_50 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = ema_50_strategy(data, buy_threshold=0, sell_threshold=0) 
    print("Profit using EMA_50 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = sma_50_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using SMA_50 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = wma_20_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using WMA_20 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = ema_20_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using EMA_20 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = sma_20_strategy(data, buy_threshold=0, sell_threshold=0)
    print("Profit using SMA_20 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = rsi_strategy(data, buy_threshold=30, sell_threshold=70)
    print("Profit using RSI strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = lagging_span_strategy(data)
    print("Profit using Lagging Span strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = leading_span_b_strategy(data)
    print("Profit using Leading Span B strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = base_line_strategy(data)
    print("Profit using Base Line strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = conversion_line_strategy(data)
    print("Profit using Conversion Line strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = support_resistance_3_strategy(data)
    print("Profit using Support/Resistance 3 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = support_resistance_2_strategy(data)
    print("Profit using Support/Resistance 2 strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = lagging_span_sma_strategy(data)
    print("Profit using Lagging Span with SMA strategy:", profit)
    max_profit=max(max_profit,profit)

    profit = bollinger_band_strategy(data)
    print("Profit from Bollinger Band strategy:", profit)

    profit = holdThenSell(data)
    print("Profit from holding then selling:", profit)

    print("Max profit:", max(max_profit,profit))

ticker_symbol = 'TQQQ'
findBestIndicator(ticker_symbol)
ticker_symbol = 'BTC'
findBestIndicator(ticker_symbol)