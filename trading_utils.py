import pandas as pd
import numpy as np

"""
File for back testing algorithms
"""

def fib_levels_strategy(data):
    # Initialize variables
    budgets = {'Fib_0.236': 10000, 'Fib_0.382': 10000, 'Fib_0.5': 10000, 'Fib_0.618': 10000, 'Fib_0.786': 10000}
    positions = {level: None for level in budgets.keys()}

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        for level in budgets.keys():
            fib_level = row[level]

            # Buy signal
            if row['Close'] > fib_level and positions[level] != 'BUY':
                if positions[level] == 'SELL':
                    budgets[level] += row['Close'] * quantity
                budgets[level], quantity = 0, budgets[level] / row['Close']  # Use all available budget
                positions[level] = 'BUY'
            # Sell signal
            elif row['Close'] < fib_level and positions[level] != 'SELL':
                if positions[level] == 'BUY':
                    budgets[level] = row['Close'] * quantity
                quantity, positions[level] = 0, 'SELL'

    # Find the most profitable Fibonacci level
    most_profitable_level = max(budgets, key=budgets.get)
    max_budget = budgets[most_profitable_level]

    return most_profitable_level, max_budget


def pivot_point_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        pivot_point = row['Pivot_Point']

        # Buy signal: Closing price crosses above the pivot point
        if row['Close'] > pivot_point and position != 'BUY':
            if position == 'SELL':
                budget += row['Close'] * quantity
            budget, quantity = 0, budget / row['Close']  # Use all available budget
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: Closing price crosses below the pivot point
        elif row['Close'] < pivot_point and position != 'SELL':
            if position == 'BUY':
                budget = row['Close'] * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def support_resistance_strategy(data, budget=10000):
    # Initialize variables
    budgets = {'Support1': budget, 'Resistance1': budget}
    positions = {level: None for level in budgets.keys()}
    entry_prices = {level: None for level in budgets.keys()}
    quantities = {level: 0 for level in budgets.keys()}

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support1_level = row['Support1']
        resistance1_level = row['Resistance1']
        close = row['Close']

        # Buy signal based on Support 1 level
        if close > support1_level and positions['Support1'] != 'BUY':
            if positions['Support1'] == 'SELL' and entry_prices['Support1'] is not None:
                budgets['Support1'] += close * quantities['Support1'] - entry_prices['Support1'] * quantities['Support1']
            quantities['Support1'] = budgets['Support1'] / close  # Use all available budget
            budgets['Support1'] = 0
            positions['Support1'] = 'BUY'
            entry_prices['Support1'] = close

        # Sell signal based on Support 1 level
        elif close < support1_level and positions['Support1'] != 'SELL':
            if positions['Support1'] == 'BUY' and entry_prices['Support1'] is not None:
                budgets['Support1'] = close * quantities['Support1']
            quantities['Support1'] = 0
            positions['Support1'] = 'SELL'
            entry_prices['Support1'] = None

        # Buy signal based on Resistance 1 level
        if close > resistance1_level and positions['Resistance1'] != 'BUY':
            if positions['Resistance1'] == 'SELL' and entry_prices['Resistance1'] is not None:
                budgets['Resistance1'] += close * quantities['Resistance1'] - entry_prices['Resistance1'] * quantities['Resistance1']
            quantities['Resistance1'] = budgets['Resistance1'] / close  # Use all available budget
            budgets['Resistance1'] = 0
            positions['Resistance1'] = 'BUY'
            entry_prices['Resistance1'] = close

        # Sell signal based on Resistance 1 level
        elif close < resistance1_level and positions['Resistance1'] != 'SELL':
            if positions['Resistance1'] == 'BUY' and entry_prices['Resistance1'] is not None:
                budgets['Resistance1'] = close * quantities['Resistance1']
            quantities['Resistance1'] = 0
            positions['Resistance1'] = 'SELL'
            entry_prices['Resistance1'] = None

    # Calculate final budget
    final_budget = sum(budgets.values())

    return final_budget


def vwap_strategy(data, budget=10000):
    # Loop through each row in the dataset
    for index, row in data.iterrows():
        vwap = row['VWAP']

        # Buy signal: Closing price crosses below VWAP
        if row['Close'] < vwap and budget != 0:
            budget, quantity = 0, budget / row['Close']  # Use all available budget

        # Sell signal: Closing price crosses above VWAP
        elif row['Close'] > vwap and budget == 0:
            budget = row['Close'] * quantity
            quantity = 0

    # If position is still open at the end, close it
    if budget == 0:
        budget = data.iloc[-1]['Close'] * quantity

    return budget

def cmf_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        cmf = row['CMF']

        # Buy signal: CMF crosses above 0
        if cmf > 0 and position != 'BUY':
            if position == 'SELL':
                budget += row['Close'] * quantity
            budget, quantity = 0, budget / row['Close']  # Use all available budget
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: CMF crosses below 0
        elif cmf < 0 and position != 'SELL':
            if position == 'BUY':
                budget = row['Close'] * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def obv_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget
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
                budget += row['Close'] * quantity
            budget, quantity = 0, budget / row['Close']  # Use all available budget
            position = 'BUY'
            entry_price = row['Close']

        # Sell signal: OBV decreases
        elif obv < obv_prev and position != 'SELL':
            if position == 'BUY':
                budget = row['Close'] * quantity
            quantity, position = 0, 'SELL'

        obv_prev = obv

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def atr_strategy(data, budget=10000, atr_multiplier=1):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget
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
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below previous low - ATR
        elif close < (row['Low'] + atr_multiplier * atr) and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

        atr_prev = atr

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def lower_bollinger_band_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Lower Bollinger Band
        if close > lower_band and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def upper_bollinger_band_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = 0  # Initialize entry price
    budget = budget
    quantity = 0

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
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Upper Bollinger Band
        elif close < upper_band and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def bollinger_bands_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        upper_band = row['Upper Band']
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Upper Bollinger Band
        if close > upper_band and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def psar_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        psar = row['PSAR']
        close = row['Close']

        # Buy signal: Closing price crosses above PSAR in an uptrend
        if close > psar and psar > 0 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below PSAR in a downtrend
        elif close < psar and psar < 0 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def adx_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

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
            
            close = row['Close']

            # Buy signal: 20-day SMA crosses above 50-day SMA
            if sma_20 > sma_50 and position != 'BUY':
                if position == 'SELL':
                    budget += close * quantity
                budget, quantity = 0, budget / close  # Use all available budget
                position = 'BUY'
                entry_price = close

            # Sell signal: 20-day SMA crosses below 50-day SMA
            elif sma_20 < sma_50 and position != 'SELL':
                if position == 'BUY':
                    budget = close * quantity
                quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def macd_signal_line_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        macd_line = row['MACD']
        signal_line = row['Signal_Line']
        close = row['Close']

        # Buy signal: MACD line crosses above Signal Line
        if macd_line > signal_line and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: MACD line crosses below Signal Line
        elif macd_line < signal_line and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget
def macd_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        macd_line = row['MACD']
        close = row['Close']

        # Buy signal: MACD line crosses above 0
        if macd_line > 0 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: MACD line crosses below 0
        elif macd_line < 0 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def momentum_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    budget = budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        momentum = row['Momentum']
        close = row['Close']

        # Buy signal: Momentum is positive
        if momentum > 0 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Momentum is negative
        elif momentum < 0 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def roc_strategy(data, budget=10000, roc_period=14, buy_threshold=0, sell_threshold=0):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Rate of Change (ROC)
    data['ROC'] = (data['Close'].diff(roc_period) / data['Close'].shift(roc_period)) * 100

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        roc = row['ROC']
        close = row['Close']

        # Buy signal: ROC crosses above the buy threshold
        if roc > buy_threshold and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: ROC crosses below the sell threshold
        elif roc < sell_threshold and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def stochastic_d_strategy(data, budget=10000, k_period=14, d_period=3, buy_threshold=20, sell_threshold=80):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

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
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: %D line crosses above sell threshold (overbought condition)
        elif percent_d > sell_threshold and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def stochastic_k_strategy(data, budget=10000, k_period=14, buy_threshold=20, sell_threshold=80):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

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
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: %K line crosses below sell threshold
        elif percent_k > sell_threshold and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget
def wma_200_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0

    # Calculate Weighted Moving Average (WMA) with a period of 200
    data['WMA_200'] = data['Close'].rolling(window=200).apply(lambda x: np.dot(x, np.arange(1, 201)) / 201, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_200 = row['WMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_200
        if close > wma_200 and position != 'BUY':
            if position == 'SELL' and entry_price is not None:
                budget += close * quantity
            quantity = budget / close  # Use all available budget
            budget = 0
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_200
        elif close < wma_200 and position == 'BUY':
            if entry_price is not None:
                budget = close * quantity
            quantity = 0
            position = 'SELL'
            entry_price = None

    # If position is still open at the end, close it
    if position == 'BUY' and entry_price is not None:
        budget += data.iloc[-1]['Close'] * quantity

    return budget


def ema_200_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Exponential Moving Average (EMA) with a period of 200
    data['EMA_200'] = data['Close'].ewm(span=200, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_200 = row['EMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_200
        if close > ema_200 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_200
        elif close < ema_200 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def sma_200_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Simple Moving Average (SMA) with a period of 200
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_200 = row['SMA_200']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_200
        if close > sma_200 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_200
        elif close < sma_200 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def wma_50_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Weighted Moving Average (WMA) with a period of 50
    data['WMA_50'] = data['Close'].rolling(window=50).apply(lambda x: np.dot(x, np.arange(1, 51)) / 1275, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_50 = row['WMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_50
        if close > wma_50 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_50
        elif close < wma_50 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget
def ema_50_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Exponential Moving Average (EMA) with a period of 50
    data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_50 = row['EMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_50
        if close > ema_50 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_50
        elif close < ema_50 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def sma_50_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Simple Moving Average (SMA) with a period of 50
    data['SMA_50'] = data['Close'].rolling(window=50).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_50 = row['SMA_50']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_50
        if close > sma_50 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_50
        elif close < sma_50 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def wma_20_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Weighted Moving Average (WMA) with a period of 20
    data['WMA_20'] = data['Close'].rolling(window=20).apply(lambda x: np.dot(x, np.arange(1, 21)) / 210, raw=True)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        wma_20 = row['WMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the WMA_20
        if close > wma_20 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the WMA_20
        elif close < wma_20 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget


def ema_20_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Exponential Moving Average (EMA) with a period of 20
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        ema_20 = row['EMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the EMA_20
        if close > ema_20 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget / close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the EMA_20
        elif close < ema_20 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget
def sma_20_strategy(data, budget=10000):
    # Initialize variables
    position = None
    entry_price = None
    quantity = 0
    budget = budget

    # Calculate Simple Moving Average (SMA) with a period of 20
    data['SMA_20'] = data['Close'].rolling(window=20).mean()

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        sma_20 = row['SMA_20']
        close = row['Close']

        # Buy signal: Closing price crosses above the SMA_20
        if close > sma_20 and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
            budget, quantity = 0, budget // close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: Closing price crosses below the SMA_20
        elif close < sma_20 and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity, position = 0, 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def rsi_strategy(data, budget=10000, buy_threshold=30, sell_threshold=70):
    # Initialize variables
    position = None
    entry_price = 0
    quantity = 0
    budget = budget

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
                budget += close * quantity
            quantity = budget // close  # Use all available budget
            position = 'BUY'
            entry_price = close

        # Sell signal: RSI crosses above the sell threshold
        elif rsi > sell_threshold and position != 'SELL':
            if position == 'BUY':
                budget = close * quantity
            quantity = 0
            position = 'SELL'

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += entry_price * quantity

    return budget

def lagging_span_strategy(data, budget=1000):
    quantity = 0
    position = None

    for index, row in data.iterrows():
        lagging_span_val = row['Lagging_Span']
        close = row['Close']

        # Buy signal: Closing price crosses above Lagging Span
        if close > lagging_span_val and position != 'BUY':
            if position == 'SELL':
                budget += close * quantity
                quantity = 0
            quantity = budget // close
            budget -= close * quantity
            position = 'BUY'

        # Sell signal: Closing price crosses below Lagging Span
        elif close < lagging_span_val and position != 'SELL':
            if position == 'BUY':
                budget += close * quantity
                quantity = 0
            quantity = budget // close
            budget -= close * quantity
            position = 'SELL'

    # Close open position at the end
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] * quantity
    elif position == 'SELL':
        budget += data.iloc[-1]['Close'] * quantity

    return budget

def multiple_lagging_span_strategy(data, budget=10000):
    max_budgets = {}  # Dictionary to store max profit for each lagging span indicator
    max_budget = -1
    
    for lag_span in range(26):  # Iterate over lagging span indicators
        quantity = 0
        position = None

        for index, row in data.iterrows():
            lagging_span_val = row[f'Lagging_Span_{lag_span}']
            close = row['Close']

            # Buy signal: Closing price crosses above Lagging Span
            if close > lagging_span_val and position != 'BUY':
                if position == 'SELL':
                    budget += close * quantity
                    quantity = 0
                quantity = budget // close
                budget -= close * quantity
                position = 'BUY'

            # Sell signal: Closing price crosses below Lagging Span
            elif close < lagging_span_val and position != 'SELL':
                if position == 'BUY':
                    budget += close * quantity
                    quantity = 0
                quantity = budget // close
                budget -= close * quantity
                position = 'SELL'

        # Close open position at the end
        if position == 'BUY':
            budget += data.iloc[-1]['Close'] * quantity
        elif position == 'SELL':
            budget += data.iloc[-1]['Close'] * quantity

        # Update profits for this lagging span indicator
        max_budget = max(max_budget, budget)
        max_budgets[f'Lagging_Span_{lag_span}'] = {'Total': budget}

    print("Max Budget:", max_budget)
    print("Max Budgets for each lagging span:", max_budgets)
    return max_budgets

def leading_span_b_strategy(data, budget=10000):
    # Initialize variables
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        leading_span_b = row['Leading_Span_B']
        close = row['Close']

        # Buy signal: Closing price crosses above Leading Span B
        if close > leading_span_b and position != 'BUY':
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

        # Sell signal: Closing price crosses below Leading Span B
        elif close < leading_span_b and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget

def base_line_strategy(data, budget=10000):
    # Initialize variables
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        base_line = row['Base_Line']
        close = row['Close']

        # Buy signal: Closing price crosses above Base Line
        if close > base_line and position != 'BUY':
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

        # Sell signal: Closing price crosses below Base Line
        elif close < base_line and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget



def conversion_line_strategy(data, budget=10000):
    # Initialize variables
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        conversion_line = row['Conversion_Line']
        close = row['Close']

        # Buy signal: Closing price crosses above Conversion Line
        if close > conversion_line and position != 'BUY' and budget >= close:
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire available budget
            position = 'BUY'
            entry_price = close
            # Deduct budget for the current trade
            budget -= close

        # Sell signal: Closing price crosses below Conversion Line
        elif close < conversion_line and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire available budget
            position = 'SELL'
            entry_price = close
            # Deduct budget for the current trade
            budget -= close

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget


def support_resistance_3_strategy(data, budget=10000):
    # Initialize variables
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support3 = row['Support3']
        resistance3 = row['Resistance3']
        close = row['Close']

        # Buy signal: Closing price crosses above Support 3
        if close > support3 and position != 'BUY' and budget >= close:
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

        # Sell signal: Closing price crosses below Resistance 3
        elif close < resistance3 and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget


def support_resistance_2_strategy(data, budget=10000):
    # Initialize variables
    position = None

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        support2 = row['Support2']
        resistance2 = row['Resistance2']
        close = row['Close']

        # Buy signal: Closing price crosses above Support 2
        if close > support2 and position != 'BUY' and budget >= close:
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

        # Sell signal: Closing price crosses below Resistance 2
        elif close < resistance2 and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget

def lagging_span_sma_strategy(data, budget=10000, lagging_span_period=26):
    # Initialize variables
    position = None

    # Calculate Lagging Span
    data['Lagging_Span'] = data['Close'].shift(-lagging_span_period)

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        lagging_span = row['Lagging_Span']
        sma = row['SMA_20']
        close = row['Close']

        # Buy signal: Lagging Span crosses above SMA
        if lagging_span > sma and position != 'BUY' and budget >= close:
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

        # Sell signal: Lagging Span crosses below SMA
        elif lagging_span < sma and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget = 0

    # If position is still open at the end, close it
    if position == 'BUY':
        budget += data.iloc[-1]['Close'] - entry_price
    elif position == 'SELL':
        budget += entry_price - data.iloc[-1]['Close']

    return budget
def bollinger_band_strategy(data, initial_budget=10000):
    # Initialize variables
    position = None
    budget = initial_budget

    # Loop through each row in the dataset
    for index, row in data.iterrows():
        upper_band = row['Upper Band']
        lower_band = row['Lower Band']
        close = row['Close']

        # Buy signal: Closing price crosses above Upper Bollinger Band and enough budget is available
        if close > upper_band and position != 'BUY' and budget >= close:
            if position == 'SELL':
                # Increment budget from the previous trade
                budget += close - entry_price
            # Buy with the entire budget
            position = 'BUY'
            entry_price = close
            # Deduct budget for the current trade
            budget -= close

        # Sell signal: Closing price crosses below Lower Bollinger Band
        elif close < lower_band and position != 'SELL':
            if position == 'BUY':
                # Increment budget from the previous trade
                budget += entry_price - close
            # Sell with the entire budget
            position = 'SELL'
            entry_price = close
            # Reset budget for the current trade
            budget += entry_price

    return budget

def holdThenSell(data, initial_budget=10000):
    # Get the close price on the first day and open price on the last day
    first_day_price = data.iloc[0]['Close']
    last_day_price = data.iloc[-1]['Open']

    # Calculate the quantity of stocks that can be bought with the initial budget
    quantity = initial_budget / first_day_price

    # Calculate the remaining budget after selling the stocks
    remaining_budget = quantity * last_day_price

    return remaining_budget

def findBestLaggingSpanPeriod(ticker_symbol):
    # Load your dataset
    data = pd.read_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv')
    lagging_span_strategy(data)

def published_vwap_strategy(ticker_symbol):
    data = pd.read_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv')
    # Get the latest entry
    latest_entry = data.iloc[-1]
    
    # Decision based on latest price and VWAP
    if latest_entry['Close'] < latest_entry['VWAP']:
        decision = 'BUY'
    elif latest_entry['Close'] > latest_entry['VWAP']:
        decision = 'SELL'
    else:
        decision = 'HOLD'
    
    return decision

def findBestIndicator(ticker_symbol):
    # Load your dataset
    data = pd.read_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv')
    max_budget = float('-inf')  # Initialize max profit to negative infinity

    budget = vwap_strategy(data)
    print("budget using VWAP strategy:", budget)
    max_budget=max(max_budget,budget)


    most_profitable_level, budget = fib_levels_strategy(data)
    print("Most budgetable Fib level:", most_profitable_level)
    print("budget:", budget)
    max_budget=max(max_budget,budget)

    budget = pivot_point_strategy(data)
    print("Total budget using Pivot Point strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = support_resistance_strategy(data)
    print("budget using Support 1 and Resistance 1 levels strategy:", budget)
    max_budget=max(max_budget,budget)



    budget = cmf_strategy(data)
    print("budget using CMF strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = obv_strategy(data)
    print("budget using OBV strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = atr_strategy(data, atr_multiplier=1.5)  # You can adjust the multiplier as needed
    print("budget using ATR strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = lower_bollinger_band_strategy(data)
    print("budget using Lower Bollinger Band strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = upper_bollinger_band_strategy(data)
    print("budget using Upper Bollinger Band strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = bollinger_bands_strategy(data)
    print("budget using Bollinger Bands strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = psar_strategy(data)
    print("budget using PSAR strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = adx_strategy(data)
    print("budget using ADX strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = macd_signal_line_strategy(data)
    print("budget using MACD with Signal Line strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = macd_strategy(data)
    print("budget using MACD strategy without Signal Line:", budget)
    max_budget=max(max_budget,budget)

    budget = momentum_strategy(data)
    print("budget using Momentum strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = roc_strategy(data, roc_period=14, buy_threshold=2, sell_threshold=-2)
    print("budget using ROC strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = stochastic_d_strategy(data, k_period=14, d_period=3, buy_threshold=20, sell_threshold=80)
    print("budget using Stochastic %D and %K strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = stochastic_k_strategy(data, k_period=14, buy_threshold=20, sell_threshold=80)
    print("budget using Stochastic strategy (just %K):", budget)
    max_budget=max(max_budget,budget)

    budget = wma_200_strategy(data)
    print("budget using WMA_200 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = ema_200_strategy(data) 
    print("budget using EMA_200 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = sma_200_strategy(data)
    print("budget using SMA_200 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = wma_50_strategy(data)
    print("budget using WMA_50 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = ema_50_strategy(data)
    print("budget using EMA_50 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = sma_50_strategy(data)
    print("budget using SMA_50 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = wma_20_strategy(data)
    print("budget using WMA_20 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = ema_20_strategy(data)
    print("budget using EMA_20 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = sma_20_strategy(data)
    print("budget using SMA_20 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = rsi_strategy(data, buy_threshold=30, sell_threshold=70)
    print("budget using RSI strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = lagging_span_strategy(data)
    print("budget using Lagging Span strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = leading_span_b_strategy(data)
    print("budget using Leading Span B strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = base_line_strategy(data)
    print("budget using Base Line strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = conversion_line_strategy(data)
    print("budget using Conversion Line strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = support_resistance_3_strategy(data)
    print("budget using Support/Resistance 3 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = support_resistance_2_strategy(data)
    print("budget using Support/Resistance 2 strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = lagging_span_sma_strategy(data)
    print("budget using Lagging Span with SMA strategy:", budget)
    max_budget=max(max_budget,budget)

    budget = bollinger_band_strategy(data)
    print("budget from Bollinger Band strategy:", budget)

    budget = holdThenSell(data)
    print("budget from holding then selling:", budget)

    print("Max budget:", max(max_budget,budget))

"""
ticker_symbol = 'BTC'
findBestIndicator(ticker_symbol)
ticker_symbol = 'TQQQ'
findBestIndicator(ticker_symbol)
"""
ticker_symbol = 'BTC'
max_budgets = findBestIndicator(ticker_symbol)