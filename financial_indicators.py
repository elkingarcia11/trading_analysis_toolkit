import pandas as pd
import numpy as np

def fibonacciRetracements(data):
    close_prices = data['Close']
    highest_point = close_prices.max()
    lowest_point = close_prices.min()
    retracement_levels = {
        0.236: highest_point - (0.236 * (highest_point - lowest_point)),
        0.382: highest_point - (0.382 * (highest_point - lowest_point)),
        0.5: highest_point - (0.5 * (highest_point - lowest_point)),
        0.618: highest_point - (0.618 * (highest_point - lowest_point)),
        0.786: highest_point - (0.786 * (highest_point - lowest_point))
    }
    for level, price in retracement_levels.items():
        data[f'Fib_{level}'] = price
    return data

def pivotPoints(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    pivot_point = (high_prices + low_prices + close_prices) / 3
    support1 = (pivot_point * 2) - high_prices
    resistance1 = (pivot_point * 2) - low_prices
    support2 = pivot_point - (high_prices - low_prices)
    resistance2 = pivot_point + (high_prices - low_prices)
    support3 = low_prices - 2 * (high_prices - pivot_point)
    resistance3 = high_prices + 2 * (pivot_point - low_prices)
    data['Pivot_Point'] = pivot_point
    data['Support1'] = support1
    data['Resistance1'] = resistance1
    data['Support2'] = support2
    data['Resistance2'] = resistance2
    data['Support3'] = support3
    data['Resistance3'] = resistance3
    return data  # Ensure to return the modified data DataFrame


def ichimokuCloud(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    nine_period_high = high_prices.rolling(window=9).max()
    nine_period_low = low_prices.rolling(window=9).min()
    conversion_line = (nine_period_high + nine_period_low) / 2
    twenty_six_period_high = high_prices.rolling(window=26).max()
    twenty_six_period_low = low_prices.rolling(window=26).min()
    base_line = (twenty_six_period_high + twenty_six_period_low) / 2
    leading_span_A = ((conversion_line + base_line) / 2).shift(26)
    fifty_two_period_high = high_prices.rolling(window=52).max()
    fifty_two_period_low = low_prices.rolling(window=52).min()
    leading_span_B = ((fifty_two_period_high + fifty_two_period_low) / 2).shift(26)
    chikou_span = close_prices.shift(-26)
    data['Conversion_Line'] = conversion_line
    data['Base_Line'] = base_line
    data['Leading_Span_A'] = leading_span_A
    data['Leading_Span_B'] = leading_span_B
    data['Lagging_Span'] = chikou_span
    return data

def calculate_moving_average(data, column_name, window_size, method):
    close_prices = data['Close']
    if method == 'SMA':
        ma = close_prices.rolling(window=window_size).mean()
    elif method == 'EMA':
        ma = close_prices.ewm(span=window_size, adjust=False).mean()
    elif method == 'WMA':
        weights = np.arange(1, window_size + 1)
        ma = close_prices.rolling(window=window_size).apply(lambda prices: np.dot(prices, weights) / sum(weights), raw=True)
    else:
        raise ValueError("Invalid method. Supported methods: 'SMA', 'EMA', 'WMA'")
    
    data[column_name] = ma
    return data  # Make sure to return the modified data DataFrame


def calculate_all_moving_averages(data, periods):
    for period in periods:
        data = calculate_moving_average(data, f'SMA_{period}', period, 'SMA')
        data = calculate_moving_average(data, f'EMA_{period}', period, 'EMA')
        data = calculate_moving_average(data, f'WMA_{period}', period, 'WMA')
    return data

def rsi(data):
    close_prices = data['Close']
    rsi_period = 14
    price_changes = close_prices.diff()
    gains = price_changes.where(price_changes > 0, 0)
    losses = -price_changes.where(price_changes < 0, 0)
    average_gain = gains.rolling(window=rsi_period, min_periods=1).mean()
    average_loss = losses.rolling(window=rsi_period, min_periods=1).mean()
    relative_strength = average_gain / average_loss
    rsi = 100 - (100 / (1 + relative_strength))
    data['RSI'] = rsi
    return data

def stochasticOscillator(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    stoch_period = 14
    highest_high = high_prices.rolling(window=stoch_period).max()
    lowest_low = low_prices.rolling(window=stoch_period).min()
    stoch_k = 100 * ((close_prices - lowest_low) / (highest_high - lowest_low))
    stoch_d = stoch_k.rolling(window=3).mean()
    data['%K'] = stoch_k
    data['%D'] = stoch_d
    return data

def rateOfChange(data):
    close_prices = data['Close']
    roc_period = 14
    roc = close_prices.pct_change(periods=roc_period) * 100
    data['ROC'] = roc
    return data

def momentum(data):
    close_prices = data['Close']
    momentum_period = 14
    momentum = close_prices.diff(momentum_period)
    data['Momentum'] = momentum
    return data


def macd(data):
    close_prices = data['Close']
    ema_short_period = 12
    ema_long_period = 26
    signal_period = 9
    ema_short = close_prices.ewm(span=ema_short_period, adjust=False).mean()
    ema_long = close_prices.ewm(span=ema_long_period, adjust=False).mean()
    macd_line = ema_short - ema_long
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    data['MACD'] = macd_line
    data['Signal_Line'] = signal_line
    return data

def adx(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    true_range = pd.DataFrame(index=data.index)
    true_range['TR1'] = high_prices - low_prices
    true_range['TR2'] = abs(high_prices - close_prices.shift())
    true_range['TR3'] = abs(low_prices - close_prices.shift())
    true_range['True_Range'] = true_range.max(axis=1)
    positive_dm = high_prices.diff()
    negative_dm = -low_prices.diff()
    positive_dm[positive_dm < 0] = 0
    negative_dm[negative_dm < 0] = 0
    smoothed_positive_dm = positive_dm.ewm(span=14, min_periods=1).mean()
    smoothed_negative_dm = negative_dm.ewm(span=14, min_periods=1).mean()
    positive_di = 100 * (smoothed_positive_dm / true_range['True_Range']).ewm(span=14, min_periods=1).mean()
    negative_di = 100 * (smoothed_negative_dm / true_range['True_Range']).ewm(span=14, min_periods=1).mean()
    dx = 100 * abs((positive_di - negative_di) / (positive_di + negative_di)).ewm(span=14, min_periods=1).mean()
    adx = dx.ewm(span=14, min_periods=1).mean()
    data['ADX'] = adx  # Assign ADX values back to DataFrame
    return data

def psar(data):
    high_prices = data['High']
    low_prices = data['Low']
    af_increment = 0.02
    af_max = 0.2
    sar = low_prices.iloc[0]
    trend = 1
    psar_values = []  # List to store PSAR values
    af = af_increment  # Initialize af here
    for i in range(len(data)):
        if trend == 1:
            sar = sar + 0.02 * (high_prices.iloc[i-1] - sar)
        else:
            sar = sar + 0.02 * (low_prices.iloc[i-1] - sar)
        if trend == 1:
            if sar > low_prices.iloc[i]:
                sar = low_prices.iloc[i]
                trend = -1
                af = af_increment
        else:
            if sar < high_prices.iloc[i]:
                sar = high_prices.iloc[i]
                trend = 1
                af = af_increment
        if trend == 1:
            if high_prices.iloc[i] > high_prices.iloc[i-1]:
                af = min(af + af_increment, af_max)
        else:
            if low_prices.iloc[i] < low_prices.iloc[i-1]:
                af = min(af + af_increment, af_max)
        psar_values.append(sar)
    data['PSAR'] = psar_values  # Assign PSAR values back to DataFrame
    return data



def bollingerBands(data):
    close_prices = data['Close']
    period = 20
    num_std = 2
    rolling_mean = close_prices.rolling(window=period).mean()
    rolling_std = close_prices.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    
    # Assigning calculated values to the DataFrame
    data['Upper Band'] = upper_band
    data['Lower Band'] = lower_band
    
    return data

def atr(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    tr1 = high_prices - low_prices
    tr2 = abs(high_prices - close_prices.shift())
    tr3 = abs(low_prices - close_prices.shift())
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_period = 14
    atr = true_range.rolling(window=atr_period).mean()
    
    # Assigning calculated values to the DataFrame
    data['ATR'] = atr
    
    return data

def obv(data):
    close_prices = data['Close']
    volume = data['Volume']
    obv = [volume.iloc[0]]
    for i in range(1, len(data)):
        if close_prices.iloc[i] > close_prices.iloc[i - 1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close_prices.iloc[i] < close_prices.iloc[i - 1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    data['OBV'] = obv  # Assign OBV values back to DataFrame
    return data

def cmf(data):
    high_prices = data['High']
    low_prices = data['Low']
    close_prices = data['Close']
    volume = data['Volume']
    typical_price = (high_prices + low_prices + close_prices) / 3
    raw_money_flow = typical_price * volume
    positive_money_flow = (typical_price > typical_price.shift(1)) * raw_money_flow
    negative_money_flow = (typical_price < typical_price.shift(1)) * raw_money_flow
    positive_money_flow_sum = positive_money_flow.rolling(window=20).sum()
    negative_money_flow_sum = negative_money_flow.rolling(window=20).sum()
    mf_multiplier = positive_money_flow_sum / negative_money_flow_sum
    mf_volume = mf_multiplier * volume
    cmf = mf_volume.rolling(window=20).sum() / volume.rolling(window=20).sum()
    data['CMF'] = cmf  # Assign CMF values back to DataFrame
    return data

def vwap(data):
    close_prices = data['Close']
    volume = data['Volume']
    vwap_period = 20
    cumulative_price_volume = (close_prices * volume).rolling(window=vwap_period).sum()
    cumulative_volume = volume.rolling(window=vwap_period).sum()
    vwap = cumulative_price_volume / cumulative_volume
    data['VWAP'] = vwap  # Assign VWAP values back to DataFrame
    return data

def generateIndicatorsForCSV(ticker_symbol):
    data = pd.read_csv(f'data/{ticker_symbol}.csv')

    # Apply technical indicators
    data = fibonacciRetracements(data)
    data = pivotPoints(data)
    data = ichimokuCloud(data)
    data = calculate_all_moving_averages(data, periods=[20, 50, 200])
    data = rsi(data)
    data = stochasticOscillator(data)
    data = rateOfChange(data)
    data = momentum(data)
    data = macd(data)
    data = adx(data)
    data = psar(data)
    data = bollingerBands(data)
    data = atr(data)
    data = obv(data)
    data = cmf(data)
    data = vwap(data)

    # Write the modified data back to CSV, overwriting the file
    data.to_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv', mode='w', index=False)

"""

ticker_symbol = "TQQQ"
generateIndicatorsForCSV(ticker_symbol)
ticker_symbol = "BTC"
generateIndicatorsForCSV(ticker_symbol)
"""