import datetime
import os 
import pandas as pd

def update_time(filepath):
    # Read in data from csv
    df = pd.read_csv(filepath)
    # Convert the 'Date' column to datetime objects
    df['Date'] = pd.to_datetime(df['Date'])

    # Subtract 4 hours from each datetime entry
    df['Date'] = df['Date'] + pd.Timedelta(hours=4)

    # Save the updated DataFrame back to a new CSV file
    df.to_csv("data/BTC.csv", index=False)

def format_data(data):
    """Convert data to a more readable format."""
    formatted_data = []
    for entry in data:
        # Assume entry['time'] contains the timestamp
        timestamp = entry['time']

        # Format the timestamp to UTC datetime string
        utc_datetime_str = datetime.datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

        formatted_data.append([
            utc_datetime_str,
            entry['open'],
            entry['high'],
            entry['low'],
            entry['close'],
            entry['close'],  # Adj Close, assumed same as Close
            entry['volumeto']  # Assuming 'volumeto' is the relevant volume
        ])
    return formatted_data

def merge_data(ticker_symbol):
    """Merge old data with the latest data and update the CSV file."""
    old_data = pd.read_csv(f'data/{ticker_symbol}.csv')
    new_data = pd.read_csv(f'data/{ticker_symbol}_LATEST.csv')
    merged_data = pd.concat([old_data, new_data]).drop_duplicates().reset_index(drop=True)

    merged_data.to_csv(f'data/{ticker_symbol}.csv', index=False)
    os.remove(f'data/{ticker_symbol}_LATEST.csv')