import requests
import datetime
import csv
import os

def fetch_btc_data(api_key, symbol="BTC", compare_currency="USD", to_timestamp=None, limit=2000, aggregate=1):
    """Fetch historical data for BTC at hourly intervals."""
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': symbol,
        'tsym': compare_currency,
        'limit': limit,
        'aggregate': aggregate,
        'e': 'CCCAGG',
        'api_key': api_key,
        'toTs': to_timestamp
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["Response"] == "Success":
        return data['Data']['Data']
    else:
        raise Exception(f"Failed to fetch data: {data.get('Message', 'No error message')}")

def fetch_latest_hourly_btc_data(api_key, symbol="BTC", compare_currency="USD"):
    """Fetch the latest hourly data for BTC."""
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': symbol,
        'tsym': compare_currency,
        'limit': 1,  # Only fetch the most recent hour
        'aggregate': 1,
        'e': 'CCCAGG',
        'api_key': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["Response"] == "Success":
        return data['Data']['Data']
    else:
        raise Exception(f"Failed to fetch data: {data.get('Message', 'No error message')}")

def format_data(data):
    """Convert data to a more readable format."""
    formatted_data = []
    for entry in data:
        date = datetime.datetime.fromtimestamp(entry['time']).strftime('%Y-%m-%d %H:%M:%S')
        formatted_data.append([
            date,
            entry['open'],
            entry['high'],
            entry['low'],
            entry['close'],
            entry['close'],  # Adj Close, assumed same as Close
            entry['volumeto']  # Assuming 'volumeto' is the relevant volume
        ])
    return formatted_data

def save_to_csv(data, filename="data/BTC.csv", include_headers=False):
    """Save formatted data to a CSV file."""
    # Check if the data to be appended already exists
    existing_data = set()
    if os.path.exists(filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            existing_data = {tuple(row) for row in reader}

    # Append only if the data is not already present
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if include_headers and not existing_data:
            writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        if not existing_data:  # If the file is empty, write headers and data
            writer.writerows(data)
        else:
            for row in data:
                if tuple(row) not in existing_data:
                    if len(existing_data) > 0:  # Check if the file is not empty
                        writer.writerow([])  # Add an empty row for separation
                    writer.writerow(row)


def fetch_historical_data():
    api_key = 'YOUR_API_KEY_HERE'  # Replace this with your actual API key
    to_timestamp = None  # Start with no timestamp to fetch the most recent data
    all_data_collected = False
    first_batch = True

    try:
        while not all_data_collected:
            raw_data = fetch_btc_data(api_key, to_timestamp=to_timestamp)
            if not raw_data:
                all_data_collected = True
                break
            
            formatted_data = format_data(raw_data)
            save_to_csv(formatted_data, include_headers=first_batch)
            to_timestamp = raw_data[0]['time'] - 1  # Update timestamp to just before the first in the batch
            first_batch = False
            print(f"Fetched data up to {formatted_data[-1][0]}")
            
    except Exception as e:
        print(e)

def fetch_and_save_latest_hourly_data():
    api_key = 'YOUR_API_KEY_HERE'  # Replace this with your actual API key
    try:
        raw_data = fetch_latest_hourly_btc_data(api_key)
        if raw_data:
            formatted_data = format_data(raw_data)
            formatted_data = formatted_data[1:]
            save_to_csv(formatted_data, include_headers=False)
            print(f"Latest hourly data fetched and saved at {formatted_data[0][0]}")
        else:
            print("No data fetched.")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    # Uncomment the following line to fetch historical data
    # fetch_historical_data()
    
    # Fetch and save the latest hourly data
    fetch_and_save_latest_hourly_data()
