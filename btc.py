import requests
import datetime
import csv

def fetch_btc_data(api_key, symbol="BTC", compare_currency="USD", to_timestamp=None, limit=2000, aggregate=1):
    """Fetch historical data for BTC at 2-hour intervals."""
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

def fetch_latest_btc_data(api_key, symbol="BTC", compare_currency="USD"):
    """Fetch the most recent data for BTC."""
    url = "https://min-api.cryptocompare.com/data/price"
    params = {
        'fsym': symbol,
        'tsyms': compare_currency,
        'api_key': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if 'USD' in data:
        return data['USD']
    else:
        raise Exception(f"Failed to fetch latest data: {data.get('Message', 'No error message')}")

def format_data(data):
    """Convert data to more readable format."""
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

def save_to_csv(data, filename="BTC.csv", include_headers=False):
    """Save formatted data to a CSV file."""
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if include_headers:
            writer.writerow(['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
        writer.writerows(data)

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

def fetch_and_save_latest_data():
    api_key = 'YOUR_API_KEY_HERE'  # Replace this with your actual API key
    try:
        latest_price = fetch_latest_btc_data(api_key)
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        latest_data = [[current_time, latest_price, latest_price, latest_price, latest_price, latest_price, 0]]
        save_to_csv(latest_data, include_headers=False)
        print(f"Latest data fetched and saved at {current_time}")
    except Exception as e:
        print(e)

if __name__ == "__main__":
    #fetch_historical_data()
    fetch_and_save_latest_data()
