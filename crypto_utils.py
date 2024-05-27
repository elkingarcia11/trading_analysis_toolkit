import datetime
import csv_utils
import data_utils
import pytz
import requests

def fetch_and_save_all_historical_data():
    """
    Fetches and saves all historical data from an API in batches, continuously updating until all data is collected.
    """
    api_key = 'YOUR_API_KEY_HERE'  # Replace this with your actual API key
    to_timestamp = None  # Start with no timestamp to fetch the most recent data
    all_data_collected = False
    first_batch = True

    try:
        while not all_data_collected:
            raw_data = fetch_data(api_key, to_timestamp=to_timestamp)
            if not raw_data:
                all_data_collected = True
                break
            
            formatted_data = data_utils.format_data(raw_data)
            csv_utils.save_to_csv(formatted_data, include_headers=first_batch)
            to_timestamp = raw_data[0]['time'] - 1  # Update timestamp to just before the first in the batch
            first_batch = False
            print(f"Fetched data up to {formatted_data[-1][0]}")
            
    except Exception as e:
        print("Exception: ", e)

def fetch_data(api_key, symbol="BTC", compare_currency="USD", to_timestamp=None, limit=2000, aggregate=1):
    """
    Fetches up to 2000 historical data points for a cryptocurrency pair from an API up to a specified timestamp.
    """
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



def fetch_latest_data(api_key, symbol="BTC", compare_currency="USD", limit=1, aggregate=1):
    """
    Fetches the last {LIMIT} data points for a cryptocurrency pair from an API.
    """
    url = "https://min-api.cryptocompare.com/data/v2/histohour"
    params = {
        'fsym': symbol,
        'tsym': compare_currency,
        'limit': limit,
        'aggregate': aggregate,
        'e': 'CCCAGG',
        'api_key': api_key
    }

    response = requests.get(url, params=params)
    data = response.json()

    if data["Response"] == "Success":
        return data['Data']['Data']
    else:
        raise Exception(f"Failed to fetch data: {data.get('Message', 'No error message')}")

def fetch_and_save_latest_data(ticker_symbol):
    """
    Fetches and saves latest data to csv file.
    """
    api_key = 'YOUR_API_KEY_HERE'  # Replace this with your actual API key
    try:
        # Get the last datetime from the CSV file
        csv_file_path = f"data/{ticker_symbol}.csv"

        # Get the last datetime from the CSV file (assume it's in UTC)
        last_datetime = csv_utils.get_last_datetime_from_csv(csv_file_path)
        last_datetime = last_datetime.replace(tzinfo=pytz.utc)  # Ensure timezone-aware in UTC

        # Get the current datetime in UTC (timezone-aware)
        new_datetime_utc = datetime.datetime.now(pytz.utc)

        # Calculate the difference in hours
        time_difference_hours = (new_datetime_utc - last_datetime).total_seconds() / 3600

        # Fetch the latest data
        raw_data = fetch_latest_data(api_key, limit=time_difference_hours)
        
        if raw_data:
            formatted_data = data_utils.format_data(raw_data)
            
            # Skip the first row to avoid duplicate entries
            formatted_data = formatted_data[1:]
            
            # Save to CSV
            csv_utils.save_to_csv(formatted_data, csv_file_path, include_headers=False)
            print(f"Latest hourly data fetched and saved at {formatted_data[-1][0]}")
        else:
            print("No data fetched.")
    except Exception as e:
        print(e)

if __name__ == "__main__":

    fetch_and_save_latest_data("BTC")
    """
    
    # Get the current datetime in local timezone
    current_datetime_local = datetime.datetime.now()

    # Convert the local datetime to UTC
    current_datetime_utc = current_datetime_local.astimezone(pytz.utc)

    # Print the datetime in UTC
    print(current_datetime_utc)
    csv_file_path = f"data/BTC.csv"
    last_datetime = csv_utils.get_last_datetime_from_csv(csv_file_path)
    print(last_datetime)
    print(type(last_datetime))
    """