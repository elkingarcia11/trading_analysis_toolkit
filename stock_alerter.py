import os
import pandas as pd
import yfinance as yf
import tweepy
from datetime import datetime
from financial_indicators import generateIndicatorsForCSV
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

def getLastDate(ticker_symbol):
    """Get the last date from the CSV file for the given ticker symbol."""
    df = pd.read_csv(f'data/{ticker_symbol}.csv')
    last_entry_date = df.iloc[-1]['Date']
    return last_entry_date

def getLatestData(ticker_symbol):
    """Retrieve the latest data from Yahoo Finance for the given ticker symbol."""
    start_date = getLastDate(ticker_symbol)
    
    if pd.isna(start_date):
        data = yf.download(ticker_symbol, interval='1d')
    else:
        data = yf.download(ticker_symbol, start=start_date, interval='1d')

    data.to_csv(f'data/{ticker_symbol}_LATEST.csv', header=True)

def mergeData(ticker_symbol):
    """Merge old data with the latest data and update the CSV file."""
    old_data = pd.read_csv(f'data/{ticker_symbol}.csv')
    new_data = pd.read_csv(f'data/{ticker_symbol}_LATEST.csv')
    merged_data = pd.concat([old_data, new_data]).drop_duplicates().reset_index(drop=True)

    merged_data.to_csv(f'data/{ticker_symbol}.csv', index=False)
    os.remove(f'data/{ticker_symbol}_LATEST.csv')

    generateIndicatorsForCSV(ticker_symbol)

    key_path = "service_account_credentials.json"
    storage_client = storage.Client.from_service_account_json(key_path)

    bucket_name = os.getenv('GOOGLE_CLOUD_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("Bucket name is not set in the environment variable 'GOOGLE_CLOUD_BUCKET_NAME'")
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{ticker_symbol}.csv")
    blob.upload_from_filename(f'data/{ticker_symbol}.csv')

    return f"Data for {ticker_symbol} merged and uploaded to Cloud Storage."

def determine_action_latest_data(ticker_symbol):
    """Determine action based on the latest data and tweet the action."""
    data = pd.read_csv(f'data/{ticker_symbol}_WITH_INDICATORS.csv')
    latest_data = data.iloc[-1]

    close = latest_data['Close']
    lagging_span = latest_data['Lagging_Span']
    conversion_line = latest_data['Conversion_Line']
    base_line = latest_data['Base_Line']
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    if close > conversion_line and close > base_line and lagging_span < close:
        message = f"{ticker_symbol}: BUY"
    elif close < conversion_line and close < base_line and lagging_span > close:
        message = f"{ticker_symbol}: SELL"
    
    tweet(message)

def tweet(message):
    """Post a tweet with the given message."""
    try:
        client = tweepy.Client(
            bearer_token=os.getenv('TWITTER_BEARER_TOKEN'),
            consumer_key=os.getenv('TWITTER_API_KEY'),
            consumer_secret=os.getenv('TWITTER_API_KEY_SECRET'),
            access_token=os.getenv('TWITTER_ACCESS_TOKEN'),
            access_token_secret=os.getenv('TWITTER_ACCESS_TOKEN_SECRET'),
            wait_on_rate_limit=True
        )
        client.create_tweet(text=message)
    except tweepy.TweepyException as e:
        print("Tweepy Exception:", e)
    except Exception as e:
        print("An unexpected error occurred:", e)

def download_csv_from_bucket(ticker_symbol):
    """Download the CSV file from Google Cloud Storage."""
    key_path = "service_account_credentials.json"
    storage_client = storage.Client.from_service_account_json(key_path)

    bucket_name = os.getenv('GOOGLE_CLOUD_BUCKET_NAME')
    if not bucket_name:
        raise ValueError("Bucket name is not set in the environment variable 'GOOGLE_CLOUD_BUCKET_NAME'")
    
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{ticker_symbol}.csv")
    blob.download_to_filename(f"data/{ticker_symbol}.csv")

    print(f"File downloaded from bucket '{bucket_name}' to 'data/{ticker_symbol}.csv'.")

def alert():
    """Main function to execute the alert process."""
    ticker_symbol = 'TQQQ'
    try:
        download_csv_from_bucket(ticker_symbol)
        getLatestData(ticker_symbol)
        mergeData(ticker_symbol)
        determine_action_latest_data(ticker_symbol)
    except Exception as e:
        print(f"An error occurred: {e}")

    ticker_symbol = 'BTC'
    try:
        #download_csv_from_bucket(ticker_symbol)
        #getLatestData(ticker_symbol)
        #mergeData(ticker_symbol)
        determine_action_latest_data(ticker_symbol)
    except Exception as e:
        print(f"An error occurred: {e}")

alert()