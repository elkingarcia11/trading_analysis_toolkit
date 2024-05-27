import cloud_utils
import crypto_utils
import financial_utils
import trading_utils
import twitter_utils

def alert(ticker_symbol):
    # Pull data from cloud
    cloud_utils.fetch_data(ticker_symbol)

    # Fetch and merge latest data
    crypto_utils.fetch_and_save_latest_data(ticker_symbol)

    # Push data to cloud
    cloud_utils.upload_data(ticker_symbol)

    # Calculate new indicators
    financial_utils.generate_vwap_indicator(ticker_symbol)

    # Determine if should buy, sell, hold
    decision = trading_utils.published_vwap_strategy(ticker_symbol)

    decision = f"{ticker_symbol}: " + decision
    
    # Tweet determination
    twitter_utils.tweet(decision)