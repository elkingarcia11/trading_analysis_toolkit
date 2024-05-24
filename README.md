# Financial Analysis Toolbox and Stock Price Forecasting

This project comprises a suite of tools for computing and testing diverse financial indicators, as well as, deploying trading tactics. Additionally, it features a script for forecasting stock prices utilizing a neural network and a script for sending real-time trading alerts via Twitter.

## Scripts

- `financial_indicators.py`: Contains functions for deriving financial indicators including Fibonacci retracements, pivot points, Ichimoku Cloud, moving averages (SMA, EMA, WMA), RSI, stochastic oscillator, rate of change, momentum, MACD, ADX, PSAR, Bollinger Bands, ATR, OBV, CMF, and VWAP. It also facilitates updating the data CSV file with these newly computed indicators.
- `neural_network_stock_predictor.py`: Houses the neural network model designed for predicting stock prices.
- `trading_strategies.py`: Contains trading strategies corresponding to each indicator and identifies the most profitable trading strategy/indicator.
- `stock_alerter.py`: Integrates with Twitter for real-time trading signals. It automatically sends out buy, sell, or hold recommendations based on the latest market data. Ensure to set up your Twitter API credentials in the environment variables for seamless integration.
- `alert.py`: Main function to download the latest data, merge it with existing data, compute financial indicators, and determine trading actions. It sends real-time alerts via Twitter based on the computed indicators.
- `btc.py`: Script for fetching and saving Bitcoin (BTC) hourly data from the CryptoCompare API.

## Trading Strategies

These strategies are formulated by leveraging pivotal thresholds of individual indicators to trigger buying and selling actions. The primary objective is to pinpoint the indicator that maximizes profitability based on the available data.

## Usage

1. Ensure Python is installed on your system.
2. Install the necessary dependencies listed in `requirements.txt`:
   ```sh
   pip install -r requirements.txt
   ```
3. Set up your environment variables. Create a `.env` file in the root directory with the following content:
   ```
   GOOGLE_CLOUD_BUCKET_NAME=your_google_cloud_bucket_name
   TWITTER_BEARER_TOKEN=your_twitter_bearer_token
   TWITTER_API_KEY=your_twitter_api_key
   TWITTER_API_KEY_SECRET=your_twitter_api_key_secret
   TWITTER_ACCESS_TOKEN=your_twitter_access_token
   TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret
   ```
4. Ensure your Google Cloud service account credentials are saved as `service_account_credentials.json` in the root directory.
5. Run the `alert.py` function to perform the full workflow:
   ```sh
   python alert.py
   ```

## Data

- The raw data should be in CSV format, with columns named 'Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', and 'Volume'.
- Processed data containing computed financial indicators will be stored in a CSV file named `data/{TICKER_SYMBOL}_WITH_INDICATORS.csv`.

## Twitter Integration

This project is integrated with Twitter for real-time trading signals. It automatically tweets buy, sell, or hold recommendations based on the latest market data. Make sure to set up your Twitter API credentials in the environment variables for seamless integration.

## Google Cloud Storage Integration

This project uses Google Cloud Storage to manage CSV files. Ensure you have set up your Google Cloud Storage bucket and service account credentials to access the bucket. The bucket name should be specified in the environment variables.

This project uses Google Cloud Function to host the program and Cloud Scheduler to trigger the Cloud Function every day after market closes.

This project uses Google Cloud Build for CI/CD to update the Cloud Function program anytime the Github repo changes.

## Example Workflow

Here's an example workflow to get you started:

1. **Download Historical Data**: Ensure the CSV files with historical data are properly formatted, downloaded and saved in the `data` directory.

2. **Generate Indicators**: Use the functions within `financial_indicators.py` to compute financial indicators for the historical data:
   ```sh
   python -c "from financial_indicators import generateIndicatorsForCSV; generateIndicatorsForCSV('TICKER_SYMBOL')"
   ```

3. **Run Alert Script**: Execute the `alert.py` script to download the latest data, merge it with the existing data, update financial indicators, and tweet the trading action:
   ```sh
   python alert.py
   ```

By following these steps, you can maintain an up-to-date dataset, apply various financial indicators, and receive real-time trading alerts based on the latest market data.