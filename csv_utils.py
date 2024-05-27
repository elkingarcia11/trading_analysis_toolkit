import datetime
import csv
import pandas as pd

def save_to_csv(data, file_path, include_headers=False):
    headers = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    mode = 'a' if not include_headers else 'w'
    with open(file_path, mode, newline='') as f:
        writer = csv.writer(f)
        if include_headers:
            writer.writerow(headers)
        for row in data:
            writer.writerow(row)
            
def get_last_datetime_from_csv(file_path):
    df = pd.read_csv(file_path)
    last_datetime_str = df['Date'].iloc[-1]
    last_datetime = datetime.datetime.strptime(last_datetime_str, '%Y-%m-%d %H:%M:%S')
    return last_datetime

