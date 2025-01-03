import yfinance as yf
import pandas as pd

def read_yfinance_crypto_data(ticker_name, period, interval=None):
    try:
        # Read the CSV file
        ticker = yf.Ticker(ticker_name)
        df = ticker.history(period, interval)

        # Clean columns
        return df.loc[:, ~df.columns.isin(["Volume", "Dividends", "Stock Splits"])]
    
    except FileNotFoundError:
        print("Error: The specified CSV file was not found.")
        return None
    
    except pd.errors.EmptyDataError:
        print("Error: The CSV file is empty.")
        return None
    
    except pd.errors.ParserError:
        print("Error: An error occurred while parsing the CSV file.")
        return None