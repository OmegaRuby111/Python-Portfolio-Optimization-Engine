import numpy as np
import pandas as pd
import yfinance as yf

def load_data(ticker_list,start_date,end_date):
    data=yf.download(tickers=ticker_list,start=start_date,end=end_date,progress=False,auto_adjust=True)
    data_close = data["Close"]
    if isinstance(data_close.columns, pd.MultiIndex):
        data_close.columns = data_close.columns.get_level_values(0)
    valid_tickers=[]
    invalid_tickers={}

    for ticker in ticker_list:
        series=data_close[ticker]
        
        if series.isna().all():
            invalid_tickers[ticker] = "No data found"
            continue
        
        nan_pct=series.isna().sum()/len(series)
        if nan_pct>0.2:
            invalid_tickers[ticker] = "Too many gaps (>20% missing)"
            continue
        
        if series.count()<252:
            invalid_tickers[ticker] = "Insufficient history (<252 days)"
            continue
        
        valid_tickers.append(ticker)

    if not valid_tickers:
        raise ValueError("No valid tickers found. Please check your tickers and add correct ones")
    prices=data_close[valid_tickers]
    prices=prices.ffill()
    prices=prices.dropna()
    return prices,valid_tickers,invalid_tickers
