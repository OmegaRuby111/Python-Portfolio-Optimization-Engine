#backtester.py
import numpy as np
import pandas as pd
from optimizer import optimize
import yfinance as yf

def backtest(log_returns,lookback,investment_strategy="equal_weight",freq=7,transaction_cost=0.001,**kwargs):
    prev_weights=None
    turnover_history=[]
    dates_list=[]
    if (len(log_returns)-lookback-freq+1)<freq:
        raise ValueError(f"Insufficient data: need at least lookback + 2*freq rows, got {len(log_returns)}")
    returns_history=[]
    for i in range(0,len(log_returns)-lookback-freq+1,freq):
        start=i
        lookback_window=log_returns[start:start+lookback]
        forward_period=log_returns[start+lookback:start+lookback+freq]
        weights=optimize(investment_strategy,lookback_window,**kwargs)
        if prev_weights is not None:
            turnover=np.sum(np.abs(weights-prev_weights))
            cost=transaction_cost*turnover
        else:
            turnover=0.0
            cost=0.0
        prev_weights=weights
        turnover_history.append(turnover)
        dates_list.append(forward_period.index[-1])
        portfolio_return=np.sum(forward_period.values@weights)-cost
        returns_history.append(portfolio_return)
    returns_history=pd.Series(returns_history,index=dates_list)
    turnover_history=pd.Series(turnover_history,index=dates_list)
    spy_data=yf.download(tickers="SPY",start=log_returns.index[0],end=log_returns.index[-1],auto_adjust=True,progress=False)
    spy_data.columns = spy_data.columns.get_level_values(0)
    spy_returns=np.log(spy_data/spy_data.shift(1)).dropna()
    spy_period_returns=[]
    for i in range(len(dates_list)):
        if i==0:
            start_date=log_returns.index[lookback]
        else:
            start_date=dates_list[i-1]
        end_date=dates_list[i]
        period_return=float(spy_returns.loc[start_date:end_date,"Close"].sum())
        spy_period_returns.append(period_return)
    spy_series=pd.Series(spy_period_returns,index=dates_list)
    results=pd.DataFrame({
        "returns":returns_history,
        "turnover":turnover_history,
        "spy_returns": spy_series
    },index=dates_list)
    return results
