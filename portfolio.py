#portfolio.py

import pandas as pd
import numpy as np
import scipy.stats as stats
from analytics import cov_matrix
from risk_metrics import calc_historical_VaR

def calc_portfolio_return(weights,log_returns):
    ann_returns=(log_returns.mean())*252
    portfolio_returns=weights@ann_returns
    return portfolio_returns

def calc_portfolio_volatility(weights,log_returns):
    cov=cov_matrix(log_returns)
    portfolio_volatility=np.sqrt(weights@cov@weights)*np.sqrt(252) #since weights is a 1D array it wont matter
    return portfolio_volatility

def calc_portfolio_historical_VaR(weights,log_returns,confidence_level=0.95):
    portfolio_daily_returns=log_returns@weights
    portfolio_historical_VaR=calc_historical_VaR(portfolio_daily_returns,confidence_level=0.95)
    return portfolio_historical_VaR

def calc_portfolio_analytical_VaR(weights,log_returns,confidence_level=0.95):
    portfolio_daily_returns=log_returns@weights
    port_mean=portfolio_daily_returns.mean()
    port_vol=portfolio_daily_returns.std()
    z_c = stats.norm.ppf(1 - confidence_level)
    return -(port_mean + z_c * port_vol)

#portfolio.py so far