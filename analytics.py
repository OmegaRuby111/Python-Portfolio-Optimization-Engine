#analytics.py
import numpy as np
import pandas as pd

def compute_returns(prices):
    return np.log(prices/prices.shift(1)).dropna()

def calc_annualized_return(log_returns):
    daily_return=np.mean(log_returns)
    annualized_return=daily_return*252
    return annualized_return

def calc_annualized_volatility(log_returns):
    daily_volatility=log_returns.std()
    annualized_volatility=daily_volatility*np.sqrt(252)
    return annualized_volatility

def corr_matrix(log_returns):
    correlation_matrix=log_returns.corr()
    return correlation_matrix

def cov_matrix(log_returns):
    covariance_matrix=log_returns.cov()
    return covariance_matrix
