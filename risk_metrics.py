#risk_metrics.py

import numpy as np
import pandas as pd
import scipy.stats as stats

def calc_sharpe_ratio(ann_volatility,ann_return,rf=0.0):
    sharpe_ratio=(ann_return-rf)/ann_volatility
    return sharpe_ratio

def calc_sortino_ratio(ann_return,log_returns,rf_per_period=0.0,freq=1):
    excess_daily=log_returns-rf_per_period
    downside_returns=excess_daily.copy()
    downside_returns[downside_returns>0]=0
    downside_std=np.sqrt((downside_returns**2).mean())*np.sqrt(252/freq)
    annual_excess=ann_return-(rf_per_period*(252/freq))
    sortino_ratio=annual_excess/downside_std
    return sortino_ratio

def calc_max_drawdown(wealth_curve):
    running_max=wealth_curve.cummax()
    drawdown=(wealth_curve-running_max)/running_max
    max_drawdown=drawdown.min()
    return max_drawdown

def calc_historical_VaR(log_returns,confidence_level=0.95):
    if(confidence_level>1):
        raise ValueError("Confidence level must be between 0 and 1")
    historical_VaR=np.percentile(log_returns,(1-confidence_level)*100)
    return historical_VaR

def calc_analytical_VaR(log_returns,confidence_level=0.95):
    if(confidence_level>1):
        raise ValueError("Confidence level must be between 0 and 1")
    z_c=stats.norm.ppf(1-confidence_level)
    var=-(log_returns.mean()+z_c*log_returns.std())
    return var

def calc_historical_ES(log_returns,confidence_level=0.95):
    if(confidence_level>1):
        raise ValueError("Confidence level must be between 0 and 1")
    var_threshold=calc_historical_VaR(log_returns,confidence_level)
    return log_returns[log_returns<=var_threshold].mean()

def calc_analytical_ES(log_returns,confidence_level=0.95):
    if(confidence_level>1):
        raise ValueError("Confidence level must be between 0 and 1")
    z_c=stats.norm.ppf(1-confidence_level)
    ES=-(log_returns.mean()-log_returns.std()*stats.norm.pdf(z_c)/(1-confidence_level))
    return ES

def compute_metrics(portfolio_returns, rf=0.0, confidence_level=0.95,freq=1):
    ann_return = portfolio_returns.mean() * (252/freq)
    ann_vol = portfolio_returns.std() * np.sqrt(252/freq)
    prices = np.exp(portfolio_returns.cumsum())
    
    return {
        "Annualized Return": round(ann_return, 4),
        "Annualized Volatility": round(ann_vol, 4),
        "Sharpe Ratio": round(calc_sharpe_ratio(ann_vol, ann_return, rf), 4),
        "Sortino Ratio": round(calc_sortino_ratio(ann_return, portfolio_returns, rf*freq/252, freq), 4),
        "Max Drawdown": round(calc_max_drawdown(prices), 4),
        "Historical VaR": round(calc_historical_VaR(portfolio_returns, confidence_level), 4),
        "Analytical VaR": round(calc_analytical_VaR(portfolio_returns, confidence_level), 4),
        "Historical ES": round(calc_historical_ES(portfolio_returns, confidence_level), 4),
        "Analytical ES": round(calc_analytical_ES(portfolio_returns, confidence_level), 4)
    }

