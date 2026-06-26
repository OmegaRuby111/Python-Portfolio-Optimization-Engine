#optimize.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from risk_metrics import calc_sharpe_ratio
from portfolio import calc_portfolio_return,calc_portfolio_volatility
from analytics import cov_matrix

def equal_weight(log_returns):
    n=log_returns.shape[1]
    weights=np.ones(n)/n
    return weights

def max_sharpe(log_returns,rf=0.0):
    n=log_returns.shape[1]
    init_weights=np.ones(n)/n

    constraints={"type":"eq","fun":lambda w:np.sum(w)-1}
    bounds=[(0,1)]*n

    def neg_sharpe(w):
        ret=calc_portfolio_return(w,log_returns)
        vol=calc_portfolio_volatility(w,log_returns)
        return -calc_sharpe_ratio(vol,ret,rf)
    
    result=minimize(neg_sharpe,init_weights,method="SLSQP",bounds=bounds,constraints=constraints)

    if not result.success:
        raise ValueError(f"Max Sharpe optimization failed: {result.message}")
    return result.x

def min_variance(log_returns):
    n=log_returns.shape[1]
    init_weights=np.ones(n)/n
    constraints={"type":"eq","fun":lambda w:np.sum(w)-1}
    bounds=[(0,1)]*n

    def portfolio_variance(w):
        vol=calc_portfolio_volatility(w, log_returns)
        return vol**2
    
    result=minimize(portfolio_variance,init_weights,method="SLSQP",bounds=bounds,constraints=constraints)

    if not result.success:
        raise ValueError(f"Min Variance optimization failed:{result.message}")
    
    return result.x

def risk_parity(log_returns,lr=0.01,iters=5000,method="gradient_descent"):
    n = log_returns.shape[1]
    cov = cov_matrix(log_returns)
    ideal_risk = 1/n

    if method=="gradient_descent":
        tol=1e-6
        weights=np.ones(n)/n
        for i in range(iters):
            portfolio_volatility=np.sqrt(weights@cov@weights)
            risk_contribution=(weights*(cov@weights))/portfolio_volatility
            normalized_risk_contribution=risk_contribution/risk_contribution.sum()
            update=(lr*(normalized_risk_contribution-ideal_risk))
            weights=weights-update
            weights=np.clip(weights,0,None)
            weights=weights/weights.sum()
            if np.max(np.abs(update))<tol:
                return weights
        else:
            print(f"Warning: Did not converge within {iters} iterations")
            return weights
    
    elif method=="slsqp":
        def risk_parity_objective(weights):
            port_vol = np.sqrt(weights@cov@weights)
            risk_contribution = (weights * (cov @ weights)) / port_vol          # risk contributions
            normalized_risk_contribution = risk_contribution / risk_contribution.sum()
            return np.sum((normalized_risk_contribution - ideal_risk) ** 2)
        init_weights = np.ones(n)/n
        constraints = {"type": "eq", "fun": lambda w: np.sum(w)-1}
        bounds = [(0, 1)]*n
        result = minimize(risk_parity_objective, init_weights, method="SLSQP", bounds=bounds, constraints=constraints)
        if not result.success:
            raise ValueError(f"Risk Parity SLSQP failed: {result.message}")
        return result.x
    
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: 'gradient_descent', 'slsqp'")

def momentum(log_returns,periods=126):
    if(periods>len(log_returns)):
        raise ValueError("Length asked by user exceeds available history of asset")
    if periods>len(log_returns)*0.8:
        print(f"Consuming atleast 80% of available history")
    signal=log_returns.tail(periods).sum()
    signal=np.clip(signal,0,None)
    if signal.sum()==0:
        weights=np.ones(log_returns.shape[1])/log_returns.shape[1]
    else:
        weights=signal/signal.sum()
    return weights

def optimize(strategy,log_returns,rf=0.0,lr=0.01,iters=10000,periods=126,method="gradient_descent"):
    strategies={
        "equal_weight":lambda:equal_weight(log_returns),
        "max_sharpe": lambda:max_sharpe(log_returns,rf),
        "min_variance":lambda:min_variance(log_returns),
        "risk_parity":lambda:risk_parity(log_returns,lr,iters,method),
        "momentum_based":lambda:momentum(log_returns,periods)
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown Strategy '{strategy}'.Choose from:{list(strategies.keys())}")
    
    return strategies[strategy]()
