#app.py
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sb
from data_loader import load_data
from optimizer import optimize
from backtester import backtest
from risk_metrics import compute_metrics
from visualizer import visualize, plot_corr_heatmap, plot_max_drawdown, plot_rolling_sharpe

st.set_page_config(page_title="Portfolio Optimizer", layout="wide")
st.title("Portfolio Optimization Engine")
st.markdown("Backtest and compare portfolio strategies against SPY benchmark.")

with st.sidebar:
    st.header("Portfolio Configuration")
    st.caption("Minimum 2 years of data recommended for reliable backtesting.")
    n_assets=st.number_input("Number of Assets",min_value=1,value=3,step=1)
    tickers=[]
    for i in range(n_assets):
        tickers.append(st.text_input(f"Ticker {i+1}", key=f"ticker_{i}"))
    start_date=st.date_input("Start Date")
    end_date=st.date_input("End Date")
    strategy=st.selectbox("Strategy",["Equal Weight","Minimum Volatility","Maximum Sharpe","Risk Parity","Momentum Based"])
    strategy_map={
        "Equal Weight":"equal_weight",
        "Minimum Volatility":"min_variance",
        "Maximum Sharpe":"max_sharpe",
        "Risk Parity":"risk_parity",
        "Momentum Based":"momentum_based"
    }
    rf=0.0
    lookback_periods=126
    lr=0.01
    iters=10000
    if strategy=="Maximum Sharpe":
        rf=st.number_input("Risk-Free Rate",value=0.0,step=0.01)
    elif strategy=="Momentum Based":
        lookback_periods=st.number_input("Lookback Periods",value=126,step=1)
    elif strategy=="Risk Parity":
        lr=st.number_input("Learning Rate",value=0.01)
        iters=st.number_input("Iterations",value=5000,step=100)
        method=st.selectbox("Method",["gradient_descent","slsqp"])
        if lr>0.1:
            st.warning("High learning rate may cause the optimizer to not converge.")
    backtest_lookback=st.number_input("Parameter Estimation Period (days)",min_value=30,value=252,step=1)
    rebalancing_frequency=st.number_input("Rebalancing Frequency (days)",min_value=1,value=7,step=1)
    rolling_window=st.number_input("Rolling Window (days)",min_value=5,value=30,step=1)

tickers=[t.strip() for t in tickers if t.strip()]

if st.button("Compute!"):
    prices,valid_tickers,invalid_tickers=load_data(tickers,str(start_date),str(end_date))
    if invalid_tickers:
        for ticker,reason in invalid_tickers.items():
            st.warning(f"{ticker}: {reason}")
    log_returns=np.log(prices/prices.shift(1)).dropna()
    kwargs={}
    if strategy=="Momentum Based":
        kwargs["periods"]=int(lookback_periods)
    elif strategy=="Risk Parity":
        kwargs["lr"]=lr
        kwargs["iters"]=int(iters)
        kwargs["method"]=method
    elif strategy=="Maximum Sharpe":
        kwargs["rf"]=rf
    bt_results=backtest(
        log_returns,
        lookback=int(backtest_lookback),
        investment_strategy=strategy_map[strategy],
        freq=int(rebalancing_frequency),
        **kwargs
    )
    weights=optimize(strategy_map[strategy],log_returns,**kwargs)
    metrics=compute_metrics(bt_results["returns"],freq=rebalancing_frequency)
    st.session_state["bt_results"]=bt_results
    st.session_state["metrics"]=metrics
    st.session_state["weights"]=weights
    st.session_state["valid_tickers"]=valid_tickers
    st.session_state["avg_turnover"]=bt_results["turnover"].mean()
    st.session_state["rebalancing_frequency"]=rebalancing_frequency
    st.session_state["rolling_window"]=rolling_window
    st.session_state["log_returns"] = log_returns

if "bt_results" in st.session_state:
    bt_results=st.session_state["bt_results"]
    metrics=st.session_state["metrics"]
    weights=st.session_state["weights"]
    valid_tickers=st.session_state["valid_tickers"]
    avg_turnover=st.session_state["avg_turnover"]
    rebalancing_frequency=st.session_state["rebalancing_frequency"]
    rolling_window=st.session_state["rolling_window"]
    portfolio_results=bt_results["returns"]
    spy_results=bt_results["spy_returns"]
    log_returns=st.session_state["log_returns"]

    st.header("RESULTS")
    st.subheader("Portfolio Weights")
    st.dataframe(pd.DataFrame({"Ticker":valid_tickers,"Weight":weights}))

    st.subheader("Risk Metrics")
    st.dataframe(pd.DataFrame(metrics,index=["Value"]).T)
    st.metric("Average Turnover",f"{round(avg_turnover*100,2)}%")

    if st.button("Compare with SPY(S&P500)"):
        spy_metrics=compute_metrics(spy_results,freq=rebalancing_frequency)
        st.dataframe(pd.DataFrame(spy_metrics,index=["Value"]).T)

    st.subheader("Performance")
    fig=visualize(portfolio_results,spy_results,window=int(rolling_window))
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    fig_corr = plot_corr_heatmap(log_returns)
    st.pyplot(fig_corr)

    st.subheader("Max Drawdown")
    fig_drawdown = plot_max_drawdown(portfolio_results,spy_results)
    st.pyplot(fig_drawdown)

    st.subheader("Rolling Sharpe Ratio")
    fig_sharpe = plot_rolling_sharpe(portfolio_results,spy_results,rolling_window)
    st.pyplot(fig_sharpe)
