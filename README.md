# Python Portfolio Optimization Engine
A quantitative portfolio optimization and backtesting engine built with Python and Streamlit. Supports multiple allocation strategies, walk-forward backtesting, risk analytics, and SPY benchmark comparison.

## Live Demo
[Launch App](https://python-portfolio-optimization-engine-ijy95dabgzegemdwhri75q.streamlit.app/)

## Methodology
The engine uses a walk-forward backtesting framework with a rolling 252-day estimation window. At each rebalancing date, portfolio weights are computed using only historical data available up to that point, then applied to the subsequent forward period. This prevents look-ahead bias and reflects realistic out-of-sample performance. Transaction costs are deducted as a percentage of turnover at each rebalance, and risk metrics are computed on the resulting return series.

## Features

**Strategies**
- Equal Weight
- Minimum Volatility — solves for the portfolio with lowest variance via quadratic programming (SciPy)
- Maximum Sharpe Ratio — maximizes the return-to-volatility ratio via constrained optimization (SciPy)
- Risk Parity (Gradient Descent and SLSQP) — allocates such that each asset contributes equally to total portfolio volatility
- Momentum Based — overweights assets with strongest recent return history

**Backtesting**
- Walk-forward backtesting with configurable lookback and rebalancing frequency
- Transaction cost modeling with portfolio turnover tracking
- Real calendar date indexing on all backtest results

**Risk Metrics**
- Annualized Return and Volatility
- Sharpe Ratio and Sortino Ratio
- Maximum Drawdown
- Historical and Analytical VaR
- Historical and Analytical Expected Shortfall (ES)

**Visualizations**
- Cumulative returns with SPY overlay
- Rolling Sharpe Ratio
- Drawdown chart
- Correlation heatmap

**Benchmark Comparison**
- SPY (S&P 500) comparison over the same backtest period
- Side-by-side risk metrics vs benchmark on demand

## Tech Stack
- Python
- Streamlit
- NumPy / Pandas
- yfinance
- Matplotlib / Seaborn
- SciPy (constrained optimization)

## Installation
```bash
git clone https://github.com/OmegaRuby111/Python-Portfolio-Optimization-Engine
cd Python-Portfolio-Optimization-Engine
pip install -r requirements.txt
streamlit run app.py
```

## Usage
1. Enter tickers and date range in the sidebar
2. Select a strategy and configure its parameters
3. Set backtester lookback and rebalancing frequency
4. Click Compute!
5. Click Compare with SPY(S&P500) to view benchmark metrics

## Project Structure
