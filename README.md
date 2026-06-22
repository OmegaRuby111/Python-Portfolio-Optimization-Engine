# Python Portfolio Optimization Engine

A quantitative portfolio optimization and backtesting engine built with Python and Streamlit. Supports multiple allocation strategies, walk-forward backtesting, risk analytics, and SPY benchmark comparison.

## Features

**Strategies**
- Equal Weight
- Minimum Volatility
- Maximum Sharpe Ratio
- Risk Parity (Gradient Descent and SLSQP)
- Momentum Based

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

**Benchmark Comparison**
- SPY (S&P 500) comparison over the same backtest period
- Side-by-side risk metrics vs benchmark on demand
- Cumulative returns and rolling returns chart with SPY overlay

## Tech Stack

- Python
- Streamlit
- NumPy / Pandas
- yfinance
- Matplotlib
- SciPy

## Installation

```bash
git clone https://github.com/yourusername/portfolio-optimizer
cd portfolio-optimizer
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Enter tickers and date range in the sidebar
2. Select a strategy and configure its parameters
3. Set backtester lookback and rebalancing frequency
4. Click **Compute!**
5. Click **Compare with SPY(S&P500)** to view benchmark metrics

## Project Structure

```
app.py            # Streamlit UI
data_loader.py    # Price data fetching via yfinance
optimizer.py      # Strategy implementations
backtester.py     # Walk-forward backtesting engine
risk_metrics.py   # Risk and performance metrics
visualizer.py     # Cumulative and rolling return charts
```

## Notes

- All returns are log returns
- Annualization accounts for rebalancing frequency
- SPY benchmark is resampled to match the portfolio's rebalancing periods for apples-to-apples comparison
- Risk Parity supports both gradient descent and SLSQP solvers
