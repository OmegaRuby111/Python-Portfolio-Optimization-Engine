#visualizer.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

def visualize(returns,spy_returns,window=30):
    fig,axes=plt.subplots(2,1,figsize=(12,12))
    
    #Cumulative returns
    cumulative_returns=returns.cumsum()
    cumulative_spy_returns=spy_returns.cumsum()
    axes[0].plot(cumulative_returns.index,cumulative_returns.values,color="steelblue",linewidth=1.5)
    axes[0].plot(cumulative_spy_returns.index,cumulative_spy_returns.values,color="green",linewidth=1.5)
    axes[0].axhline(0,color="black",linewidth=0.8,linestyle="--")
    axes[0].set_ylabel("Cumulative log returns")
    axes[0].set_title("Cumulative Returns")
    axes[0].legend(["SPY", "Portfolio"])
    axes[0].grid(True,alpha=0.3)

    #rolling return
    rolling_returns=returns.rolling(window=window).mean()
    rolling_spy_returns=spy_returns.rolling(window=window).mean()
    axes[1].plot(rolling_returns.index,rolling_returns.values,color="darkorange",linewidth=1.5)
    axes[1].plot(rolling_spy_returns.index,rolling_spy_returns.values,color="pink",linewidth=1.5)
    axes[1].axhline(0,color="black",linewidth=0.8,linestyle="--")
    axes[1].set_ylabel(f"Rolling Mean Return ({window}p)")
    axes[1].set_title(f"Rolling Returns (window={window})")
    axes[1].legend(["Portfolio", "SPY"])
    axes[1].grid(True,alpha=0.3)

    plt.tight_layout()
    return fig

def plot_corr_heatmap(log_returns):
    fig,ax=plt.subplots(figsize=(8,6))
    corr=log_returns.corr()
    sb.heatmap(corr,annot=True,fmt=".2f",cmap="coolwarm",center=0,ax=ax,linewidths=0.5)
    ax.set_title("Asset Correlation Heatmap")
    plt.tight_layout()
    return fig
