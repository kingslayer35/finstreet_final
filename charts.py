"""
Charts: Generate visualization graphs for trading system
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

OUTPUT_DIR = Path(__file__).parent / "output"
FIGURES_DIR = OUTPUT_DIR / "figures"
DATA_DIR = Path(__file__).parent / "data"


def plot_cumulative_returns():
    """Line chart showing cumulative returns over time."""
    equity_path = OUTPUT_DIR / "equity_curve.csv"
    if not equity_path.exists():
        print("No equity curve data found")
        return
    
    df = pd.read_csv(equity_path)
    df['date'] = pd.to_datetime(df['date'])
    
    initial = df['equity'].iloc[0]
    df['return_pct'] = (df['equity'] - initial) / initial * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    ax.plot(df['date'], df['return_pct'], 'b-', linewidth=2.5, marker='o', markersize=4)
    ax.fill_between(df['date'], 0, df['return_pct'], alpha=0.3, color='blue')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Cumulative Returns - IRCON Trading System', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.4)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'cumulative_returns.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: cumulative_returns.png")


def plot_trade_execution():
    """Trade execution chart with price, entry/exit points, and signals."""
    trades_path = OUTPUT_DIR / "trades.csv"
    daily_path = DATA_DIR / "ircon_daily.csv"
    
    if not trades_path.exists() or not daily_path.exists():
        print("Missing data for trade execution chart")
        return
    
    trades = pd.read_csv(trades_path)
    prices = pd.read_csv(daily_path)
    prices['date'] = pd.to_datetime(prices['date'])
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot price line
    ax.plot(prices['date'], prices['close'], 'gray', linewidth=1.5, alpha=0.7, label='Close Price')
    
    # Plot trades
    for _, trade in trades.iterrows():
        entry_date = pd.to_datetime(trade['entry_date'])
        exit_date = pd.to_datetime(trade['exit_date'])
        entry_price = trade['entry_price']
        exit_price = trade['exit_price']
        is_win = trade['pnl'] > 0
        
        color = '#27ae60' if is_win else '#e74c3c'
        marker_entry = '^' if trade['direction'] == 'UP' else 'v'
        
        # Entry point
        ax.scatter(entry_date, entry_price, c=color, marker=marker_entry, s=120, zorder=5, edgecolors='black')
        # Exit point
        ax.scatter(exit_date, exit_price, c=color, marker='o', s=80, zorder=5, edgecolors='black')
        # Connect entry to exit
        ax.plot([entry_date, exit_date], [entry_price, exit_price], color=color, linestyle='--', alpha=0.6, linewidth=1.5)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price (₹)', fontsize=12)
    ax.set_title('Trade Execution - IRCON', fontsize=14, fontweight='bold')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.4)
    
    # Legend
    legend_elements = [
        Patch(facecolor='#27ae60', label='Winning Trade'),
        Patch(facecolor='#e74c3c', label='Losing Trade'),
        plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', markersize=10, label='Long Entry'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='gray', markersize=10, label='Short Entry')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'trade_execution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: trade_execution.png")


def plot_trade_scatter():
    """Scatter plot of trades: entry price vs P&L."""
    trades_path = OUTPUT_DIR / "trades.csv"
    if not trades_path.exists():
        return
    
    df = pd.read_csv(trades_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#27ae60' if x > 0 else '#e74c3c' for x in df['pnl']]
    sizes = np.abs(df['pnl']) / np.abs(df['pnl']).max() * 200 + 50
    
    ax.scatter(df['entry_price'], df['pnl'], c=colors, s=sizes, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Entry Price (₹)', fontsize=12)
    ax.set_ylabel('P&L (₹)', fontsize=12)
    ax.set_title('Trade Performance: Entry Price vs P&L', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4)
    
    legend_elements = [Patch(facecolor='#27ae60', label='Winning Trade'),
                      Patch(facecolor='#e74c3c', label='Losing Trade')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'trade_scatter.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: trade_scatter.png")


def plot_performance_donut():
    """Donut chart showing win/loss breakdown."""
    trades_path = OUTPUT_DIR / "trades.csv"
    if not trades_path.exists():
        return
    
    df = pd.read_csv(trades_path)
    
    wins = (df['pnl'] > 0).sum()
    losses = (df['pnl'] <= 0).sum()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    sizes = [wins, losses]
    labels = [f'Winners\n{wins}', f'Losers\n{losses}']
    colors = ['#2ecc71', '#e74c3c']
    explode = (0.02, 0.02)
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           startangle=90, explode=explode, pctdistance=0.75,
           wedgeprops=dict(width=0.4, edgecolor='white'))
    
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0, f'{wins+losses}\nTrades', ha='center', va='center', fontsize=16, fontweight='bold')
    
    ax.set_title('Trade Performance', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'performance_donut.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: performance_donut.png")


def plot_pnl_histogram():
    """Histogram of P&L distribution."""
    trades_path = OUTPUT_DIR / "trades.csv"
    if not trades_path.exists():
        return
    
    df = pd.read_csv(trades_path)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    n, bins_arr, patches = ax.hist(df['pnl'], bins=12, edgecolor='black', alpha=0.75)
    
    for i, patch in enumerate(patches):
        if bins_arr[i] >= 0:
            patch.set_facecolor('#27ae60')
        else:
            patch.set_facecolor('#e74c3c')
    
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax.axvline(x=df['pnl'].mean(), color='blue', linestyle='-', linewidth=2, label=f"Mean: ₹{df['pnl'].mean():.0f}")
    
    ax.set_xlabel('P&L (₹)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('P&L Distribution', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.4, axis='y')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'pnl_histogram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: pnl_histogram.png")


def plot_exit_reasons():
    """Bar chart for exit reasons."""
    trades_path = OUTPUT_DIR / "trades.csv"
    if not trades_path.exists():
        return
    
    df = pd.read_csv(trades_path)
    if 'exit_reason' not in df.columns:
        return
    
    exit_counts = df['exit_reason'].value_counts()
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(exit_counts)))
    bars = ax.barh(exit_counts.index, exit_counts.values, color=colors, edgecolor='black')
    
    for bar, val in zip(bars, exit_counts.values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height()/2, str(val), va='center', fontsize=11, fontweight='bold')
    
    ax.set_xlabel('Number of Trades', fontsize=12)
    ax.set_title('Exit Reason Breakdown', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.4, axis='x')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exit_reasons.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: exit_reasons.png")


def main():
    """Generate all visualizations."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Generating visualizations...")
    plot_cumulative_returns()
    plot_trade_execution()
    plot_trade_scatter()
    plot_performance_donut()
    plot_pnl_histogram()
    plot_exit_reasons()
    print("Done!")


if __name__ == "__main__":
    main()
