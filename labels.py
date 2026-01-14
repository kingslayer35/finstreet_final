"""
Triple-Barrier Labeling for ML Training
Creates better labels for trading: considers profit, stop-loss, and time barriers.
"""
import pandas as pd
import numpy as np
from typing import Tuple


def triple_barrier_labeling(
    df: pd.DataFrame,
    profit_pct: float = 0.03,    # 3% profit target
    stop_pct: float = 0.02,      # 2% stop loss
    max_hold: int = 5,           # Max 5 bars holding
    min_hold: int = 1            # Min 1 bar holding
) -> pd.DataFrame:
    """
    Triple-barrier labeling method (from Advances in Financial ML).
    
    Creates labels based on which barrier is touched first:
    - Upper barrier (profit): Label = 1 (BUY signal was correct)
    - Lower barrier (stop):   Label = 0 (SELL signal was correct)  
    - Time barrier:           Label based on final return
    
    This is better than simple next-day direction because it accounts
    for risk/reward during the holding period.
    """
    df = df.copy()
    n = len(df)
    
    labels = []
    t1_list = []  # Time of barrier touch
    returns_list = []
    
    for i in range(n):
        if i >= n - min_hold:
            labels.append(np.nan)
            t1_list.append(np.nan)
            returns_list.append(np.nan)
            continue
        
        entry_price = df.iloc[i]['close']
        upper_barrier = entry_price * (1 + profit_pct)
        lower_barrier = entry_price * (1 - stop_pct)
        
        # Look ahead up to max_hold bars
        end_idx = min(i + max_hold + 1, n)
        future_bars = df.iloc[i+1:end_idx]
        
        if len(future_bars) == 0:
            labels.append(np.nan)
            t1_list.append(np.nan)
            returns_list.append(np.nan)
            continue
        
        # Check which barrier is hit first
        label = None
        t1 = None
        
        for j, (idx, bar) in enumerate(future_bars.iterrows()):
            bar_num = j + 1
            
            # Check upper barrier (profit hit)
            if bar['high'] >= upper_barrier:
                label = 1
                t1 = bar_num
                break
            
            # Check lower barrier (stop hit)
            if bar['low'] <= lower_barrier:
                label = 0
                t1 = bar_num
                break
        
        # If no barrier hit, use time barrier
        if label is None:
            final_price = future_bars.iloc[-1]['close']
            final_return = (final_price - entry_price) / entry_price
            label = 1 if final_return > 0 else 0
            t1 = len(future_bars)
        
        labels.append(label)
        t1_list.append(t1)
        
        # Calculate actual return
        exit_price = df.iloc[min(i + t1, n-1)]['close'] if t1 else entry_price
        returns_list.append((exit_price - entry_price) / entry_price)
    
    df['label'] = labels
    df['t1'] = t1_list
    df['future_return'] = returns_list
    
    return df


def calculate_sample_weights(df: pd.DataFrame, decay: float = 0.5) -> pd.Series:
    """
    Calculate sample weights for training.
    More recent samples get higher weights.
    """
    n = len(df)
    weights = np.exp(decay * np.arange(n) / n)
    weights = weights / weights.sum() * n  # Normalize
    return pd.Series(weights, index=df.index)


if __name__ == "__main__":
    # Test
    from data_loader import prepare_data
    from config import RAW_DATA_PATH, DAILY_DATA_PATH
    
    df = prepare_data(str(RAW_DATA_PATH), str(DAILY_DATA_PATH))
    df = triple_barrier_labeling(df)
    
    print(f"Label distribution:")
    print(df['label'].value_counts())
    print(f"\nAverage holding period: {df['t1'].mean():.1f} bars")
