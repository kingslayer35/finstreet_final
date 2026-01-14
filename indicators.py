"""
Feature Engineering: Technical indicators for ML model
"""
import pandas as pd
import numpy as np
from typing import List


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Relative Strength Index - momentum oscillator."""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def add_moving_averages(df: pd.DataFrame, short: int = 10, long: int = 20) -> pd.DataFrame:
    """Simple and Exponential Moving Averages."""
    df['SMA_short'] = df['close'].rolling(window=short).mean()
    df['SMA_long'] = df['close'].rolling(window=long).mean()
    df['EMA_short'] = df['close'].ewm(span=short, adjust=False).mean()
    
    # Price relative to MAs
    df['price_to_sma_short'] = (df['close'] - df['SMA_short']) / df['SMA_short'] * 100
    df['price_to_sma_long'] = (df['close'] - df['SMA_long']) / df['SMA_long'] * 100
    
    # MA crossover signal
    df['sma_crossover'] = (df['SMA_short'] > df['SMA_long']).astype(int)
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD - trend following momentum indicator."""
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average True Range - volatility measure."""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=period).mean()
    df['ATR_pct'] = df['ATR'] / df['close'] * 100
    return df


def add_bollinger_bands(df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands - volatility bands around MA."""
    df['BB_mid'] = df['close'].rolling(window=period).mean()
    rolling_std = df['close'].rolling(window=period).std()
    df['BB_upper'] = df['BB_mid'] + (std * rolling_std)
    df['BB_lower'] = df['BB_mid'] - (std * rolling_std)
    df['BB_pct'] = (df['close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'] + 1e-10)
    return df


def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """ADX - trend strength indicator."""
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - df['close'].shift()).abs(),
        (df['low'] - df['close'].shift()).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-10))
    
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10))
    df['ADX'] = dx.rolling(window=period).mean()
    df['DI_plus'] = plus_di
    df['DI_minus'] = minus_di
    df['DI_spread'] = plus_di - minus_di
    return df


def add_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Price returns over different periods."""
    df['return_1d'] = df['close'].pct_change(1)
    df['return_3d'] = df['close'].pct_change(3)
    df['return_5d'] = df['close'].pct_change(5)
    df['volatility_5d'] = df['return_1d'].rolling(5).std()
    return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features."""
    df['volume_sma'] = df['volume'].rolling(window=10).mean()
    df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)
    
    # On-Balance Volume (OBV) - tracks cumulative buying/selling pressure
    obv = [0]
    for i in range(1, len(df)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            obv.append(obv[-1] + df['volume'].iloc[i])
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            obv.append(obv[-1] - df['volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv
    df['OBV_slope'] = df['OBV'].diff(5) / (df['OBV'].shift(5).abs() + 1e-10) * 100
    
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced statistical and pattern features."""
    
    # Z-scores for mean reversion signals
    df['close_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)
    df['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / (df['volume'].rolling(20).std() + 1e-10)
    
    # Kaufman Efficiency Ratio - measures trend efficiency
    change = abs(df['close'] - df['close'].shift(10))
    volatility = df['close'].diff().abs().rolling(10).sum()
    df['KER'] = change / (volatility + 1e-10)
    
    # Candlestick body analysis
    df['body_size'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-10)
    df['upper_wick'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (df['high'] - df['low'] + 1e-10)
    df['lower_wick'] = (df[['close', 'open']].min(axis=1) - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    # Daily range as % of price
    df['daily_range'] = (df['high'] - df['low']) / df['close'] * 100
    
    # Realized volatility (5-day)
    df['realized_vol'] = df['return_1d'].rolling(5).std() * np.sqrt(252) * 100
    
    # Rate of Change
    df['ROC_5'] = df['close'].pct_change(5) * 100
    df['ROC_10'] = df['close'].pct_change(10) * 100
    
    return df


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all technical indicators (INTERMEDIATE complexity)."""
    df = df.copy()
    
    # Basic indicators
    df = add_rsi(df)
    df = add_moving_averages(df)
    df = add_macd(df)
    df = add_atr(df)
    df = add_bollinger_bands(df)
    df = add_adx(df)
    df = add_returns(df)
    df = add_volume_features(df)
    
    # Advanced features (new!)
    df = add_advanced_features(df)
    
    # === MOMENTUM CONFLUENCE (for stronger signals) ===
    # Trend direction signals
    df['trend_up'] = (
        (df['close'] > df['SMA_long']) & 
        (df['SMA_short'] > df['SMA_long']) &
        (df['MACD'] > df['MACD_signal'])
    ).astype(int)
    
    df['trend_down'] = (
        (df['close'] < df['SMA_long']) & 
        (df['SMA_short'] < df['SMA_long']) &
        (df['MACD'] < df['MACD_signal'])
    ).astype(int)
    
    # Strong momentum signals
    df['momentum_score'] = (
        (df['RSI'] > 50).astype(int) +
        (df['MACD_hist'] > 0).astype(int) +
        (df['close'] > df['SMA_short']).astype(int) +
        (df['DI_plus'] > df['DI_minus']).astype(int)
    )
    
    # Breakout detection
    df['high_20'] = df['high'].rolling(20).max()
    df['low_20'] = df['low'].rolling(20).min()
    df['breakout_up'] = (df['close'] >= df['high_20'].shift(1)).astype(int)
    df['breakout_down'] = (df['close'] <= df['low_20'].shift(1)).astype(int)
    
    # Trend strength score
    df['trend_score'] = (
        (df['ADX'] > 25).astype(int) * 2 +
        df['trend_up'] +
        df['breakout_up']
    )
    
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Get list of feature columns (exclude raw OHLCV and date)."""
    exclude = [
        'date', 'open', 'high', 'low', 'close', 'volume', 
        'label', 'future_return', 't1',  # Labeling columns
        'high_20', 'low_20'  # Rolling max/min used for breakout, not features
    ]
    return [col for col in df.columns if col not in exclude and not df[col].isna().all()]


def add_labels(df: pd.DataFrame, forward_days: int = 1) -> pd.DataFrame:
    """
    Create labels for ML model.
    Label = 1 if price goes UP in next N days, else 0
    """
    df = df.copy()
    df['future_return'] = df['close'].shift(-forward_days) / df['close'] - 1
    df['label'] = (df['future_return'] > 0).astype(int)
    return df


if __name__ == "__main__":
    # Test features
    from data_prep import prepare_data
    from config import RAW_DATA_PATH, DAILY_DATA_PATH
    
    df = prepare_data(str(RAW_DATA_PATH), str(DAILY_DATA_PATH))
    df = add_all_features(df)
    df = add_labels(df)
    
    print(f"Features: {get_feature_columns(df)}")
    print(f"Shape: {df.shape}")
    print(df.head())
