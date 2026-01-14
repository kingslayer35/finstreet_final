"""
Configuration for IRCON Trading System
All parameters in one place for easy modification
"""
from dataclasses import dataclass
from typing import List
from pathlib import Path

# === Paths ===
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"

# === Stock ===
SYMBOL = "IRCON"
RAW_DATA_PATH = DATA_DIR / "ircon.csv"  # Hourly data
DAILY_DATA_PATH = DATA_DIR / "ircon_daily.csv"

# === Date Ranges ===
TRAIN_START = "2025-11-01"
TRAIN_END = "2025-12-31"

# Prediction dates (Jan 1-8, 2026 trading days)
PREDICT_DATES = [
    "2026-01-01",
    "2026-01-02", 
    "2026-01-03",
    "2026-01-06",
    "2026-01-07",
    "2026-01-08",
]

# === Model ===
@dataclass
class ModelConfig:
    # XGBoost parameters
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    
    # LightGBM parameters
    lgbm_n_estimators: int = 100
    lgbm_max_depth: int = 3
    lgbm_learning_rate: float = 0.1
    
    # Ensemble weights (30% XGBoost, 70% LightGBM)
    xgb_weight: float = 0.3
    lgbm_weight: float = 0.7
    
    random_state: int = 42

# === Risk Management (SIMPLIFIED) ===
@dataclass
class RiskConfig:
    initial_capital: float = 100_000.0
    
    # Position sizing - fixed
    base_risk_pct: float = 0.04      # 4% risk per trade
    max_risk_pct: float = 0.05       # 5% max risk per trade
    min_confidence: float = 0.55     # Confidence threshold
    
    # Stop loss & Take profit (ATR-based)
    stop_loss_atr: float = 1.5       # 1.5x ATR stop loss
    take_profit_atr: float = 3.0     # 3x ATR take profit

# === Feature Engineering ===
@dataclass  
class FeatureConfig:
    rsi_period: int = 14
    sma_short: int = 10
    sma_long: int = 20
    ema_period: int = 10
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    adx_period: int = 14

# === Backtest ===
@dataclass
class BacktestConfig:
    commission_pct: float = 0.001   # 0.1% commission
    slippage_pct: float = 0.0005    # 0.05% slippage

# === Instantiate configs ===
model_config = ModelConfig()
risk_config = RiskConfig()
feature_config = FeatureConfig()
backtest_config = BacktestConfig()
