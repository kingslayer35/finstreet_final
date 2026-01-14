"""
IRCON Trading System - Main Entry Point

Usage:
    python main.py              # Run full pipeline
    python main.py --backtest   # Run backtest only
    python main.py --predict    # Generate predictions only
"""
import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    RAW_DATA_PATH, DAILY_DATA_PATH, MODELS_DIR, OUTPUT_DIR,
    PREDICT_DATES, risk_config
)
from data_loader import prepare_data
from indicators import add_all_features, add_labels, get_feature_columns
from ml_ensemble import EnsembleTrainer
from signals import RiskManager, generate_signals
from backtester import Backtester, export_results


def step_1_prepare_data() -> pd.DataFrame:
    """Step 1: Convert hourly data to daily and save."""
    logger.info("=" * 50)
    logger.info("STEP 1: Preparing Data")
    logger.info("=" * 50)
    
    df = prepare_data(str(RAW_DATA_PATH), str(DAILY_DATA_PATH))
    logger.info(f"Daily data: {len(df)} rows from {df['date'].min()} to {df['date'].max()}")
    return df


def step_2_add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Step 2: Add technical indicators."""
    logger.info("=" * 50)
    logger.info("STEP 2: Feature Engineering")
    logger.info("=" * 50)
    
    df = add_all_features(df)
    feature_cols = get_feature_columns(df)
    logger.info(f"Added {len(feature_cols)} features")
    return df


def step_3_train_model(df: pd.DataFrame) -> EnsembleTrainer:
    """Step 3: Train ensemble model with triple-barrier labels."""
    logger.info("=" * 50)
    logger.info("STEP 3: Training Ensemble Model (INTERMEDIATE)")
    logger.info("=" * 50)
    
    # Try to use triple-barrier labeling (better for trading)
    try:
        from labeling import triple_barrier_labeling
        df_train = triple_barrier_labeling(df.copy(), profit_pct=0.03, stop_pct=0.02, max_hold=5)
        logger.info("Using triple-barrier labeling")
    except ImportError:
        # Fallback to simple labeling
        df_train = add_labels(df.copy(), forward_days=1)
        logger.info("Using simple direction labeling")
    
    df_train = df_train.dropna()
    
    feature_cols = get_feature_columns(df_train)
    X = df_train[feature_cols]
    y = df_train['label'].values
    
    logger.info(f"Training samples: {len(X)}")
    logger.info(f"Features: {len(feature_cols)}")
    logger.info(f"Label distribution: UP={y.sum()}, DOWN={len(y)-y.sum()}")
    
    # Train
    trainer = EnsembleTrainer()
    metrics = trainer.train(X, y)
    trainer.save(str(MODELS_DIR))
    
    logger.info(f"Model metrics: {metrics.to_dict()}")
    return trainer


def step_4_generate_signals(
    df: pd.DataFrame, 
    trainer: EnsembleTrainer
) -> pd.DataFrame:
    """Step 4: Generate trade signals with risk management."""
    logger.info("=" * 50)
    logger.info("STEP 4: Generating Trade Signals")
    logger.info("=" * 50)
    
    # Prepare features
    feature_cols = trainer.feature_cols
    df_pred = df.copy()
    
    # Get ML probabilities
    X = df_pred[feature_cols].fillna(0)
    probabilities = trainer.predict_proba(X)
    
    # Generate signals with risk management
    risk_manager = RiskManager(
        base_risk_pct=risk_config.base_risk_pct,
        max_risk_pct=risk_config.max_risk_pct,
        min_confidence=risk_config.min_confidence,
        stop_loss_atr=risk_config.stop_loss_atr,
        take_profit_atr=risk_config.take_profit_atr
    )
    
    signals = generate_signals(df_pred, probabilities, risk_manager)
    
    # Summary
    signal_counts = signals['signal'].value_counts()
    logger.info(f"Signal distribution: {signal_counts.to_dict()}")
    
    return signals


def step_5_run_backtest(df: pd.DataFrame, signals: pd.DataFrame):
    """Step 5: Run chronological backtest."""
    logger.info("=" * 50)
    logger.info("STEP 5: Running Backtest")
    logger.info("=" * 50)
    
    backtester = Backtester(initial_capital=risk_config.initial_capital)
    result = backtester.run(df, signals)
    
    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)
    for key, value in result.to_dict().items():
        print(f"  {key:20s}: {value}")
    print("=" * 60)
    
    # Export
    export_results(result, str(OUTPUT_DIR))
    
    return result


def step_6_generate_predictions(
    df: pd.DataFrame,
    trainer: EnsembleTrainer
) -> pd.DataFrame:
    """Step 6: Generate predictions for Jan 1-8, 2026."""
    logger.info("=" * 50)
    logger.info("STEP 6: Generating Predictions (Jan 1-8, 2026)")
    logger.info("=" * 50)
    
    # Use last available data point as base
    last_row = df.iloc[-1:].copy()
    feature_cols = trainer.feature_cols
    
    # Risk manager
    risk_manager = RiskManager(
        base_risk_pct=risk_config.base_risk_pct,
        max_risk_pct=risk_config.max_risk_pct,
        min_confidence=risk_config.min_confidence,
        stop_loss_atr=risk_config.stop_loss_atr,
        take_profit_atr=risk_config.take_profit_atr
    )
    
    predictions = []
    
    for pred_date in PREDICT_DATES:
        # Get prediction using latest features
        X = last_row[feature_cols].fillna(0)
        probability = trainer.predict_proba(X)[0]
        
        signal = risk_manager.generate_signal(
            date=pred_date,
            close_price=last_row['close'].values[0],
            ml_probability=probability,
            atr=last_row['ATR'].values[0] if 'ATR' in last_row.columns else 3.0
        )
        
        predictions.append({
            'date': signal.date,
            'signal': signal.signal,
            'direction': signal.direction,
            'confidence': f"{signal.confidence:.2%}",
            'position_size': f"{signal.position_size:.2%}",
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit
        })
    
    pred_df = pd.DataFrame(predictions)
    
    # Print predictions
    print("\n" + "=" * 80)
    print("PREDICTIONS FOR JAN 1-8, 2026")
    print("=" * 80)
    print(pred_df.to_string(index=False))
    print("=" * 80)
    
    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(OUTPUT_DIR / 'predictions.csv', index=False)
    logger.info(f"Predictions saved to {OUTPUT_DIR / 'predictions.csv'}")
    
    return pred_df


def run_full_pipeline():
    """Run the complete trading pipeline."""
    logger.info("Starting IRCON Trading System Pipeline")
    logger.info(f"Raw data: {RAW_DATA_PATH}")
    
    # Execute pipeline
    df = step_1_prepare_data()
    df = step_2_add_features(df)
    trainer = step_3_train_model(df)
    signals = step_4_generate_signals(df, trainer)
    result = step_5_run_backtest(df, signals)
    predictions = step_6_generate_predictions(df, trainer)
    
    logger.info("Pipeline completed successfully!")
    return result, predictions


def main():
    parser = argparse.ArgumentParser(description='IRCON Trading System')
    parser.add_argument('--backtest', action='store_true', help='Run backtest only')
    parser.add_argument('--predict', action='store_true', help='Generate predictions only')
    args = parser.parse_args()
    
    if args.backtest or args.predict:
        # Load existing data and model
        df = pd.read_csv(DAILY_DATA_PATH)
        df['date'] = pd.to_datetime(df['date'])
        df = add_all_features(df)
        
        trainer = EnsembleTrainer()
        trainer.load(str(MODELS_DIR))
        
        if args.backtest:
            signals = step_4_generate_signals(df, trainer)
            step_5_run_backtest(df, signals)
        
        if args.predict:
            step_6_generate_predictions(df, trainer)
    else:
        run_full_pipeline()


if __name__ == "__main__":
    main()
