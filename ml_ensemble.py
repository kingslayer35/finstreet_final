"""
Ensemble Model: XGBoost + LightGBM with dynamic weighting
"""
import os
import logging
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Container for model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    log_loss_val: float
    
    def to_dict(self) -> Dict:
        return {
            'accuracy': round(self.accuracy, 4),
            'precision': round(self.precision, 4),
            'recall': round(self.recall, 4),
            'f1': round(self.f1, 4),
            'log_loss': round(self.log_loss_val, 4)
        }


class EnsembleTrainer:
    """
    Ensemble model trainer with XGBoost + LightGBM.
    Falls back to XGBoost-only if LightGBM is not available.
    """
    
    def __init__(self, xgb_weight: float = 0.3, lgbm_weight: float = 0.7):
        self.xgb_weight = xgb_weight
        self.lgbm_weight = lgbm_weight
        self.xgb_model = None
        self.lgbm_model = None
        self.feature_cols = None
        self.has_lgbm = self._check_lgbm()
        
    def _check_lgbm(self) -> bool:
        """Check if LightGBM is available."""
        try:
            import lightgbm
            return True
        except ImportError:
            logger.warning("LightGBM not available - using XGBoost only")
            return False
    
    def _create_xgb_model(self) -> XGBClassifier:
        """Create XGBoost classifier."""
        return XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1,
            verbosity=0
        )
    
    def _create_lgbm_model(self):
        """Create LightGBM classifier."""
        if not self.has_lgbm:
            return None
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1
        )
    
    def train(self, X: pd.DataFrame, y: np.ndarray) -> ModelMetrics:
        """
        Train ensemble model directly on full data (no warmup/walk-forward).
        """
        self.feature_cols = list(X.columns)
        
        # Create models
        self.xgb_model = self._create_xgb_model()
        if self.has_lgbm:
            self.lgbm_model = self._create_lgbm_model()
        
        # Train directly on full data - no warmup
        logger.info(f"Training on {len(X)} samples (no warmup)")
        self.xgb_model.fit(X, y, verbose=False)
        if self.has_lgbm and self.lgbm_model:
            self.lgbm_model.fit(X, y)
        
        # Get predictions for metrics
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        metrics = self._calculate_metrics(y, y_pred, y_proba)
        logger.info(f"Training Metrics: {metrics.to_dict()}")
        return metrics
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray) -> ModelMetrics:
        """Calculate comprehensive model metrics."""
        return ModelMetrics(
            accuracy=accuracy_score(y_true, y_pred),
            precision=precision_score(y_true, y_pred, zero_division=0),
            recall=recall_score(y_true, y_pred, zero_division=0),
            f1=f1_score(y_true, y_pred, zero_division=0),
            log_loss_val=log_loss(y_true, y_proba)
        )
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get ensemble probability predictions.
        Returns probability of class 1 (price going UP).
        """
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        
        if self.has_lgbm and self.lgbm_model:
            lgbm_proba = self.lgbm_model.predict_proba(X)[:, 1]
            return self.xgb_weight * xgb_proba + self.lgbm_weight * lgbm_proba
        return xgb_proba
    
    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Get binary predictions."""
        proba = self.predict_proba(X)
        return (proba > threshold).astype(int)
    
    def save(self, models_dir: str):
        """Save trained models."""
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(self.xgb_model, os.path.join(models_dir, 'xgb_model.pkl'))
        joblib.dump(self.feature_cols, os.path.join(models_dir, 'features.pkl'))
        if self.has_lgbm and self.lgbm_model:
            joblib.dump(self.lgbm_model, os.path.join(models_dir, 'lgbm_model.pkl'))
        logger.info(f"Models saved to {models_dir}")
    
    def load(self, models_dir: str):
        """Load trained models."""
        self.xgb_model = joblib.load(os.path.join(models_dir, 'xgb_model.pkl'))
        self.feature_cols = joblib.load(os.path.join(models_dir, 'features.pkl'))
        lgbm_path = os.path.join(models_dir, 'lgbm_model.pkl')
        if os.path.exists(lgbm_path):
            self.lgbm_model = joblib.load(lgbm_path)
            self.has_lgbm = True
        logger.info(f"Models loaded from {models_dir}")
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance from XGBoost."""
        if self.xgb_model is None or self.feature_cols is None:
            return pd.DataFrame()
        
        importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance


if __name__ == "__main__":
    # Test model training
    from data_loader import prepare_data
    from indicators import add_all_features, add_labels, get_feature_columns
    from config import RAW_DATA_PATH, DAILY_DATA_PATH, MODELS_DIR
    
    # Prepare data
    df = prepare_data(str(RAW_DATA_PATH), str(DAILY_DATA_PATH))
    df = add_all_features(df)
    df = add_labels(df, forward_days=1)
    df = df.dropna()
    
    # Get features and labels
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df['label'].values
    
    # Train
    trainer = EnsembleTrainer()
    metrics = trainer.train(X, y)
    trainer.save(str(MODELS_DIR))
    
    print(f"\nMetrics: {metrics.to_dict()}")
    print(f"\nTop Features:")
    print(trainer.get_feature_importance().head(10))
