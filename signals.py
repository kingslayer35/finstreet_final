"""
Risk Management: Simple position sizing and trade signals
"""
import logging
from dataclasses import dataclass
from typing import Dict

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    """Container for a trade signal."""
    date: str
    signal: str          # BUY, SELL, HOLD
    direction: str       # UP, DOWN
    confidence: float    # ML probability
    position_size: float # Fraction of capital to risk
    stop_loss: float     # Stop loss price
    take_profit: float   # Take profit price
    entry_price: float   # Expected entry price


class RiskManager:
    """
    Simple risk management with fixed position sizing.
    """
    
    def __init__(
        self,
        base_risk_pct: float = 0.04,
        max_risk_pct: float = 0.05,
        min_confidence: float = 0.55,
        stop_loss_atr: float = 1.5,
        take_profit_atr: float = 3.0
    ):
        self.base_risk_pct = base_risk_pct
        self.max_risk_pct = max_risk_pct
        self.min_confidence = min_confidence
        self.stop_loss_atr = stop_loss_atr
        self.take_profit_atr = take_profit_atr
    
    def calculate_position_size(self, confidence: float) -> float:
        """Fixed position sizing based on confidence threshold."""
        if confidence < self.min_confidence:
            return 0.0
        return self.base_risk_pct
    
    def should_trade(self, confidence: float) -> bool:
        """Trade if confidence exceeds threshold."""
        return confidence >= self.min_confidence
    
    def calculate_stops(
        self, 
        entry_price: float, 
        atr: float, 
        direction: str
    ) -> Dict[str, float]:
        """Calculate ATR-based stop loss and take profit."""
        sl_distance = atr * self.stop_loss_atr
        tp_distance = atr * self.take_profit_atr
        
        if direction == "UP":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        return {
            'stop_loss': round(stop_loss, 2),
            'take_profit': round(take_profit, 2)
        }
    
    def generate_signal(
        self,
        date: str,
        close_price: float,
        ml_probability: float,
        atr: float,
        adx: float = 0  # Unused but kept for compatibility
    ) -> TradeSignal:
        """Generate a trade signal with risk parameters."""
        # Determine direction
        direction = "UP" if ml_probability > 0.5 else "DOWN"
        confidence = ml_probability if direction == "UP" else (1 - ml_probability)
        
        # Check if we should trade
        if not self.should_trade(confidence):
            return TradeSignal(
                date=date,
                signal="HOLD",
                direction=direction,
                confidence=round(confidence, 4),
                position_size=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                entry_price=close_price
            )
        
        # Calculate position size and stops
        position_size = self.calculate_position_size(confidence)
        stops = self.calculate_stops(close_price, atr, direction)
        signal = "BUY" if direction == "UP" else "SELL"
        
        return TradeSignal(
            date=date,
            signal=signal,
            direction=direction,
            confidence=round(confidence, 4),
            position_size=position_size,
            stop_loss=stops['stop_loss'],
            take_profit=stops['take_profit'],
            entry_price=close_price
        )


def generate_signals(
    df: pd.DataFrame,
    ml_probabilities: np.ndarray,
    risk_manager: RiskManager
) -> pd.DataFrame:
    """Generate trade signals based on ML predictions."""
    signals = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        ml_prob = ml_probabilities[i]
        atr = row['ATR'] if 'ATR' in row else 3.0
        
        signal = risk_manager.generate_signal(
            date=str(row['date'].date()) if hasattr(row['date'], 'date') else str(row['date']),
            close_price=row['close'],
            ml_probability=ml_prob,
            atr=atr
        )
        signals.append(signal)
    
    # Convert to DataFrame
    signal_df = pd.DataFrame([
        {
            'date': s.date,
            'signal': s.signal,
            'direction': s.direction,
            'confidence': s.confidence,
            'position_size': s.position_size,
            'stop_loss': s.stop_loss,
            'take_profit': s.take_profit,
            'entry_price': s.entry_price
        }
        for s in signals
    ])
    
    return signal_df
