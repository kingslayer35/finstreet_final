"""
Backtesting Engine: Chronological simulation with performance metrics
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Record of a single trade."""
    entry_date: str
    exit_date: str
    direction: str
    entry_price: float
    exit_price: float
    quantity: int
    pnl: float
    pnl_pct: float
    exit_reason: str  # STOP_LOSS, TAKE_PROFIT, END_OF_DAY
    confidence: float
    position_size: float


@dataclass
class BacktestResult:
    """Container for backtest results."""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    
    def to_dict(self) -> Dict:
        # Handle infinite profit factor (no losses)
        if self.losing_trades == 0 and self.winning_trades > 0:
            pf_display = "∞ (no losses)"
        else:
            pf_display = f"{self.profit_factor:.2f}"
        
        return {
            'Initial Capital': f"₹{self.initial_capital:,.0f}",
            'Final Capital': f"₹{self.final_capital:,.0f}",
            'Total Return': f"₹{self.total_return:,.0f}",
            'Total Return %': f"{self.total_return_pct:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Max Drawdown %': f"{self.max_drawdown_pct:.2%}",
            'Win Rate': f"{self.win_rate:.1%}",
            'Total Trades': self.total_trades,
            'Winning Trades': self.winning_trades,
            'Losing Trades': self.losing_trades,
            'Profit Factor': pf_display
        }


class Backtester:
    """
    Chronological walk-forward backtester.
    
    Features:
    - Simulates trades day by day
    - Uses stop loss and take profit from risk manager
    - Tracks equity curve and drawdown
    - Calculates comprehensive metrics
    """
    
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
    
    def run(
        self,
        df: pd.DataFrame,
        signals: pd.DataFrame
    ) -> BacktestResult:
        """
        Run backtest on historical data with generated signals.
        
        Args:
            df: OHLCV data with date column
            signals: Signal dataframe from risk_management.generate_signals()
        
        Returns:
            BacktestResult with all metrics and trades
        """
        capital = self.initial_capital
        equity = []
        trades = []
        
        # Merge signals with price data
        df = df.copy()
        df['date_str'] = df['date'].astype(str).str[:10]
        signals['date_str'] = signals['date'].astype(str).str[:10]
        
        merged = df.merge(signals, on='date_str', how='left')
        
        for i in range(len(merged) - 1):
            row = merged.iloc[i]
            next_row = merged.iloc[i + 1]
            
            # Skip if no signal or HOLD
            if pd.isna(row.get('signal')) or row['signal'] == 'HOLD':
                equity.append({'date': row['date_str'], 'equity': capital})
                continue
            
            # Get trade parameters
            signal = row['signal']
            direction = row['direction']
            entry_price = row['close']
            stop_loss = row.get('stop_loss', 0)
            take_profit = row.get('take_profit', 0)
            position_size = row.get('position_size', 0.02)
            confidence = row.get('confidence', 0.5)
            
            # Calculate position
            risk_amount = capital * position_size
            
            # Apply slippage to entry
            if direction == 'UP':
                entry_price = entry_price * (1 + self.slippage_pct)
            else:
                entry_price = entry_price * (1 - self.slippage_pct)
            
            # Calculate quantity
            quantity = int(risk_amount / entry_price)
            if quantity == 0:
                equity.append({'date': row['date_str'], 'equity': capital})
                continue
            
            # Simulate next day: check if stop/target hit using high/low
            next_high = next_row['high']
            next_low = next_row['low']
            next_close = next_row['close']
            
            exit_price = next_close
            exit_reason = 'END_OF_DAY'
            
            # Get trailing stop settings (if available)
            use_trailing = row.get('use_trailing', False)
            trailing_stop = row.get('trailing_stop', 0)
            
            if direction == 'UP':
                # Update trailing stop if price moved favorably
                if use_trailing and trailing_stop > 0:
                    if next_high > entry_price:
                        # Trail stop up
                        new_trail = next_high - (entry_price - stop_loss)
                        trailing_stop = max(trailing_stop, new_trail)
                
                # Check trailing stop first (if active)
                if use_trailing and trailing_stop > 0 and next_low <= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = 'TRAIL_STOP'
                # Check hard stop loss
                elif stop_loss > 0 and next_low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                # Check take profit
                elif take_profit > 0 and next_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TAKE_PROFIT'
                
                pnl = (exit_price - entry_price) * quantity
            else:  # SHORT
                # Update trailing stop if price moved favorably
                if use_trailing and trailing_stop > 0:
                    if next_low < entry_price:
                        # Trail stop down
                        new_trail = next_low + (stop_loss - entry_price)
                        trailing_stop = min(trailing_stop, new_trail)
                
                # Check trailing stop first (if active)
                if use_trailing and trailing_stop > 0 and next_high >= trailing_stop:
                    exit_price = trailing_stop
                    exit_reason = 'TRAIL_STOP'
                # Check hard stop loss
                elif stop_loss > 0 and next_high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = 'STOP_LOSS'
                # Check take profit
                elif take_profit > 0 and next_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = 'TAKE_PROFIT'
                
                pnl = (entry_price - exit_price) * quantity
            
            # Apply commission
            commission = (entry_price + exit_price) * quantity * self.commission_pct
            pnl -= commission
            
            # Update capital
            capital += pnl
            
            # Record trade
            trade = Trade(
                entry_date=row['date_str'],
                exit_date=next_row['date_str'],
                direction=direction,
                entry_price=round(entry_price, 2),
                exit_price=round(exit_price, 2),
                quantity=quantity,
                pnl=round(pnl, 2),
                pnl_pct=round(pnl / (entry_price * quantity), 4),
                exit_reason=exit_reason,
                confidence=confidence,
                position_size=position_size
            )
            trades.append(trade)
            equity.append({'date': row['date_str'], 'equity': capital})
        
        # Add final equity point
        equity.append({'date': merged.iloc[-1]['date_str'], 'equity': capital})
        
        # Calculate metrics
        equity_df = pd.DataFrame(equity)
        result = self._calculate_metrics(trades, equity_df)
        
        return result
    
    def _calculate_metrics(
        self, 
        trades: List[Trade], 
        equity_df: pd.DataFrame
    ) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        
        # Basic metrics
        initial = self.initial_capital
        final = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial
        total_return = final - initial
        total_return_pct = total_return / initial
        
        # Trade metrics
        total_trades = len(trades)
        winning_trades = sum(1 for t in trades if t.pnl > 0)
        losing_trades = sum(1 for t in trades if t.pnl <= 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl <= 0]
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        # Profit factor
        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = equity_df['equity'] - equity_df['peak']
        equity_df['drawdown_pct'] = equity_df['drawdown'] / equity_df['peak']
        max_drawdown = abs(equity_df['drawdown'].min())
        max_drawdown_pct = abs(equity_df['drawdown_pct'].min())
        
        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equity_df) > 1:
            equity_df['returns'] = equity_df['equity'].pct_change()
            daily_returns = equity_df['returns'].dropna()
            if len(daily_returns) > 0 and daily_returns.std() > 0:
                sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        return BacktestResult(
            initial_capital=initial,
            final_capital=final,
            total_return=total_return,
            total_return_pct=total_return_pct,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown_pct,
            win_rate=win_rate,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor if profit_factor != float('inf') else 0,
            trades=trades,
            equity_curve=equity_df
        )


def export_results(result: BacktestResult, output_dir: str):
    """Export backtest results to CSV files."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Export trades
    trades_df = pd.DataFrame([
        {
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'direction': t.direction,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'pnl': t.pnl,
            'pnl_pct': f"{t.pnl_pct:.2%}",
            'exit_reason': t.exit_reason,
            'confidence': f"{t.confidence:.2%}"
        }
        for t in result.trades
    ])
    trades_df.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)
    
    # Export equity curve
    result.equity_curve[['date', 'equity']].to_csv(
        os.path.join(output_dir, 'equity_curve.csv'), index=False
    )
    
    # Export summary
    summary = pd.DataFrame([result.to_dict()]).T
    summary.columns = ['Value']
    summary.to_csv(os.path.join(output_dir, 'backtest_summary.csv'))
    
    logger.info(f"Results exported to {output_dir}")


if __name__ == "__main__":
    # Test backtester with dummy data
    print("Backtester module loaded successfully")
