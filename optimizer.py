#!/usr/bin/env python3
"""
🔧 TPE OPTIMIZER - Find the Best Parameters
=============================================

Uses Optuna's TPE (Tree-structured Parzen Estimator) to optimize
your trading strategy parameters on historical data.

WORKFLOW:
1. Load historical data (OHLC CSV files)
2. Run backtests with different parameter combinations
3. Find parameters that maximize profit while respecting 5ers rules
4. Save best parameters for live trading

USAGE:
    python optimizer.py --trials 100 --period 2023-2025
    python optimizer.py --status  # Check optimization progress

5ERS RULES ENFORCED:
- Max 5% daily drawdown (DDD)
- Max 10% total drawdown (TDD) from starting balance
- Starting balance: $100,000

Author: Trading Bot Template
Version: 1.0.0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from tqdm import tqdm

# Optuna for optimization
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("❌ Optuna not installed!")
    print("   Run: pip install optuna")
    sys.exit(1)

# Import strategy
from strategy_template import (
    StrategyParams,
    Signal,
    generate_signal,
    calculate_lot_size,
    get_default_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OptimizerConfig:
    """Optimization configuration."""
    # Account
    initial_balance: float = 100_000  # 5ers $100K account
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5ERS / PROP FIRM RISK RULES (CRITICAL - DO NOT MODIFY)
    # ═══════════════════════════════════════════════════════════════════════════
    # These rules MUST be followed to pass prop firm challenges
    
    # Daily Drawdown (DDD) - measured from day start balance
    max_daily_dd_pct: float = 5.0           # HARD LIMIT - breach = fail
    daily_dd_warning_pct: float = 2.0       # Warning threshold - alert
    daily_dd_reduce_pct: float = 3.2        # TRADE HALF at this level
    daily_dd_halt_pct: float = 4.0          # Stop ALL trading for the day
    
    # Total Drawdown (TDD) - STATIC from initial balance (NOT trailing!)
    max_total_dd_pct: float = 10.0          # HARD LIMIT - breach = fail
    total_dd_warning_pct: float = 5.0       # Warning threshold
    total_dd_reduce_pct: float = 7.0        # Reduce to minimum risk
    
    # Risk reduction multipliers
    reduced_risk_multiplier: float = 0.5    # Half size at reduce threshold
    emergency_risk_multiplier: float = 0.25 # Quarter size near limit
    
    # Risk
    base_risk_pct: float = 0.6        # Base risk per trade
    
    # Data paths
    data_dir: Path = Path("data/ohlcv")
    
    # Output
    output_dir: Path = Path("optimization_output")
    
    # Optimization
    n_trials: int = 100
    study_name: str = "trading_bot_optimization"
    
    # Period
    start_date: str = "2023-01-01"
    end_date: str = "2025-12-31"


# Contract specifications for position sizing
CONTRACT_SPECS = {
    "EURUSD": {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0},
    "GBPUSD": {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0},
    "USDJPY": {"pip_size": 0.01, "pip_value": 6.67, "commission": 4.0},
    "AUDUSD": {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0},
    "NZDUSD": {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0},
    "USDCAD": {"pip_size": 0.0001, "pip_value": 7.5, "commission": 4.0},
    "USDCHF": {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0},
    "EURJPY": {"pip_size": 0.01, "pip_value": 6.67, "commission": 4.0},
    "GBPJPY": {"pip_size": 0.01, "pip_value": 6.67, "commission": 4.0},
    "EURGBP": {"pip_size": 0.0001, "pip_value": 12.5, "commission": 4.0},
    "XAUUSD": {"pip_size": 0.01, "pip_value": 1.0, "commission": 4.0},
}

DEFAULT_SPEC = {"pip_size": 0.0001, "pip_value": 10.0, "commission": 4.0}

# Symbols to trade (customize this list!)
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
    "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
]


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_ohlc_data(symbol: str, timeframe: str = "D1", data_dir: Path = Path("data/ohlcv")) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.
    
    Expected file format: {SYMBOL}_{TIMEFRAME}*.csv
    Columns: time (or date), open, high, low, close
    """
    # Try different file patterns
    patterns = [
        f"{symbol}_{timeframe}*.csv",
        f"{symbol.replace('USD', '_USD')}_{timeframe}*.csv",
        f"{symbol}*.csv",
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            df = pd.read_csv(files[0])
            
            # Normalize column names
            df.columns = df.columns.str.lower()
            
            # Parse dates
            date_col = 'time' if 'time' in df.columns else 'date'
            if date_col in df.columns:
                df['time'] = pd.to_datetime(df[date_col])
                df = df.sort_values('time')
            
            return df
    
    return pd.DataFrame()


def prepare_candles(df: pd.DataFrame, start_date: str, end_date: str) -> List[Dict]:
    """Convert DataFrame to list of candle dictionaries."""
    if df.empty:
        return []
    
    # Filter by date
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    mask = (df['time'] >= start) & (df['time'] <= end)
    df = df[mask].copy()
    
    candles = []
    for _, row in df.iterrows():
        candles.append({
            'time': row['time'],
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
        })
    
    return candles


# ═══════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    """Completed trade record."""
    symbol: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    lot_size: float
    pnl: float
    pnl_pct: float
    exit_reason: str


@dataclass
class BacktestResult:
    """Backtest result summary."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    final_balance: float
    max_daily_dd_pct: float
    max_total_dd_pct: float
    sharpe_ratio: float
    profit_factor: float
    trades: List[Trade]
    daily_pnl: Dict[str, float]
    
    def is_valid(self, config: OptimizerConfig) -> bool:
        """Check if result passes 5ers rules."""
        return (
            self.max_daily_dd_pct <= config.max_daily_dd_pct and
            self.max_total_dd_pct <= config.max_total_dd_pct
        )


def simulate_trade(
    signal: Signal,
    candles: List[Dict],
    start_idx: int,
    balance: float,
    params: StrategyParams,
    spec: Dict
) -> Tuple[Optional[Trade], int, float]:
    """
    Simulate a single trade with 3-TP partial exit system.
    
    Returns: (Trade or None, bars_held, final_pnl)
    """
    entry_price = signal.entry
    stop_loss = signal.stop_loss
    tp1 = signal.tp1
    tp2 = signal.tp2
    tp3 = signal.tp3
    direction = signal.direction
    
    # Calculate lot size
    lot_size = calculate_lot_size(
        balance=balance,
        risk_pct=params.risk_per_trade_pct,
        entry=entry_price,
        stop_loss=stop_loss,
        pip_value=spec['pip_value'],
        pip_size=spec['pip_size']
    )
    
    # Track partial positions
    remaining_pct = 1.0
    total_pnl = 0.0
    exit_reason = "max_bars"
    exit_price = entry_price
    exit_time = signal.signal_time
    
    tp1_hit = False
    tp2_hit = False
    
    # Simulate bar by bar
    max_bars = 50  # Max 50 bars holding period
    for i in range(start_idx + 1, min(start_idx + max_bars + 1, len(candles))):
        candle = candles[i]
        high = candle['high']
        low = candle['low']
        
        # Check stop loss
        if direction == 'bullish' and low <= stop_loss:
            exit_reason = "stop_loss"
            exit_price = stop_loss
            exit_time = candle['time']
            
            pips = (stop_loss - entry_price) / spec['pip_size']
            pnl = pips * spec['pip_value'] * lot_size * remaining_pct
            total_pnl += pnl - (spec['commission'] * lot_size * remaining_pct)
            break
            
        elif direction == 'bearish' and high >= stop_loss:
            exit_reason = "stop_loss"
            exit_price = stop_loss
            exit_time = candle['time']
            
            pips = (entry_price - stop_loss) / spec['pip_size']
            pnl = pips * spec['pip_value'] * lot_size * remaining_pct
            total_pnl += pnl - (spec['commission'] * lot_size * remaining_pct)
            break
        
        # Check TP1
        if not tp1_hit:
            if (direction == 'bullish' and high >= tp1) or (direction == 'bearish' and low <= tp1):
                tp1_hit = True
                close_pct = params.tp1_close_pct
                
                if direction == 'bullish':
                    pips = (tp1 - entry_price) / spec['pip_size']
                else:
                    pips = (entry_price - tp1) / spec['pip_size']
                
                pnl = pips * spec['pip_value'] * lot_size * close_pct
                total_pnl += pnl - (spec['commission'] * lot_size * close_pct)
                remaining_pct -= close_pct
                
                # Move SL to breakeven
                stop_loss = entry_price
        
        # Check TP2
        if tp1_hit and not tp2_hit:
            if (direction == 'bullish' and high >= tp2) or (direction == 'bearish' and low <= tp2):
                tp2_hit = True
                close_pct = params.tp2_close_pct
                
                if direction == 'bullish':
                    pips = (tp2 - entry_price) / spec['pip_size']
                else:
                    pips = (entry_price - tp2) / spec['pip_size']
                
                pnl = pips * spec['pip_value'] * lot_size * close_pct
                total_pnl += pnl - (spec['commission'] * lot_size * close_pct)
                remaining_pct -= close_pct
                
                # Trail SL to TP1
                stop_loss = tp1 if direction == 'bullish' else tp1
        
        # Check TP3
        if tp1_hit and tp2_hit and remaining_pct > 0:
            if (direction == 'bullish' and high >= tp3) or (direction == 'bearish' and low <= tp3):
                exit_reason = "tp3"
                exit_price = tp3
                exit_time = candle['time']
                
                if direction == 'bullish':
                    pips = (tp3 - entry_price) / spec['pip_size']
                else:
                    pips = (entry_price - tp3) / spec['pip_size']
                
                pnl = pips * spec['pip_value'] * lot_size * remaining_pct
                total_pnl += pnl - (spec['commission'] * lot_size * remaining_pct)
                remaining_pct = 0
                break
    
    # Close any remaining position at last price
    if remaining_pct > 0:
        last_candle = candles[min(start_idx + max_bars, len(candles) - 1)]
        exit_price = last_candle['close']
        exit_time = last_candle['time']
        
        if direction == 'bullish':
            pips = (exit_price - entry_price) / spec['pip_size']
        else:
            pips = (entry_price - exit_price) / spec['pip_size']
        
        pnl = pips * spec['pip_value'] * lot_size * remaining_pct
        total_pnl += pnl - (spec['commission'] * lot_size * remaining_pct)
    
    trade = Trade(
        symbol=signal.symbol,
        direction=direction,
        entry_time=signal.signal_time,
        exit_time=exit_time,
        entry_price=entry_price,
        exit_price=exit_price,
        lot_size=lot_size,
        pnl=total_pnl,
        pnl_pct=(total_pnl / balance) * 100,
        exit_reason=exit_reason
    )
    
    bars_held = min(max_bars, len(candles) - start_idx - 1)
    return trade, bars_held, total_pnl


def run_backtest(
    params: StrategyParams,
    config: OptimizerConfig,
    symbols: List[str] = None
) -> BacktestResult:
    """
    Run full backtest with given parameters.
    
    PROP FIRM COMPLIANT BACKTEST:
    - Compounding: lot sizes calculated from current balance
    - Daily DD tracking: reduces risk at 3.2%, halts at 4%, fails at 5%
    - Total DD tracking: STATIC from initial balance (not trailing)
    - Risk reduction when approaching limits
    """
    if symbols is None:
        symbols = SYMBOLS
    
    balance = config.initial_balance
    peak_balance = balance
    daily_start_balance = balance
    
    trades: List[Trade] = []
    daily_pnl: Dict[str, float] = {}
    
    max_daily_dd_pct = 0.0
    max_total_dd_pct = 0.0
    
    # Risk state tracking
    current_risk_multiplier = 1.0
    daily_halted = False
    
    current_date = None
    
    # Load all data
    all_data = {}
    for symbol in symbols:
        df = load_ohlc_data(symbol, "D1", config.data_dir)
        if not df.empty:
            candles = prepare_candles(df, config.start_date, config.end_date)
            if candles:
                all_data[symbol] = candles
    
    if not all_data:
        print("⚠️  No data found! Returning empty result.")
        return BacktestResult(
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_pnl=0, final_balance=balance,
            max_daily_dd_pct=0, max_total_dd_pct=0,
            sharpe_ratio=0, profit_factor=0,
            trades=[], daily_pnl={}
        )
    
    # Find common date range
    all_dates = set()
    for symbol, candles in all_data.items():
        for c in candles:
            all_dates.add(c['time'].date())
    
    sorted_dates = sorted(all_dates)
    
    # Iterate through dates
    for date in sorted_dates:
        date_str = str(date)
        
        # New day tracking
        if current_date != date:
            if current_date is not None:
                # Record previous day's PnL
                day_pnl = balance - daily_start_balance
                daily_pnl[str(current_date)] = day_pnl
                
                # Check daily drawdown
                if daily_start_balance > 0:
                    daily_dd = max(0, (daily_start_balance - balance) / daily_start_balance * 100)
                    max_daily_dd_pct = max(max_daily_dd_pct, daily_dd)
            
            current_date = date
            daily_start_balance = balance
            
            # Reset daily risk state for new day
            daily_halted = False
            current_risk_multiplier = 1.0
        
        # Update peak and total drawdown
        if balance > peak_balance:
            peak_balance = balance
        
        total_dd = max(0, (config.initial_balance - balance) / config.initial_balance * 100)
        max_total_dd_pct = max(max_total_dd_pct, total_dd)
        
        # ═══════════════════════════════════════════════════════════════
        # 5ERS RISK MANAGEMENT - CHECK BEFORE EVERY TRADE
        # ═══════════════════════════════════════════════════════════════
        
        # Calculate current daily drawdown
        current_daily_dd = 0.0
        if daily_start_balance > 0:
            current_daily_dd = max(0, (daily_start_balance - balance) / daily_start_balance * 100)
        
        # Calculate current total drawdown (STATIC from initial balance)
        current_total_dd = max(0, (config.initial_balance - balance) / config.initial_balance * 100)
        
        # HARD STOP: Daily DD limit hit - fail challenge
        if current_daily_dd >= config.max_daily_dd_pct:
            max_daily_dd_pct = current_daily_dd
            break  # Challenge failed - stop trading
        
        # HARD STOP: Total DD limit hit - fail challenge  
        if current_total_dd >= config.max_total_dd_pct:
            max_total_dd_pct = current_total_dd
            break  # Challenge failed - stop trading
        
        # Daily halt threshold (4%) - no new trades today
        if current_daily_dd >= config.daily_dd_halt_pct:
            daily_halted = True
        
        # Skip trading if halted for the day
        if daily_halted:
            continue
        
        # Determine risk multiplier based on DD levels
        if current_daily_dd >= config.daily_dd_reduce_pct or current_total_dd >= config.total_dd_reduce_pct:
            # At 3.2% daily or 7% total: TRADE HALF
            current_risk_multiplier = config.reduced_risk_multiplier  # 0.5
        elif current_daily_dd >= config.daily_dd_warning_pct or current_total_dd >= config.total_dd_warning_pct:
            # At 2% daily or 5% total: Slightly reduced
            current_risk_multiplier = 0.75
        else:
            current_risk_multiplier = 1.0
        
        # Update max values
        max_daily_dd_pct = max(max_daily_dd_pct, current_daily_dd)
        max_total_dd_pct = max(max_total_dd_pct, current_total_dd)
        
        # Generate signals for each symbol
        for symbol, candles in all_data.items():
            # Find candles up to this date
            idx = None
            for i, c in enumerate(candles):
                if c['time'].date() == date:
                    idx = i
                    break
            
            if idx is None or idx < 50:
                continue
            
            # Need at least 50 bars of history for indicators
            history = candles[:idx+1]
            
            # Get contract specs
            spec = CONTRACT_SPECS.get(symbol.replace("_", ""), DEFAULT_SPEC)
            
            # Generate signal
            signal = generate_signal(
                symbol=symbol,
                candles=history,
                params=params,
                signal_time=candles[idx]['time']
            )
            
            if signal:
                # Apply risk multiplier to position size
                adjusted_params = StrategyParams(
                    **{**params.to_dict(), 
                       'risk_per_trade_pct': params.risk_per_trade_pct * current_risk_multiplier}
                )
                
                # Simulate trade with adjusted risk
                trade, bars_held, pnl = simulate_trade(
                    signal=signal,
                    candles=candles,
                    start_idx=idx,
                    balance=balance,
                    params=adjusted_params,
                    spec=spec
                )
                
                if trade:
                    trades.append(trade)
                    balance += pnl
                    
                    # Re-check DD after trade closes
                    if daily_start_balance > 0:
                        post_trade_dd = max(0, (daily_start_balance - balance) / daily_start_balance * 100)
                        if post_trade_dd >= config.daily_dd_halt_pct:
                            daily_halted = True
    
    # Calculate statistics
    winning_trades = len([t for t in trades if t.pnl > 0])
    losing_trades = len([t for t in trades if t.pnl <= 0])
    total_trades = len(trades)
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - config.initial_balance
    
    # Profit factor
    gross_profit = sum(t.pnl for t in trades if t.pnl > 0)
    gross_loss = abs(sum(t.pnl for t in trades if t.pnl < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
    
    # Sharpe ratio (simplified)
    if trades:
        returns = [t.pnl_pct for t in trades]
        mean_return = np.mean(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    else:
        sharpe_ratio = 0
    
    return BacktestResult(
        total_trades=total_trades,
        winning_trades=winning_trades,
        losing_trades=losing_trades,
        win_rate=win_rate,
        total_pnl=total_pnl,
        final_balance=balance,
        max_daily_dd_pct=max_daily_dd_pct,
        max_total_dd_pct=max_total_dd_pct,
        sharpe_ratio=sharpe_ratio,
        profit_factor=profit_factor,
        trades=trades,
        daily_pnl=daily_pnl
    )


# ═══════════════════════════════════════════════════════════════════════════
# OPTUNA OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════

def create_objective(config: OptimizerConfig):
    """Create Optuna objective function."""
    
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function to maximize.
        
        Returns final balance if 5ers rules are met, otherwise -1.
        """
        # Sample parameters for CONFLUENCE STRATEGY
        params = StrategyParams(
            # Risk
            risk_per_trade_pct=trial.suggest_float("risk_per_trade_pct", 0.3, 0.8),
            
            # SMA for trend detection
            sma_fast_period=trial.suggest_int("sma_fast_period", 10, 30),
            sma_slow_period=trial.suggest_int("sma_slow_period", 40, 80),
            sma_trend_period=trial.suggest_int("sma_trend_period", 150, 250),
            
            # MACD
            macd_fast=trial.suggest_int("macd_fast", 8, 15),
            macd_slow=trial.suggest_int("macd_slow", 20, 30),
            macd_signal=trial.suggest_int("macd_signal", 7, 12),
            
            # Stochastic
            stoch_k_period=trial.suggest_int("stoch_k_period", 10, 20),
            stoch_d_period=trial.suggest_int("stoch_d_period", 3, 5),
            stoch_overbought=trial.suggest_float("stoch_overbought", 75, 85),
            stoch_oversold=trial.suggest_float("stoch_oversold", 15, 25),
            
            # Bollinger Bands
            bb_period=trial.suggest_int("bb_period", 15, 25),
            bb_std_dev=trial.suggest_float("bb_std_dev", 1.5, 2.5),
            
            # ATR
            atr_period=trial.suggest_int("atr_period", 10, 20),
            atr_sl_multiplier=trial.suggest_float("atr_sl_multiplier", 1.2, 2.5),
            
            # Structure
            swing_lookback=trial.suggest_int("swing_lookback", 3, 8),
            structure_break_atr=trial.suggest_float("structure_break_atr", 0.2, 0.5),
            
            # TPs (ensure TP1 < TP2 < TP3)
            tp1_r_multiple=trial.suggest_float("tp1_r_multiple", 1.0, 2.0),
            tp2_r_multiple=trial.suggest_float("tp2_r_multiple", 2.0, 4.0),
            tp3_r_multiple=trial.suggest_float("tp3_r_multiple", 4.0, 6.0),
            
            # Partial closes
            tp1_close_pct=trial.suggest_float("tp1_close_pct", 0.30, 0.45),
            tp2_close_pct=trial.suggest_float("tp2_close_pct", 0.25, 0.40),
            tp3_close_pct=trial.suggest_float("tp3_close_pct", 0.20, 0.35),
            
            # Confluence
            min_confluence=trial.suggest_int("min_confluence", 2, 4),
            
            # Regime filter
            use_regime_filter=trial.suggest_categorical("use_regime_filter", [True, False]),
            range_atr_threshold=trial.suggest_float("range_atr_threshold", 0.5, 0.8),
        )
        
        # Run backtest
        result = run_backtest(params, config)
        
        # Check 5ers compliance
        if not result.is_valid(config):
            return -1_000_000  # Invalid - violated rules
        
        if result.total_trades < 10:
            return -500_000  # Too few trades
        
        # Objective: maximize profit while penalizing drawdown
        score = result.total_pnl
        
        # Penalty for approaching limits
        dd_penalty = (result.max_daily_dd_pct / config.max_daily_dd_pct) * 0.1
        score *= (1 - dd_penalty)
        
        return score
    
    return objective


def run_optimization(config: OptimizerConfig) -> Dict:
    """Run TPE optimization."""
    print("\n" + "=" * 60)
    print("🔧 TPE OPTIMIZER - Finding Best Parameters")
    print("=" * 60)
    print(f"   Initial Balance: ${config.initial_balance:,.0f}")
    print(f"   Period: {config.start_date} to {config.end_date}")
    print(f"   Max Daily DD: {config.max_daily_dd_pct}%")
    print(f"   Max Total DD: {config.max_total_dd_pct}%")
    print(f"   Trials: {config.n_trials}")
    print("=" * 60)
    
    # Create output directory
    config.output_dir.mkdir(exist_ok=True)
    
    # Create study
    study = optuna.create_study(
        study_name=config.study_name,
        direction="maximize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
        storage=f"sqlite:///{config.output_dir}/optuna.db",
        load_if_exists=True
    )
    
    # Run optimization
    objective = create_objective(config)
    
    study.optimize(
        objective,
        n_trials=config.n_trials,
        show_progress_bar=True,
        n_jobs=1  # Single thread for reproducibility
    )
    
    # Get best result
    best_trial = study.best_trial
    best_params = best_trial.params
    best_score = best_trial.value
    
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"   Best Score: ${best_score:,.2f}")
    print(f"   Total Trials: {len(study.trials)}")
    print("\n📊 Best Parameters:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")
    
    # Run final backtest with best params
    final_params = StrategyParams(**{k: v for k, v in best_params.items() if hasattr(StrategyParams, k)})
    final_result = run_backtest(final_params, config)
    
    print("\n📈 Final Backtest Results:")
    print(f"   Total Trades: {final_result.total_trades}")
    print(f"   Win Rate: {final_result.win_rate:.1f}%")
    print(f"   Total PnL: ${final_result.total_pnl:,.2f}")
    print(f"   Final Balance: ${final_result.final_balance:,.2f}")
    print(f"   Max Daily DD: {final_result.max_daily_dd_pct:.2f}%")
    print(f"   Max Total DD: {final_result.max_total_dd_pct:.2f}%")
    print(f"   Profit Factor: {final_result.profit_factor:.2f}")
    print(f"   Sharpe Ratio: {final_result.sharpe_ratio:.2f}")
    
    # Check 5ers compliance
    if final_result.is_valid(config):
        print("\n✅ 5ERS COMPLIANCE: PASSED")
    else:
        print("\n❌ 5ERS COMPLIANCE: FAILED")
        if final_result.max_daily_dd_pct > config.max_daily_dd_pct:
            print(f"   Daily DD {final_result.max_daily_dd_pct:.2f}% > {config.max_daily_dd_pct}% limit")
        if final_result.max_total_dd_pct > config.max_total_dd_pct:
            print(f"   Total DD {final_result.max_total_dd_pct:.2f}% > {config.max_total_dd_pct}% limit")
    
    # Save results
    output = {
        "optimization_time": datetime.now().isoformat(),
        "config": {
            "initial_balance": config.initial_balance,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "n_trials": config.n_trials,
        },
        "best_params": best_params,
        "best_score": best_score,
        "results": {
            "total_trades": final_result.total_trades,
            "win_rate": final_result.win_rate,
            "total_pnl": final_result.total_pnl,
            "final_balance": final_result.final_balance,
            "max_daily_dd_pct": final_result.max_daily_dd_pct,
            "max_total_dd_pct": final_result.max_total_dd_pct,
            "profit_factor": final_result.profit_factor,
            "sharpe_ratio": final_result.sharpe_ratio,
            "fiveers_compliant": final_result.is_valid(config),
        }
    }
    
    output_file = config.output_dir / "best_params.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Save trades CSV
    if final_result.trades:
        trades_df = pd.DataFrame([
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "lot_size": t.lot_size,
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
            }
            for t in final_result.trades
        ])
        trades_file = config.output_dir / "trades.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"💾 Trades saved to: {trades_file}")
    
    return output


def show_status(config: OptimizerConfig):
    """Show optimization status."""
    db_path = config.output_dir / "optuna.db"
    
    if not db_path.exists():
        print("❌ No optimization in progress.")
        print(f"   Run: python optimizer.py --trials 100")
        return
    
    try:
        study = optuna.load_study(
            study_name=config.study_name,
            storage=f"sqlite:///{db_path}"
        )
        
        print("\n" + "=" * 60)
        print("📊 OPTIMIZATION STATUS")
        print("=" * 60)
        print(f"   Study Name: {config.study_name}")
        print(f"   Completed Trials: {len(study.trials)}")
        
        if study.best_trial:
            print(f"\n   Best Score: ${study.best_value:,.2f}")
            print(f"   Best Trial: #{study.best_trial.number}")
            print("\n   Best Parameters:")
            for key, value in study.best_params.items():
                print(f"      {key}: {value}")
        
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error loading study: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TPE Optimizer for Trading Bot")
    parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
    parser.add_argument("--balance", type=float, default=100_000, help="Initial balance")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument("--status", action="store_true", help="Show optimization status")
    parser.add_argument("--data-dir", type=str, default="data/ohlcv", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="optimization_output", help="Output directory")
    
    args = parser.parse_args()
    
    config = OptimizerConfig(
        initial_balance=args.balance,
        n_trials=args.trials,
        start_date=args.start,
        end_date=args.end,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
    
    if args.status:
        show_status(config)
    else:
        run_optimization(config)


if __name__ == "__main__":
    main()
