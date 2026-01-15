#!/usr/bin/env python3
"""
📊 H1 TRADE SIMULATOR - Realistic Performance Validation
==========================================================

Simulates trades on hourly (H1) data for realistic validation.
This is the final step after TPE optimization to validate results
with more granular price action.

FEATURES:
1. Entry queue system (signals wait for price proximity)
2. Limit orders fill when H1 price touches entry level
3. Lot sizing at FILL moment (enables compounding)
4. 3-TP partial close system with trailing stops
5. 5ers DDD/TDD safety tracking

USAGE:
    python simulator.py --params optimization_output/best_params.json
    python simulator.py --balance 100000 --start 2023-01-01 --end 2025-12-31

Author: Trading Bot Template
Version: 1.0.0
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """Simulation configuration matching live bot."""
    # Account
    initial_balance: float = 100_000
    
    # Risk
    risk_per_trade_pct: float = 0.6
    
    # Entry queue settings
    limit_order_proximity_r: float = 0.3    # Place limit when within 0.3R
    max_entry_distance_r: float = 1.5       # Remove if beyond 1.5R
    max_entry_wait_hours: int = 120         # 5 days max wait
    immediate_entry_r: float = 0.05         # Market order if within 0.05R
    
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
    reduced_risk_pct: float = 0.4           # Reduced risk % per trade
    
    # Scaling (confluence based)
    confluence_base_score: int = 2
    confluence_scale_per_point: float = 0.15
    max_confluence_multiplier: float = 1.5
    min_confluence_multiplier: float = 0.6


# Contract specifications
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


# ═══════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Trading signal from daily scan."""
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confluence: int
    signal_time: datetime
    
    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)


@dataclass
class PendingOrder:
    """Limit order waiting to fill."""
    signal: Signal
    created_at: datetime


@dataclass
class Position:
    """Open position."""
    signal: Signal
    fill_time: datetime
    fill_price: float
    lot_size: float
    remaining_pct: float = 1.0
    total_pnl: float = 0.0
    current_sl: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False


@dataclass
class ClosedTrade:
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
    confluence: int


@dataclass
class DailySnapshot:
    """Daily account snapshot."""
    date: str
    balance: float
    equity: float
    daily_pnl: float
    daily_dd_pct: float
    total_dd_pct: float
    open_positions: int
    trades_today: int


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY IMPORT
# ═══════════════════════════════════════════════════════════════════════════

try:
    from strategy_template import (
        StrategyParams,
        generate_signal as strategy_generate_signal,
        calculate_lot_size,
        get_default_params,
    )
except ImportError:
    print("❌ Could not import strategy_template.py")
    print("   Make sure strategy_template.py exists in the same directory.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_h1_data(symbol: str, data_dir: Path = Path("data/ohlcv")) -> pd.DataFrame:
    """Load H1 data for a symbol."""
    patterns = [
        f"{symbol}_H1*.csv",
        f"{symbol.replace('USD', '_USD')}_H1*.csv",
        f"*{symbol}*H1*.csv",
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            df = pd.read_csv(files[0])
            df.columns = df.columns.str.lower()
            
            date_col = 'time' if 'time' in df.columns else 'date'
            if date_col in df.columns:
                df['time'] = pd.to_datetime(df[date_col])
                df = df.sort_values('time')
            
            return df
    
    return pd.DataFrame()


def load_d1_data(symbol: str, data_dir: Path = Path("data/ohlcv")) -> pd.DataFrame:
    """Load D1 data for a symbol."""
    patterns = [
        f"{symbol}_D1*.csv",
        f"{symbol.replace('USD', '_USD')}_D1*.csv",
        f"*{symbol}*D1*.csv",
    ]
    
    for pattern in patterns:
        files = list(data_dir.glob(pattern))
        if files:
            df = pd.read_csv(files[0])
            df.columns = df.columns.str.lower()
            
            date_col = 'time' if 'time' in df.columns else 'date'
            if date_col in df.columns:
                df['time'] = pd.to_datetime(df[date_col])
                df = df.sort_values('time')
            
            return df
    
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

class TradingSimulator:
    """Simulates trading with entry queue and H1 execution."""
    
    def __init__(self, config: SimConfig, params: StrategyParams):
        self.config = config
        self.params = params
        
        self.balance = config.initial_balance
        self.peak_balance = config.initial_balance
        self.daily_start_balance = config.initial_balance
        
        self.pending_orders: List[PendingOrder] = []
        self.positions: List[Position] = []
        self.closed_trades: List[ClosedTrade] = []
        self.daily_snapshots: List[DailySnapshot] = []
        
        self.max_daily_dd_pct = 0.0
        self.max_total_dd_pct = 0.0
        self.current_date = None
        self.ddd_halt = False
        self.tdd_halt = False
        
        self.daily_pnl = 0.0
        self.trades_today = 0
    
    def get_spec(self, symbol: str) -> Dict:
        """Get contract specifications."""
        return CONTRACT_SPECS.get(symbol.replace("_", ""), DEFAULT_SPEC)
    
    def calculate_lot_size(self, signal: Signal) -> float:
        """Calculate lot size with confluence scaling."""
        spec = self.get_spec(signal.symbol)
        
        # Base risk percentage
        risk_pct = self.config.risk_per_trade_pct
        
        # Confluence scaling
        if signal.confluence > self.config.confluence_base_score:
            bonus = (signal.confluence - self.config.confluence_base_score) * self.config.confluence_scale_per_point
            multiplier = min(1 + bonus, self.config.max_confluence_multiplier)
        else:
            reduction = (self.config.confluence_base_score - signal.confluence) * self.config.confluence_scale_per_point
            multiplier = max(1 - reduction, self.config.min_confluence_multiplier)
        
        risk_pct *= multiplier
        
        # Calculate lot size
        risk_amount = self.balance * (risk_pct / 100)
        sl_pips = abs(signal.entry - signal.stop_loss) / spec['pip_size']
        
        if sl_pips <= 0:
            return 0.01
        
        lot_size = risk_amount / (sl_pips * spec['pip_value'])
        return max(0.01, round(lot_size, 2))
    
    def new_day(self, date: datetime.date):
        """Handle new trading day."""
        if self.current_date is not None:
            # Record previous day's snapshot
            self.record_daily_snapshot()
        
        self.current_date = date
        self.daily_start_balance = self.balance
        self.daily_pnl = 0.0
        self.trades_today = 0
        self.ddd_halt = False
    
    def record_daily_snapshot(self):
        """Record end-of-day snapshot."""
        equity = self.balance + sum(p.total_pnl for p in self.positions)
        
        daily_dd = 0.0
        if self.daily_start_balance > 0:
            daily_dd = max(0, (self.daily_start_balance - self.balance) / self.daily_start_balance * 100)
        
        total_dd = max(0, (self.config.initial_balance - self.balance) / self.config.initial_balance * 100)
        
        snapshot = DailySnapshot(
            date=str(self.current_date),
            balance=self.balance,
            equity=equity,
            daily_pnl=self.daily_pnl,
            daily_dd_pct=daily_dd,
            total_dd_pct=total_dd,
            open_positions=len(self.positions),
            trades_today=self.trades_today
        )
        
        self.daily_snapshots.append(snapshot)
        self.max_daily_dd_pct = max(self.max_daily_dd_pct, daily_dd)
        self.max_total_dd_pct = max(self.max_total_dd_pct, total_dd)
    
    def check_ddd_safety(self):
        """Check daily drawdown safety."""
        if self.daily_start_balance <= 0:
            return
        
        daily_dd = (self.daily_start_balance - self.balance) / self.daily_start_balance * 100
        
        if daily_dd >= self.config.daily_loss_halt_pct:
            self.ddd_halt = True
    
    def check_tdd_safety(self):
        """Check total drawdown safety."""
        total_dd = (self.config.initial_balance - self.balance) / self.config.initial_balance * 100
        
        if total_dd >= self.config.max_total_dd_pct:
            self.tdd_halt = True
    
    def add_signal_to_queue(self, signal: Signal, current_price: float):
        """Add signal to entry queue or execute immediately."""
        if self.ddd_halt or self.tdd_halt:
            return
        
        # Calculate distance from entry in R
        distance_r = abs(current_price - signal.entry) / signal.risk if signal.risk > 0 else 999
        
        if distance_r <= self.config.immediate_entry_r:
            # Execute immediately as market order
            lot_size = self.calculate_lot_size(signal)
            self.open_position(signal, signal.signal_time, current_price, lot_size)
        
        elif distance_r <= self.config.limit_order_proximity_r:
            # Add to pending orders
            self.pending_orders.append(PendingOrder(signal=signal, created_at=signal.signal_time))
    
    def process_pending_orders(self, current_time: datetime, h1_candle: Dict):
        """Check if any pending orders should fill."""
        if self.ddd_halt or self.tdd_halt:
            return
        
        to_remove = []
        
        for order in self.pending_orders:
            signal = order.signal
            
            # Check expiry
            hours_waiting = (current_time - order.created_at).total_seconds() / 3600
            if hours_waiting > self.config.max_entry_wait_hours:
                to_remove.append(order)
                continue
            
            # Check if price touched entry
            if signal.direction == 'bullish':
                if h1_candle['low'] <= signal.entry <= h1_candle['high']:
                    lot_size = self.calculate_lot_size(signal)
                    self.open_position(signal, current_time, signal.entry, lot_size)
                    to_remove.append(order)
            else:
                if h1_candle['low'] <= signal.entry <= h1_candle['high']:
                    lot_size = self.calculate_lot_size(signal)
                    self.open_position(signal, current_time, signal.entry, lot_size)
                    to_remove.append(order)
        
        for order in to_remove:
            if order in self.pending_orders:
                self.pending_orders.remove(order)
    
    def open_position(self, signal: Signal, fill_time: datetime, fill_price: float, lot_size: float):
        """Open a new position."""
        position = Position(
            signal=signal,
            fill_time=fill_time,
            fill_price=fill_price,
            lot_size=lot_size,
            remaining_pct=1.0,
            total_pnl=0.0,
            current_sl=signal.stop_loss,
            tp1_hit=False,
            tp2_hit=False
        )
        
        self.positions.append(position)
    
    def process_positions(self, current_time: datetime, h1_candle: Dict, symbol: str):
        """Process open positions against H1 candle."""
        to_close = []
        
        for position in self.positions:
            if position.signal.symbol.replace("_", "") != symbol.replace("_", ""):
                continue
            
            signal = position.signal
            spec = self.get_spec(signal.symbol)
            high = h1_candle['high']
            low = h1_candle['low']
            
            exit_reason = None
            exit_price = None
            
            # Check stop loss
            if signal.direction == 'bullish' and low <= position.current_sl:
                exit_reason = "stop_loss"
                exit_price = position.current_sl
            elif signal.direction == 'bearish' and high >= position.current_sl:
                exit_reason = "stop_loss"
                exit_price = position.current_sl
            
            if exit_reason:
                # Close entire remaining position
                if signal.direction == 'bullish':
                    pips = (exit_price - position.fill_price) / spec['pip_size']
                else:
                    pips = (position.fill_price - exit_price) / spec['pip_size']
                
                pnl = pips * spec['pip_value'] * position.lot_size * position.remaining_pct
                pnl -= spec['commission'] * position.lot_size * position.remaining_pct
                
                position.total_pnl += pnl
                position.remaining_pct = 0
                to_close.append((position, current_time, exit_price, exit_reason))
                continue
            
            # Check TP1
            if not position.tp1_hit:
                if (signal.direction == 'bullish' and high >= signal.tp1) or \
                   (signal.direction == 'bearish' and low <= signal.tp1):
                    position.tp1_hit = True
                    
                    close_pct = self.params.tp1_close_pct
                    if signal.direction == 'bullish':
                        pips = (signal.tp1 - position.fill_price) / spec['pip_size']
                    else:
                        pips = (position.fill_price - signal.tp1) / spec['pip_size']
                    
                    pnl = pips * spec['pip_value'] * position.lot_size * close_pct
                    pnl -= spec['commission'] * position.lot_size * close_pct
                    position.total_pnl += pnl
                    position.remaining_pct -= close_pct
                    
                    # Move SL to breakeven
                    position.current_sl = position.fill_price
            
            # Check TP2
            if position.tp1_hit and not position.tp2_hit:
                if (signal.direction == 'bullish' and high >= signal.tp2) or \
                   (signal.direction == 'bearish' and low <= signal.tp2):
                    position.tp2_hit = True
                    
                    close_pct = self.params.tp2_close_pct
                    if signal.direction == 'bullish':
                        pips = (signal.tp2 - position.fill_price) / spec['pip_size']
                    else:
                        pips = (position.fill_price - signal.tp2) / spec['pip_size']
                    
                    pnl = pips * spec['pip_value'] * position.lot_size * close_pct
                    pnl -= spec['commission'] * position.lot_size * close_pct
                    position.total_pnl += pnl
                    position.remaining_pct -= close_pct
                    
                    # Trail SL to TP1
                    position.current_sl = signal.tp1
            
            # Check TP3
            if position.tp1_hit and position.tp2_hit and position.remaining_pct > 0:
                if (signal.direction == 'bullish' and high >= signal.tp3) or \
                   (signal.direction == 'bearish' and low <= signal.tp3):
                    
                    if signal.direction == 'bullish':
                        pips = (signal.tp3 - position.fill_price) / spec['pip_size']
                    else:
                        pips = (position.fill_price - signal.tp3) / spec['pip_size']
                    
                    pnl = pips * spec['pip_value'] * position.lot_size * position.remaining_pct
                    pnl -= spec['commission'] * position.lot_size * position.remaining_pct
                    position.total_pnl += pnl
                    position.remaining_pct = 0
                    
                    to_close.append((position, current_time, signal.tp3, "tp3"))
        
        # Close completed positions
        for position, exit_time, exit_price, exit_reason in to_close:
            self.close_position(position, exit_time, exit_price, exit_reason)
    
    def close_position(self, position: Position, exit_time: datetime, exit_price: float, exit_reason: str):
        """Close a position and record trade."""
        trade = ClosedTrade(
            symbol=position.signal.symbol,
            direction=position.signal.direction,
            entry_time=position.fill_time,
            exit_time=exit_time,
            entry_price=position.fill_price,
            exit_price=exit_price,
            lot_size=position.lot_size,
            pnl=position.total_pnl,
            pnl_pct=(position.total_pnl / self.daily_start_balance) * 100 if self.daily_start_balance > 0 else 0,
            exit_reason=exit_reason,
            confluence=position.signal.confluence
        )
        
        self.closed_trades.append(trade)
        self.balance += position.total_pnl
        self.daily_pnl += position.total_pnl
        self.trades_today += 1
        
        if position in self.positions:
            self.positions.remove(position)
        
        # Update peak balance
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        # Check safety
        self.check_ddd_safety()
        self.check_tdd_safety()
    
    def get_results(self) -> Dict:
        """Get simulation results."""
        # Record final day
        if self.current_date is not None:
            self.record_daily_snapshot()
        
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.pnl > 0])
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        total_pnl = self.balance - self.config.initial_balance
        
        # Profit factor
        gross_profit = sum(t.pnl for t in self.closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self.closed_trades if t.pnl < 0))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        return {
            "initial_balance": self.config.initial_balance,
            "final_balance": self.balance,
            "total_pnl": total_pnl,
            "return_pct": (total_pnl / self.config.initial_balance) * 100,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "max_daily_dd_pct": self.max_daily_dd_pct,
            "max_total_dd_pct": self.max_total_dd_pct,
            "fiveers_compliant": (
                self.max_daily_dd_pct <= 5.0 and
                self.max_total_dd_pct <= 10.0
            )
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION
# ═══════════════════════════════════════════════════════════════════════════

def run_simulation(
    params: StrategyParams,
    config: SimConfig,
    symbols: List[str],
    start_date: str,
    end_date: str,
    data_dir: Path = Path("data/ohlcv")
) -> Dict:
    """Run full simulation with H1 data."""
    
    print("\n" + "=" * 60)
    print("📊 H1 TRADE SIMULATOR")
    print("=" * 60)
    print(f"   Initial Balance: ${config.initial_balance:,.0f}")
    print(f"   Period: {start_date} to {end_date}")
    print(f"   Symbols: {len(symbols)}")
    print("=" * 60)
    
    simulator = TradingSimulator(config, params)
    
    # Load all data
    print("\n📥 Loading data...")
    d1_data = {}
    h1_data = {}
    
    for symbol in tqdm(symbols, desc="Loading"):
        d1 = load_d1_data(symbol, data_dir)
        h1 = load_h1_data(symbol, data_dir)
        
        if not d1.empty:
            d1_data[symbol] = d1
        if not h1.empty:
            h1_data[symbol] = h1
    
    print(f"   D1 data loaded: {len(d1_data)} symbols")
    print(f"   H1 data loaded: {len(h1_data)} symbols")
    
    if not h1_data:
        print("⚠️  No H1 data found! Cannot simulate.")
        return simulator.get_results()
    
    # Get all H1 timestamps
    all_times = set()
    for symbol, df in h1_data.items():
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        mask = (df['time'] >= start) & (df['time'] <= end)
        for t in df[mask]['time']:
            all_times.add(t)
    
    sorted_times = sorted(all_times)
    print(f"   H1 bars to process: {len(sorted_times)}")
    
    # Process each H1 bar
    print("\n⏳ Running simulation...")
    
    for current_time in tqdm(sorted_times, desc="Simulating"):
        current_date = current_time.date()
        
        # New day handling
        if simulator.current_date != current_date:
            simulator.new_day(current_date)
        
        # Skip if halted
        if simulator.ddd_halt or simulator.tdd_halt:
            continue
        
        # Daily signal generation (at 00:00)
        if current_time.hour == 0:
            for symbol in d1_data.keys():
                if symbol not in d1_data:
                    continue
                
                d1_df = d1_data[symbol]
                
                # Get D1 candles up to yesterday
                mask = d1_df['time'].dt.date < current_date
                history_df = d1_df[mask]
                
                if len(history_df) < 50:
                    continue
                
                candles = [
                    {
                        'time': row['time'],
                        'open': row['open'],
                        'high': row['high'],
                        'low': row['low'],
                        'close': row['close']
                    }
                    for _, row in history_df.iterrows()
                ]
                
                # Generate signal
                signal = strategy_generate_signal(
                    symbol=symbol,
                    candles=candles,
                    params=params,
                    signal_time=current_time
                )
                
                if signal:
                    current_price = candles[-1]['close']
                    simulator.add_signal_to_queue(signal, current_price)
        
        # Process each symbol's H1 bar
        for symbol, h1_df in h1_data.items():
            mask = h1_df['time'] == current_time
            if not mask.any():
                continue
            
            row = h1_df[mask].iloc[0]
            h1_candle = {
                'time': row['time'],
                'open': row['open'],
                'high': row['high'],
                'low': row['low'],
                'close': row['close']
            }
            
            # Process pending orders
            simulator.process_pending_orders(current_time, h1_candle)
            
            # Process open positions
            simulator.process_positions(current_time, h1_candle, symbol)
    
    # Get results
    results = simulator.get_results()
    
    print("\n" + "=" * 60)
    print("📈 SIMULATION RESULTS")
    print("=" * 60)
    print(f"   Final Balance: ${results['final_balance']:,.2f}")
    print(f"   Total PnL: ${results['total_pnl']:,.2f}")
    print(f"   Return: {results['return_pct']:.1f}%")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Win Rate: {results['win_rate']:.1f}%")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Max Daily DD: {results['max_daily_dd_pct']:.2f}%")
    print(f"   Max Total DD: {results['max_total_dd_pct']:.2f}%")
    
    if results['fiveers_compliant']:
        print("\n✅ 5ERS COMPLIANCE: PASSED")
    else:
        print("\n❌ 5ERS COMPLIANCE: FAILED")
    
    print("=" * 60)
    
    return {
        "results": results,
        "trades": [
            {
                "symbol": t.symbol,
                "direction": t.direction,
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "lot_size": t.lot_size,
                "pnl": t.pnl,
                "exit_reason": t.exit_reason
            }
            for t in simulator.closed_trades
        ],
        "daily_snapshots": [
            {
                "date": s.date,
                "balance": s.balance,
                "daily_pnl": s.daily_pnl,
                "daily_dd_pct": s.daily_dd_pct,
                "total_dd_pct": s.total_dd_pct
            }
            for s in simulator.daily_snapshots
        ]
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="H1 Trade Simulator")
    parser.add_argument("--params", type=str, help="Path to params JSON file")
    parser.add_argument("--balance", type=float, default=100_000, help="Initial balance")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date")
    parser.add_argument("--data-dir", type=str, default="data/ohlcv", help="Data directory")
    parser.add_argument("--output", type=str, default="simulation_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Load parameters
    if args.params and Path(args.params).exists():
        with open(args.params) as f:
            data = json.load(f)
            params_dict = data.get('best_params', data)
            params = StrategyParams(**{k: v for k, v in params_dict.items() if hasattr(StrategyParams, k)})
    else:
        params = get_default_params()
    
    config = SimConfig(initial_balance=args.balance)
    
    symbols = [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
        "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
    ]
    
    results = run_simulation(
        params=params,
        config=config,
        symbols=symbols,
        start_date=args.start,
        end_date=args.end,
        data_dir=Path(args.data_dir)
    )
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(results['results'], f, indent=2)
    
    if results['trades']:
        pd.DataFrame(results['trades']).to_csv(output_dir / "trades.csv", index=False)
    
    if results['daily_snapshots']:
        pd.DataFrame(results['daily_snapshots']).to_csv(output_dir / "daily_snapshots.csv", index=False)
    
    print(f"\n💾 Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
