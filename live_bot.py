#!/usr/bin/env python3
"""
🤖 LIVE TRADING BOT - Template Version
========================================

Simple live trading bot for MetaTrader 5.
Uses the template strategy for signal generation.

REQUIREMENTS:
- Windows OS with MetaTrader 5
- MT5 account credentials in .env file
- Python 3.10+

USAGE:
    python live_bot.py

Author: Trading Bot Template
Version: 1.0.0

NOTE: This is a simplified template. For production use, add:
- Better error handling
- Position management (partial closes, trailing stops)
- More robust MT5 connection handling
- Proper logging and monitoring
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List

# Check if MT5 is available
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

import numpy as np
import pandas as pd

from strategy_template import (
    StrategyParams,
    Signal,
    generate_signal,
    get_default_params,
)


# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class BotConfig:
    """Live bot configuration."""
    # MT5 credentials
    mt5_login: int = 0
    mt5_password: str = ""
    mt5_server: str = ""
    
    # Trading settings
    symbols: List[str] = field(default_factory=lambda: [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "NZDUSD",
        "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "EURGBP",
    ])
    max_positions: int = 10
    
    # Timing
    scan_hour: int = 0       # Daily scan at midnight
    scan_minute: int = 10    # 10 minutes after daily close
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 5ERS / PROP FIRM RISK RULES (CRITICAL - DO NOT MODIFY)
    # ═══════════════════════════════════════════════════════════════════════════
    # These rules MUST be followed to pass prop firm challenges
    
    # Account (for DD calculation)
    initial_balance: float = 100_000        # Starting balance for DD calc
    
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


# ═══════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging() -> logging.Logger:
    """Setup logging."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"bot_{datetime.now():%Y%m%d}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
# MT5 FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def init_mt5(config: BotConfig) -> bool:
    """Initialize MT5 connection."""
    if not MT5_AVAILABLE:
        logger.error("MT5 not available (Windows only)")
        return False
    
    if not mt5.initialize():
        logger.error(f"MT5 init failed: {mt5.last_error()}")
        return False
    
    if config.mt5_login:
        if not mt5.login(config.mt5_login, config.mt5_password, config.mt5_server):
            logger.error(f"MT5 login failed: {mt5.last_error()}")
            return False
    
    info = mt5.account_info()
    logger.info(f"MT5 connected: Balance=${info.balance:,.2f}")
    return True


def get_candles(symbol: str, bars: int = 100) -> List[Dict]:
    """Get D1 candles from MT5."""
    if not MT5_AVAILABLE:
        return []
    
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, bars)
    if rates is None:
        return []
    
    return [
        {
            'time': datetime.fromtimestamp(r['time']),
            'open': r['open'],
            'high': r['high'],
            'low': r['low'],
            'close': r['close'],
        }
        for r in rates
    ]


def place_trade(signal: Signal, lot_size: float) -> Optional[int]:
    """Place a market order."""
    if not MT5_AVAILABLE:
        return None
    
    symbol_info = mt5.symbol_info(signal.symbol)
    if not symbol_info or not symbol_info.visible:
        mt5.symbol_select(signal.symbol, True)
        symbol_info = mt5.symbol_info(signal.symbol)
    
    order_type = mt5.ORDER_TYPE_BUY if signal.direction == 'bullish' else mt5.ORDER_TYPE_SELL
    tick = mt5.symbol_info_tick(signal.symbol)
    price = tick.ask if signal.direction == 'bullish' else tick.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": signal.symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "sl": signal.stop_loss,
        "tp": signal.tp1,
        "deviation": 20,
        "magic": 123456,
        "comment": f"Bot_{signal.confluence}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        logger.error(f"Order failed: {result.comment}")
        return None
    
    logger.info(f"✅ Order: {signal.symbol} {signal.direction} {lot_size:.2f} lots")
    return result.order


def get_open_positions() -> List[Dict]:
    """Get open positions."""
    if not MT5_AVAILABLE:
        return []
    
    positions = mt5.positions_get()
    return [{"symbol": p.symbol, "profit": p.profit} for p in (positions or [])]


def get_account_balance() -> float:
    """Get current account balance."""
    if not MT5_AVAILABLE:
        return 100000
    info = mt5.account_info()
    return info.balance if info else 100000


# ═══════════════════════════════════════════════════════════════════════════
# BOT CLASS
# ═══════════════════════════════════════════════════════════════════════════

class TradingBot:
    """Simple trading bot with prop firm compliant risk management."""
    
    def __init__(self, config: BotConfig, params: StrategyParams):
        self.config = config
        self.params = params
        self.running = False
        self.last_scan_date = None
        
        # Risk state tracking
        self.day_start_balance = config.initial_balance
        self.current_risk_multiplier = 1.0
        self.daily_halted = False
    
    def calculate_lot_size(self, signal: Signal, balance: float) -> float:
        """Calculate lot size based on risk, applying risk multiplier."""
        # Apply risk multiplier based on DD levels
        effective_risk = self.params.risk_per_trade_pct * self.current_risk_multiplier
        risk_amount = balance * (effective_risk / 100)
        sl_distance = abs(signal.entry - signal.stop_loss)
        
        # Approximate pip value
        pip_size = 0.0001 if 'JPY' not in signal.symbol else 0.01
        sl_pips = sl_distance / pip_size
        pip_value = 10.0  # Approximate
        
        lot_size = risk_amount / (sl_pips * pip_value) if sl_pips > 0 else 0.01
        return max(0.01, round(lot_size, 2))
    
    def update_risk_state(self) -> bool:
        """
        Update risk state based on current DD levels.
        
        Returns True if trading is allowed, False if halted.
        
        PROP FIRM RULES:
        - At 2% daily DD: Warning, slightly reduce risk
        - At 3.2% daily DD: TRADE HALF
        - At 4% daily DD: HALT all trading for the day
        - At 5% daily DD: HARD LIMIT - would fail challenge
        - At 7% total DD: Reduce to minimum risk
        - At 10% total DD: HARD LIMIT - would fail challenge
        """
        balance = get_account_balance()
        positions = get_open_positions()
        unrealized_pnl = sum(p['profit'] for p in positions)
        
        equity = balance + unrealized_pnl
        
        # Calculate daily drawdown (from day start balance)
        daily_dd_pct = 0.0
        if self.day_start_balance > 0:
            daily_dd_pct = max(0, (self.day_start_balance - equity) / self.day_start_balance * 100)
        
        # Calculate total drawdown (STATIC from initial balance)
        total_dd_pct = max(0, (self.config.initial_balance - equity) / self.config.initial_balance * 100)
        
        # ═══════════════════════════════════════════════════════════════
        # HARD STOPS - These would fail the challenge
        # ═══════════════════════════════════════════════════════════════
        
        if daily_dd_pct >= self.config.max_daily_dd_pct:
            logger.critical(f"🚨 DAILY DD LIMIT HIT: {daily_dd_pct:.2f}% >= {self.config.max_daily_dd_pct}%")
            logger.critical("🛑 CHALLENGE FAILED - Close all positions and stop trading!")
            self.daily_halted = True
            return False
        
        if total_dd_pct >= self.config.max_total_dd_pct:
            logger.critical(f"🚨 TOTAL DD LIMIT HIT: {total_dd_pct:.2f}% >= {self.config.max_total_dd_pct}%")
            logger.critical("🛑 CHALLENGE FAILED - Close all positions and stop trading!")
            self.daily_halted = True
            return False
        
        # ═══════════════════════════════════════════════════════════════
        # DAILY HALT - Stop trading for the day
        # ═══════════════════════════════════════════════════════════════
        
        if daily_dd_pct >= self.config.daily_dd_halt_pct:
            logger.warning(f"⛔ DAILY HALT: {daily_dd_pct:.2f}% >= {self.config.daily_dd_halt_pct}%")
            logger.warning("No more trades today - protecting account.")
            self.daily_halted = True
            return False
        
        # ═══════════════════════════════════════════════════════════════
        # RISK REDUCTION - Reduce position sizes
        # ═══════════════════════════════════════════════════════════════
        
        if daily_dd_pct >= self.config.daily_dd_reduce_pct or total_dd_pct >= self.config.total_dd_reduce_pct:
            # At 3.2% daily or 7% total: TRADE HALF
            self.current_risk_multiplier = self.config.reduced_risk_multiplier  # 0.5
            logger.warning(f"⚠️ RISK REDUCED TO 50%: Daily DD {daily_dd_pct:.2f}%, Total DD {total_dd_pct:.2f}%")
        elif daily_dd_pct >= self.config.daily_dd_warning_pct or total_dd_pct >= self.config.total_dd_warning_pct:
            # At 2% daily or 5% total: Slightly reduced
            self.current_risk_multiplier = 0.75
            logger.info(f"⚡ Risk at 75%: Daily DD {daily_dd_pct:.2f}%, Total DD {total_dd_pct:.2f}%")
        else:
            self.current_risk_multiplier = 1.0
        
        return not self.daily_halted
    
    def check_dd_safety(self) -> bool:
        """Check if within DD limits - called before every trade."""
        return self.update_risk_state()
    
    def daily_scan(self):
        """Scan for signals once per day."""
        if not self.check_dd_safety():
            return
        
        positions = get_open_positions()
        if len(positions) >= self.config.max_positions:
            logger.info(f"Max positions reached ({len(positions)})")
            return
        
        balance = get_account_balance()
        symbols_in_position = {p['symbol'] for p in positions}
        
        for symbol in self.config.symbols:
            if symbol in symbols_in_position:
                continue
            
            candles = get_candles(symbol, 100)
            if len(candles) < 50:
                continue
            
            signal = generate_signal(symbol, candles, self.params, datetime.utcnow())
            
            if signal:
                lot_size = self.calculate_lot_size(signal, balance)
                place_trade(signal, lot_size)
    
    def run(self):
        """Main loop with prop firm compliant risk management."""
        print("\n" + "=" * 50)
        print("🤖 TRADING BOT TEMPLATE")
        print("=" * 50)
        print("\n📋 5ERS RISK RULES ACTIVE:")
        print(f"   • Daily DD Limit: {self.config.max_daily_dd_pct}%")
        print(f"   • Total DD Limit: {self.config.max_total_dd_pct}% (static)")
        print(f"   • Trade Half at: {self.config.daily_dd_reduce_pct}% daily DD")
        print(f"   • Halt Trading at: {self.config.daily_dd_halt_pct}% daily DD")
        print("=" * 50)
        
        if not MT5_AVAILABLE:
            print("\n⚠️  MetaTrader5 not available!")
            print("   This bot only works on Windows with MT5 installed.")
            print("   Install: pip install MetaTrader5")
            print("\n   For backtesting, use: python optimizer.py")
            return
        
        if not init_mt5(self.config):
            return
        
        self.running = True
        
        # Initialize day start balance
        self.day_start_balance = get_account_balance()
        logger.info(f"Day start balance: ${self.day_start_balance:,.2f}")
        logger.info("Bot started. Waiting for daily scan time...")
        
        try:
            while self.running:
                now = datetime.now()
                today = now.date()
                
                # New day - reset daily state
                if self.last_scan_date != today and self.last_scan_date is not None:
                    self.day_start_balance = get_account_balance()
                    self.daily_halted = False
                    self.current_risk_multiplier = 1.0
                    logger.info(f"📅 New day! Day start balance: ${self.day_start_balance:,.2f}")
                
                # Continuous DD monitoring (every minute)
                self.update_risk_state()
                
                # Daily scan at configured time
                if (now.hour == self.config.scan_hour and 
                    now.minute >= self.config.scan_minute and
                    self.last_scan_date != today):
                    
                    if not self.daily_halted:
                        logger.info("📊 Running daily scan...")
                        self.daily_scan()
                    else:
                        logger.warning("⛔ Daily scan skipped - trading halted")
                    
                    self.last_scan_date = today
                
                time.sleep(60)
                
        except KeyboardInterrupt:
            logger.info("Stopped by user")
        finally:
            mt5.shutdown()


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # Load MT5 credentials from environment
    config = BotConfig(
        mt5_login=int(os.getenv("MT5_LOGIN", 0)),
        mt5_password=os.getenv("MT5_PASSWORD", ""),
        mt5_server=os.getenv("MT5_SERVER", ""),
        initial_balance=float(os.getenv("INITIAL_BALANCE", 100_000)),  # 5ers $100K
    )
    
    # Load optimized parameters
    params_file = Path("optimization_output/best_params.json")
    if params_file.exists():
        with open(params_file) as f:
            data = json.load(f)
            params_dict = data.get('best_params', data)
            params = StrategyParams(**{
                k: v for k, v in params_dict.items() 
                if hasattr(StrategyParams, k)
            })
        print(f"✅ Loaded params from {params_file}")
    else:
        params = get_default_params()
        print("ℹ️  Using default parameters")
    
    # Display key risk parameters
    print(f"\n💰 Account: ${config.initial_balance:,.0f}")
    print(f"📊 Risk per trade: {params.risk_per_trade_pct}%")
    
    bot = TradingBot(config, params)
    bot.run()


if __name__ == "__main__":
    main()
