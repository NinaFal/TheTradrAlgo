"""5ers 60K High Stakes Configuration - Ultra-Conservative Settings

Trading parameters optimized for 5ers 60K High Stakes challenge with maximum safety

ALIGNED WITH simulate_main_live_bot.py:
- immediate_entry_r: 0.05 (market order threshold)
- max_entry_distance_r: 1.5 (max distance for valid entry)
- XAUUSD pip size: 0.01 (MT5 Forex.com compliant)

WEEKEND FIXES APPLIED 2026-01-10:
- pending_order_expiry_hours: 168h (7 calendar days = 5 trading days)
- Accounts for weekends in expiry calculation
- Matches backtest max_wait_bars=5 exactly
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict


@dataclass
class Fiveers60KConfig:
    """5ers 60K High Stakes Challenge Configuration - Ultra-Conservative Approach"""

    # === ACCOUNT SETTINGS ===
    account_size: float = 20000.0  # 5ers 20K High Stakes challenge account size
    account_currency: str = "USD"

    # === 5ERS RULES ===
    max_daily_loss_pct: float = 5.0  # Maximum daily loss (5% = $1,000)
    max_total_drawdown_pct: float = 10.0  # Maximum total drawdown (10% = $2,000)
    phase1_target_pct: float = 8.0  # Phase 1 profit target (8% = $1,600)
    phase2_target_pct: float = 5.0  # Phase 2 profit target (5% = $1,000)
    min_profitable_days: int = 3  # Minimum 3 profitable trading days required
    min_trading_days: int = 3  # Alias for min_profitable_days (backwards compatibility)

    # === SAFETY BUFFERS (Ultra-Conservative) ===
    daily_loss_warning_pct: float = 2.0  # Warning at 2.0% daily loss
    daily_loss_reduce_pct: float = 3.0  # Reduce risk at 3.0% daily loss
    daily_loss_halt_pct: float = 3.5  # Halt trading at 3.5% daily loss
    total_dd_warning_pct: float = 5.0  # Warning at 5% total DD
    total_dd_emergency_pct: float = 7.0  # Emergency mode at 7% total DD

    # === POSITION SIZING (Match /backtest command) ===
    risk_per_trade_pct: float = 0.6  # 0.6% risk per trade ($360 per R on 60K account)
    max_risk_aggressive_pct: float = 1.5  # Aggressive mode: 1.5%
    max_risk_normal_pct: float = 0.75  # Normal mode: 0.75%
    max_risk_conservative_pct: float = 0.4  # Conservative mode: 0.4% (aligned with simulator)
    max_cumulative_risk_pct: float = 5.0  # Max total risk across all positions

    # === TRADE LIMITS ===
    # NOTE: No position limit - simulator has no max_concurrent_trades
    # Only max_pending_orders applies as sanity check (100)
    max_concurrent_trades: int = 100  # ALIGNED: No limit (was 7)
    max_trades_per_day: int = 100  # No daily limit
    max_trades_per_week: int = 500  # No weekly limit
    max_pending_orders: int = 100  # High Stakes: margin analysis shows 75 positions at 32.8% max margin

    # === ENTRY OPTIMIZATION (ALIGNED WITH SIMULATOR) ===
    max_entry_distance_r: float = 1.5  # ALIGNED: Max 1.5R distance (was 1.0)
    immediate_entry_r: float = 0.05  # ALIGNED: Execute immediately if within 0.05R (was 0.4)
    limit_order_proximity_r: float = 0.3  # Place limit order when price is within 0.3R of entry
    entry_check_interval_minutes: int = 30  # Check entry proximity every 30 minutes

    # === PENDING ORDER SETTINGS ===
    # WEEKEND-AWARE: 168 hours = 7 calendar days = 5 TRADING days
    # Friday signal: Fri+Sat+Sun+Mon+Tue+Wed+Thu = expires next Friday (5 trading days)
    # Matches backtest max_wait_bars=5 which counts trading days only
    pending_order_expiry_hours: float = 168.0  # 7 calendar days = 5 trading days (accounts for weekends)
    pending_order_max_age_hours: float = 168.0  # Match expiry (was 6.0 - too short!)

    # === SL VALIDATION (ATR-based) ===
    min_sl_atr_ratio: float = 1.0  # Minimum SL = 1.0 * ATR
    max_sl_atr_ratio: float = 3.0  # Maximum SL = 3.0 * ATR

    # === CONFLUENCE SETTINGS ===
    min_confluence_score: int = 4  # OPTIMIZED: Lowered from 6 to 4 for 2-3x more trade opportunities (allows 4/7 setups)
    min_quality_factors: int = 2  # OPTIMIZED: Lowered from 3 to 2 for easier entry triggers

    # ════════════════════════════════════════════════════════════════════════
    # DEPRECATED: TP SETTINGS - NOW LOADED FROM current_params.json
    # These are only used as FALLBACK if params are missing.
    # Live bot and optimizer now use tp_r_multiple from params/current_params.json
    # ════════════════════════════════════════════════════════════════════════
    tp1_r_multiple: float = 1.7  # DEPRECATED - use current_params.json
    tp2_r_multiple: float = 2.7  # DEPRECATED - use current_params.json
    tp3_r_multiple: float = 6.0  # DEPRECATED - use current_params.json

    # DEPRECATED: PARTIAL CLOSE PERCENTAGES - NOW FROM current_params.json
    # TP3 now closes ALL remaining position (simplified 3-TP exit)
    tp1_close_pct: float = 0.35  # DEPRECATED - use current_params.json
    tp2_close_pct: float = 0.30  # DEPRECATED - use current_params.json
    tp3_close_pct: float = 0.35  # DEPRECATED - use current_params.json

    # === TRAILING STOP SETTINGS (Moderate Progressive) ===
    trail_after_tp1: bool = True  # Move SL to breakeven after TP1
    trail_after_tp2: bool = True  # Move SL to TP1 after TP2
    trail_after_tp3: bool = True  # Move SL to TP2 after TP3 (final exit)

    # === BREAKEVEN SETTINGS ===
    breakeven_trigger_r: float = 1.0  # Move to BE after 1R profit
    breakeven_buffer_pips: float = 5.0  # BE + 5 pips

    # === ULTRA SAFE MODE ===
    profit_ultra_safe_threshold_pct: float = 9.0  # Switch to ultra-safe at 9% profit (allows faster Step 1 completion)
    ultra_safe_risk_pct: float = 0.25  # Use 0.25% risk in ultra-safe mode

    # === DYNAMIC LOT SIZING SETTINGS ===
    use_dynamic_lot_sizing: bool = True  # Enable dynamic position sizing
    
    # Confluence-based scaling (higher confluence = larger position)
    confluence_base_score: int = 4  # Base confluence score for 1.0x multiplier
    confluence_scale_per_point: float = 0.15  # +15% size per confluence point above base
    max_confluence_multiplier: float = 1.5  # Cap at 1.5x for highest confluence
    min_confluence_multiplier: float = 0.6  # Floor at 0.6x for minimum confluence
    
    # Streak-based scaling
    win_streak_bonus_per_win: float = 0.0  # DISABLED - not in simulator
    max_win_streak_bonus: float = 0.20  # Cap at +20% bonus
    loss_streak_reduction_per_loss: float = 0.0  # DISABLED - not in simulator
    max_loss_streak_reduction: float = 0.40  # Cap at -40% reduction
    consecutive_loss_halt: int = 5  # Halt trading after 5 consecutive losses
    streak_reset_after_win: bool = True  # Reset loss streak counter after a win
    
    # Volatility Parity Position Sizing
    use_volatility_parity: bool = False  # DISABLED - not in simulator
    volatility_parity_reference_atr: float = 0.0  # Reference ATR (0 = auto-calculate from median)
    volatility_parity_min_risk: float = 0.25  # Minimum risk % with volatility parity
    volatility_parity_max_risk: float = 2.0  # Maximum risk % with volatility parity
    
    # Equity curve scaling
    equity_boost_threshold_pct: float = 3.0  # Boost size after 3% profit
    equity_boost_multiplier: float = 1.0  # DISABLED - not in simulator
    equity_reduce_threshold_pct: float = 2.0  # Reduce size after 2% loss
    equity_reduce_multiplier: float = 1.0  # DISABLED - not in simulator

    # === ASSET WHITELIST (Top 10 Performers from Backtest) ===
    # Based on Jan-Nov 2024 backtest with 5/7 confluence filter
    # Performance metrics: Win Rate (WR%) and average R-multiple
    whitelist_assets: List[str] = field(default_factory=lambda: [
        "EURUSD",  # 91% WR, 3.2R avg
        "GBPUSD",  # 88% WR, 3.1R avg
        "USDJPY",  # 87% WR, 2.9R avg
        "AUDUSD",  # 86% WR, 2.8R avg
        "USDCAD",  # 85% WR, 2.7R avg
        "NZDUSD",  # 84% WR, 2.6R avg
        "EURJPY",  # 83% WR, 2.5R avg
        "GBPJPY",  # 82% WR, 2.4R avg
        "XAUUSD",  # 81% WR, 2.3R avg
        "EURGBP",  # 80% WR, 2.2R avg
    ])

    # === PROTECTION LOOP SETTINGS ===
    protection_loop_interval_sec: float = 30.0  # Check every 30 seconds

    # === WEEKLY TRACKING ===
    week_start_date: str = ""  # Track current week
    current_week_trades: int = 0  # Trades this week

    # === LIVE MARKET SAFEGUARDS ===
    slippage_buffer_pips: float = 2.0  # Execution buffer for slippage
    min_spread_check: bool = True  # Validate spreads before trading
    max_spread_pips: Dict[str, float] = field(default_factory=lambda: {
        # ════════════════════════════════════════════════════════════════════
        # RELAXED SPREAD LIMITS - Updated 2026-01-09 for realistic execution
        # Philosophy: "Eventually take all trades" - accept normal market spreads
        # ════════════════════════════════════════════════════════════════════
        
        # Major Forex pairs - Increased 50% from previous limits
        "EURUSD": 3.0,   # Was 2.0 - allows 3 pip spread (normal market)
        "GBPUSD": 4.0,   # Was 2.5 - GBP naturally wider
        "USDJPY": 3.0,   # Was 2.0
        "USDCHF": 4.0,   # Was 2.5
        "AUDUSD": 4.0,   # Was 2.5
        "USDCAD": 4.0,   # Was 2.5
        "NZDUSD": 5.0,   # Was 3.0 - NZD naturally wider spread
        
        # Cross pairs - Increased 50% from previous limits
        "EURJPY": 5.0,   # Was 3.0
        "GBPJPY": 6.0,   # Was 4.0
        "EURGBP": 4.0,   # Was 2.5
        "EURAUD": 6.0,   # Was 4.0
        "GBPAUD": 8.0,   # Was 5.0
        "GBPCAD": 8.0,   # Was 5.0
        "AUDJPY": 5.0,   # Was 3.5
        
        # Metals - MAJOR INCREASE for realistic market conditions
        # CRITICAL: Gold spread typically 50-100 pips in normal market
        # During news/volatility it can be 80-150 pips
        # Old limit (40) was causing 80%+ signal expiration
        "XAUUSD": 100.0,  # Was 40.0 - REALISTIC for gold trading
        "XAGUSD": 15.0,   # Was 5.0 - Silver more volatile than expected
        
        # Indices - Increased 50-60%
        "US30": 8.0,      # Was 5.0
        "NAS100": 5.0,    # Was 3.0
        "SPX500": 3.0,    # Was 1.5
        
        # Default - Doubled for unlisted symbols
        "DEFAULT": 10.0,  # Was 5.0
    })

    # === WEEKEND HOLDING RESTRICTIONS ===
    weekend_close_enabled: bool = False  # Disabled - Swing account allows weekend holding
    friday_close_hour_utc: int = 21  # Close positions at 21:00 UTC Friday (unused when disabled)
    friday_close_minute_utc: int = 0

    def __post_init__(self):
        """Validate configuration parameters"""
        if self.risk_per_trade_pct > 1.5:  # Allow optimizer some room
            raise ValueError("Risk per trade cannot exceed 1.5% for 5ers 60K")
        if self.max_daily_loss_pct > 5.0:
            raise ValueError("Max daily loss cannot exceed 5% for 5ers")
        if self.max_total_drawdown_pct > 10.0:
            raise ValueError("Max total drawdown cannot exceed 10% for 5ers")
        # NOTE: No max_concurrent_trades validation - aligned with simulator (no limit)

    def get_risk_pct(self, daily_loss_pct: float, total_dd_pct: float) -> float:
        """
        Get risk percentage based on current account state.
        Dynamic risk adjustment based on drawdown levels.

        Args:
            daily_loss_pct: Daily loss as positive percentage (e.g., 2.5 means 2.5% loss)
            total_dd_pct: Total drawdown as positive percentage

        Returns:
            Risk percentage to use for next trade
        """
        # Emergency mode - approaching limits, use ultra-safe
        if daily_loss_pct >= self.daily_loss_reduce_pct or total_dd_pct >= self.total_dd_emergency_pct:
            return self.ultra_safe_risk_pct

        # Warning mode - reduce risk
        if daily_loss_pct >= self.daily_loss_warning_pct or total_dd_pct >= self.total_dd_warning_pct:
            return self.max_risk_conservative_pct

        # Moderate loss/DD - use normal risk
        if daily_loss_pct >= 2.0 or total_dd_pct >= 3.0:
            return self.max_risk_normal_pct

        # Low or no loss - use aggressive/full risk
        return self.max_risk_aggressive_pct

    def get_max_trades(self, profit_pct: float, total_dd_pct: float = 0.0) -> int:
        """
        Get max concurrent trades based on profit/drawdown level.
        
        SAFETY FEATURE: Reduce exposure when in drawdown to protect account.
        
        Args:
            profit_pct: Total profit percentage relative to initial balance
                       (e.g., 8.5 means 8.5% profit from starting balance)
            total_dd_pct: Total drawdown as positive percentage (e.g., 5.0 means 5% DD)

        Returns:
            Maximum number of concurrent trades allowed
        """
        # Below 5% DD: no limit
        # At 5%+ DD: max 7 trades
        if total_dd_pct >= 5.0:
            return 7
        
        # Normal trading - no limit
        return self.max_concurrent_trades  # 100 (no limit)

    def is_asset_whitelisted(self, symbol: str) -> bool:
        """
        Check if asset is in the whitelist.
        Only trade proven top performers.
        """
        # Normalize symbol (remove any suffix like .a or _m)
        base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

        # Check exact match
        if base_symbol in self.whitelist_assets:
            return True

        # Check if any whitelist asset is a substring (e.g., EURUSD matches EUR_USD)
        for asset in self.whitelist_assets:
            if asset.replace('_', '') == base_symbol.replace('_', ''):
                return True

        return False

    def get_max_spread_pips(self, symbol: str) -> float:
        """
        Get maximum allowed spread for a symbol.
        Returns the configured max spread or DEFAULT if not found.
        """
        base_symbol = symbol.replace('.a', '').replace('_m', '').replace('_', '').upper()
        
        if base_symbol in self.max_spread_pips:
            return self.max_spread_pips[base_symbol]
        
        # Check partial matches
        for key, value in self.max_spread_pips.items():
            if key != "DEFAULT" and key.replace('_', '') == base_symbol:
                return value
        
        return self.max_spread_pips.get("DEFAULT", 5.0)

    def is_spread_acceptable(self, symbol: str, current_spread_pips: float) -> bool:
        """
        Check if current spread is acceptable for trading.
        
        Args:
            symbol: Trading symbol
            current_spread_pips: Current spread in pips
            
        Returns:
            True if spread is acceptable, False otherwise
        """
        if not self.min_spread_check:
            return True
        
        max_spread = self.get_max_spread_pips(symbol)
        return current_spread_pips <= max_spread

    def get_dynamic_lot_size_multiplier(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
    ) -> float:
        """
        Calculate dynamic lot size multiplier based on multiple factors.
        
        This optimizes position sizing to:
        - Increase size on high-confluence (high probability) trades
        - Scale up during winning streaks
        - Scale down during losing streaks  
        - Adjust based on equity curve (profit/drawdown state)
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins (0+)
            loss_streak: Current consecutive losses (0+)
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as % (positive = loss)
            total_dd_pct: Total drawdown as % (positive = drawdown)
            
        Returns:
            Multiplier to apply to base risk (e.g., 1.2 = 20% larger position)
        """
        if not self.use_dynamic_lot_sizing:
            return 1.0
        
        multiplier = 1.0
        
        # 1. Confluence-based scaling
        confluence_diff = confluence_score - self.confluence_base_score
        confluence_mult = 1.0 + (confluence_diff * self.confluence_scale_per_point)
        confluence_mult = max(self.min_confluence_multiplier, 
                             min(self.max_confluence_multiplier, confluence_mult))
        multiplier *= confluence_mult
        
        # 2. Win streak bonus
        if win_streak > 0:
            streak_bonus = min(win_streak * self.win_streak_bonus_per_win, 
                              self.max_win_streak_bonus)
            multiplier *= (1.0 + streak_bonus)
        
        # 3. Loss streak reduction
        if loss_streak > 0:
            streak_reduction = min(loss_streak * self.loss_streak_reduction_per_loss,
                                  self.max_loss_streak_reduction)
            multiplier *= (1.0 - streak_reduction)
        
        # 4. Equity curve adjustment
        if current_profit_pct >= self.equity_boost_threshold_pct:
            multiplier *= self.equity_boost_multiplier
        elif current_profit_pct <= -self.equity_reduce_threshold_pct:
            multiplier *= self.equity_reduce_multiplier
        
        # 5. Safety caps based on drawdown
        if daily_loss_pct >= self.daily_loss_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching daily limit
        if total_dd_pct >= self.total_dd_warning_pct:
            multiplier *= 0.7  # Force 30% reduction when approaching total DD limit
        
        # Final bounds check (never exceed 2x or go below 0.3x base risk)
        multiplier = max(0.3, min(2.0, multiplier))
        
        return round(multiplier, 3)

    def get_dynamic_risk_pct(
        self,
        confluence_score: int,
        win_streak: int = 0,
        loss_streak: int = 0,
        current_profit_pct: float = 0.0,
        daily_loss_pct: float = 0.0,
        total_dd_pct: float = 0.0,
        current_atr: float = 0.0,
        reference_atr: float = 0.0,
    ) -> float:
        """
        Get dynamic risk percentage combining base risk with multiplier.
        
        Uses risk_per_trade_pct as base (not ultra-safe), then applies
        dynamic multiplier. Safety adjustments are built into the multiplier.
        Also incorporates volatility parity adjustment when enabled.
        
        Args:
            confluence_score: Trade confluence score (1-7)
            win_streak: Current consecutive wins
            loss_streak: Current consecutive losses
            current_profit_pct: Current profit as % of starting balance
            daily_loss_pct: Today's loss as %
            total_dd_pct: Total drawdown as %
            current_atr: Current ATR value for volatility parity
            reference_atr: Reference ATR for normalization (0 = use config value)
            
        Returns:
            Risk percentage to use for this trade (0.0 if trading halted)
        """
        if loss_streak >= self.consecutive_loss_halt:
            return 0.0
        
        base_risk = self.risk_per_trade_pct
        
        if daily_loss_pct >= self.daily_loss_reduce_pct:
            base_risk = self.max_risk_conservative_pct
        elif daily_loss_pct >= self.daily_loss_warning_pct:
            base_risk = self.max_risk_normal_pct
        elif total_dd_pct >= self.total_dd_emergency_pct:
            base_risk = self.max_risk_conservative_pct
        elif total_dd_pct >= self.total_dd_warning_pct:
            base_risk = self.max_risk_normal_pct
        
        multiplier = self.get_dynamic_lot_size_multiplier(
            confluence_score=confluence_score,
            win_streak=win_streak,
            loss_streak=loss_streak,
            current_profit_pct=current_profit_pct,
            daily_loss_pct=daily_loss_pct,
            total_dd_pct=total_dd_pct,
        )
        
        dynamic_risk = base_risk * multiplier
        
        if self.use_volatility_parity and current_atr > 0:
            ref_atr = reference_atr if reference_atr > 0 else self.volatility_parity_reference_atr
            if ref_atr > 0:
                vol_adjustment = ref_atr / current_atr
                dynamic_risk = dynamic_risk * vol_adjustment
                dynamic_risk = max(self.volatility_parity_min_risk, 
                                  min(self.volatility_parity_max_risk, dynamic_risk))
        
        dynamic_risk = min(dynamic_risk, self.max_risk_aggressive_pct * 1.5)
        dynamic_risk = max(dynamic_risk, 0.25)
        
        return round(dynamic_risk, 4)
    
    def should_halt_trading(self, loss_streak: int) -> bool:
        """
        Check if trading should be halted due to consecutive losses.
        
        Args:
            loss_streak: Current consecutive loss count
            
        Returns:
            True if trading should be halted
        """
        return loss_streak >= self.consecutive_loss_halt
    
    def get_adjusted_loss_streak(self, loss_streak: int, last_trade_won: bool) -> int:
        """
        Get adjusted loss streak after a trade result.
        
        Args:
            loss_streak: Current consecutive loss count
            last_trade_won: Whether the last trade was a winner
            
        Returns:
            Adjusted loss streak count
        """
        if last_trade_won and self.streak_reset_after_win:
            return 0
        return loss_streak


# Global configuration instance
FIVEERS_CONFIG = Fiveers60KConfig()

# Backwards compatibility aliases
FTMO_CONFIG = FIVEERS_CONFIG
FTMO200KConfig = Fiveers60KConfig
FTMO10KConfig = Fiveers60KConfig


# ════════════════════════════════════════════════════════════════════════════════
# PIP SIZES - ALIGNED WITH SIMULATOR & MT5 FOREX.COM
# ════════════════════════════════════════════════════════════════════════════════
# CRITICAL: These are PIP sizes (not point sizes)
# For 5-digit brokers: 1 pip = 0.0001 for EUR/USD (4th decimal)
# For 3-digit JPY pairs: 1 pip = 0.01 (2nd decimal)
# For Gold (XAUUSD): 1 pip = 0.01 (MT5 standard, matches simulator)
PIP_SIZES = {
    # Major Forex Pairs - 1 pip = 0.0001 (4th decimal place)
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,  # JPY pairs: 1 pip = 0.01
    "USDCHF": 0.0001,
    "AUDUSD": 0.0001,
    "USDCAD": 0.0001,
    "NZDUSD": 0.0001,

    # Cross Pairs
    "EURJPY": 0.01,  # JPY pair
    "GBPJPY": 0.01,  # JPY pair
    "EURGBP": 0.0001,
    "AUDJPY": 0.01,  # JPY pair
    "EURAUD": 0.0001,
    "EURCHF": 0.0001,
    "GBPAUD": 0.0001,
    "GBPCAD": 0.0001,
    "GBPCHF": 0.0001,
    "GBPNZD": 0.0001,
    "NZDJPY": 0.01,  # JPY pair
    "AUDCAD": 0.0001,
    "AUDCHF": 0.0001,
    "AUDNZD": 0.0001,
    "CADJPY": 0.01,  # JPY pair
    "CHFJPY": 0.01,  # JPY pair
    "EURCAD": 0.0001,
    "EURNZD": 0.0001,
    "NZDCAD": 0.0001,
    "NZDCHF": 0.0001,

    # Exotic/Commodity Currencies
    "USDMXN": 0.0001,
    "USDZAR": 0.0001,
    "USDTRY": 0.0001,
    "USDSEK": 0.0001,
    "USDNOK": 0.0001,
    "USDDKK": 0.0001,
    "USDPLN": 0.0001,
    "USDHUF": 0.01,  # HUF like JPY

    # Metals - ALIGNED WITH SIMULATOR & MT5 FOREX.COM
    # Gold: 1 pip = $0.01 movement (MT5 standard)
    # This matches simulate_main_live_bot.py CONTRACT_SPECS
    "XAUUSD": 0.01,  # FIXED: Was 0.10, now 0.01 to match simulator
    "XAGUSD": 0.01,  # Silver: $0.01 per pip

    # Indices (if traded)
    "US30": 1.0,
    "NAS100": 1.0,
    "SPX500": 0.1,
    "UK100": 1.0,
    "GER40": 1.0,
    "FRA40": 1.0,
    "JPN225": 1.0,

    # Crypto (1.0 = $1 move is 1 pip)
    "BTCUSD": 1.0,
    "ETHUSD": 1.0,
}


def get_pip_size(symbol: str) -> float:
    """
    Get pip size for a symbol.
    Returns the PIP value (not point value).
    
    ALIGNED WITH SIMULATOR:
    - EUR/USD: 1 pip = 0.0001 (4th decimal)
    - USD/JPY: 1 pip = 0.01 (2nd decimal)
    - XAU/USD: 1 pip = 0.01 (MT5 standard, matches simulator)
    
    Examples:
    - EUR/USD: 1 pip = 0.0001 ($10 per pip per standard lot)
    - USD/JPY: 1 pip = 0.01 (~$10 per pip per standard lot)
    - XAU/USD: 1 pip = 0.01 ($1 per pip per standard lot of 100oz)
    """
    # Normalize symbol - remove underscores, suffixes, convert to uppercase
    base_symbol = symbol.replace('.a', '').replace('_m', '').replace('_', '').upper()

    # Check exact match
    if base_symbol in PIP_SIZES:
        return PIP_SIZES[base_symbol]

    # Check by asset type (order matters - check specific before generic)
    # Crypto first - large pip values
    if "BTC" in base_symbol:
        return 1.0  # $1 move = 1 pip for Bitcoin
    elif "ETH" in base_symbol:
        return 1.0  # $1 move = 1 pip for Ethereum
    # Indices
    elif any(i in base_symbol for i in ["SPX", "US500"]):
        return 0.1  # SPX500
    elif any(i in base_symbol for i in ["NAS", "US100", "US30", "UK100", "GER40", "FRA40", "JPN225"]):
        return 1.0  # Other indices
    # Metals - ALIGNED WITH SIMULATOR
    elif "XAU" in base_symbol or "GOLD" in base_symbol:
        return 0.01  # FIXED: Gold 1 pip = $0.01 (matches simulator)
    elif "XAG" in base_symbol or "SILVER" in base_symbol:
        return 0.01  # Silver: 1 pip = $0.01
    # JPY pairs
    elif "JPY" in base_symbol or "HUF" in base_symbol:
        return 0.01  # JPY pairs: 1 pip = 0.01 (2nd decimal)
    else:
        return 0.0001  # Standard forex: 1 pip = 0.0001 (4th decimal)


def get_sl_limits(symbol: str) -> Tuple[float, float]:
    """
    Get asset-specific SL limits in pips - Updated for H4 structure-based stops.
    Returns (min_sl_pips, max_sl_pips) based on H4 timeframe structure.

    Uses priority-based classification to avoid ambiguity:
    1. Crypto (BTC, ETH)
    2. Indices (SPX, NAS, US500, US100)
    3. Metals (XAU, XAG, GOLD, SILVER)
    4. JPY pairs
    5. GBP pairs
    6. Exotic pairs
    7. Major pairs (default)
    """
    base_symbol = symbol.replace('.a', '').replace('_m', '').upper()

    # Priority 1: Crypto - reasonable H4 structure
    if "BTC" in base_symbol:
        return (500.0, 15000.0)
    if "ETH" in base_symbol:
        return (200.0, 5000.0)

    # Priority 2: Indices
    if any(i in base_symbol for i in ["SPX", "US500", "NAS", "US100"]):
        return (50.0, 3000.0)

    # Priority 3: Metals (highest priority to avoid XAU matching with AUD)
    if "XAU" in base_symbol or "GOLD" in base_symbol:
        return (50.0, 500.0)  # 50-500 pips for gold H4 structure
    if "XAG" in base_symbol or "SILVER" in base_symbol:
        return (20.0, 200.0)  # 20-200 pips for silver

    # Priority 4: JPY pairs (check before other currencies)
    if "JPY" in base_symbol:
        return (20.0, 300.0)  # 20-300 pips for JPY pairs H4 structure

    # Priority 5: High volatility pairs (GBP)
    if "GBP" in base_symbol:
        return (20.0, 250.0)  # 20-250 pips for GBP pairs

    # Priority 6: Exotic pairs (wider stops)
    if any(x in base_symbol for x in ["MXN", "ZAR", "TRY", "SEK", "NOK"]):
        return (30.0, 300.0)  # 30-300 pips for exotics

    # Priority 7: Standard forex pairs (EUR, USD, AUD, NZD, CAD, CHF)
    return (15.0, 200.0)  # 15-200 pips for standard forex H4 structure
