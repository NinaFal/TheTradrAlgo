#!/usr/bin/env python3
"""
🎯 TRADING BOT TEMPLATE - Confluence Strategy Module
=====================================================

A TEMPLATE strategy using a confluence-based approach.
This is DIFFERENT from other trading systems - customize it!

STRATEGY: Multi-Factor Confluence System
- 5 independent confluence pillars
- Regime-adaptive (trend vs range detection)
- Multiple indicator confirmation
- Structure-based entries

CONFLUENCE PILLARS:
1. Trend Bias (MA alignment)
2. Momentum (MACD + Stochastic)
3. Volatility (Bollinger Band position)
4. Structure (swing high/low break)
5. Price Action confirmation

CUSTOMIZE:
1. Add your own confluence factors
2. Modify entry/exit logic
3. Adjust parameters via Optuna optimization

Author: Trading Bot Template
Version: 2.0.0 - Confluence Edition
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Tuple, Any
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# STRATEGY PARAMETERS - Customize these!
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class StrategyParams:
    """
    Strategy parameters that can be optimized via TPE.
    
    CUSTOMIZE: Add your own parameters here for optimization.
    """
    # === RISK MANAGEMENT ===
    risk_per_trade_pct: float = 0.5  # Risk per trade (0.5% = safe for 5ers)
    
    # === TREND DETECTION (SMA) ===
    sma_fast_period: int = 20       # Fast SMA period (10-30)
    sma_slow_period: int = 50       # Slow SMA period (40-100)
    sma_trend_period: int = 200     # Long-term trend SMA
    
    # === MACD SETTINGS ===
    macd_fast: int = 12             # MACD fast EMA
    macd_slow: int = 26             # MACD slow EMA
    macd_signal: int = 9            # MACD signal line
    
    # === STOCHASTIC SETTINGS ===
    stoch_k_period: int = 14        # Stochastic %K period
    stoch_d_period: int = 3         # Stochastic %D period
    stoch_overbought: float = 80.0  # Overbought level
    stoch_oversold: float = 20.0    # Oversold level
    
    # === BOLLINGER BANDS ===
    bb_period: int = 20             # BB period
    bb_std_dev: float = 2.0         # BB standard deviations
    
    # === ATR SETTINGS ===
    atr_period: int = 14            # ATR calculation period
    atr_sl_multiplier: float = 1.5  # Stop loss = ATR * multiplier
    
    # === STRUCTURE DETECTION ===
    swing_lookback: int = 5         # Bars to find swing highs/lows
    structure_break_atr: float = 0.3  # Min break beyond structure (ATR%)
    
    # === TAKE PROFIT LEVELS (R-multiples) ===
    tp1_r_multiple: float = 1.5     # TP1 at 1.5R (quick profit)
    tp2_r_multiple: float = 3.0     # TP2 at 3.0R (standard target)
    tp3_r_multiple: float = 5.0     # TP3 at 5.0R (runner)
    
    # === PARTIAL CLOSE PERCENTAGES ===
    tp1_close_pct: float = 0.35     # Close 35% at TP1
    tp2_close_pct: float = 0.35     # Close 35% at TP2
    tp3_close_pct: float = 0.30     # Close 30% at TP3
    
    # === CONFLUENCE REQUIREMENTS ===
    min_confluence: int = 3         # Minimum score to take trade (0-5)
    
    # === REGIME FILTER ===
    use_regime_filter: bool = True  # Enable trend/range detection
    range_atr_threshold: float = 0.7  # ATR percentile for range mode
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            "risk_per_trade_pct": self.risk_per_trade_pct,
            "sma_fast_period": self.sma_fast_period,
            "sma_slow_period": self.sma_slow_period,
            "sma_trend_period": self.sma_trend_period,
            "macd_fast": self.macd_fast,
            "macd_slow": self.macd_slow,
            "macd_signal": self.macd_signal,
            "stoch_k_period": self.stoch_k_period,
            "stoch_d_period": self.stoch_d_period,
            "stoch_overbought": self.stoch_overbought,
            "stoch_oversold": self.stoch_oversold,
            "bb_period": self.bb_period,
            "bb_std_dev": self.bb_std_dev,
            "atr_period": self.atr_period,
            "atr_sl_multiplier": self.atr_sl_multiplier,
            "swing_lookback": self.swing_lookback,
            "structure_break_atr": self.structure_break_atr,
            "tp1_r_multiple": self.tp1_r_multiple,
            "tp2_r_multiple": self.tp2_r_multiple,
            "tp3_r_multiple": self.tp3_r_multiple,
            "tp1_close_pct": self.tp1_close_pct,
            "tp2_close_pct": self.tp2_close_pct,
            "tp3_close_pct": self.tp3_close_pct,
            "min_confluence": self.min_confluence,
            "use_regime_filter": self.use_regime_filter,
            "range_atr_threshold": self.range_atr_threshold,
        }
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StrategyParams":
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in d.items() if hasattr(cls, k)})


# ═══════════════════════════════════════════════════════════════════════════
# SIGNAL CLASS
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Signal:
    """Trading signal with all necessary information."""
    symbol: str
    direction: str  # 'bullish' or 'bearish'
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confluence: int
    signal_time: datetime
    regime: str = "trend"  # 'trend' or 'range'
    reason: str = ""
    
    @property
    def risk(self) -> float:
        """Risk in price terms (|entry - sl|)."""
        return abs(self.entry - self.stop_loss)
    
    @property
    def risk_reward_ratio(self) -> float:
        """Risk/reward ratio to TP2."""
        risk = self.risk
        if risk == 0:
            return 0
        reward = abs(self.tp2 - self.entry)
        return reward / risk


# ═══════════════════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════════════════

def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Simple Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    sma = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        sma[i] = np.mean(prices[i - period + 1:i + 1])
    
    return sma


def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Calculate Exponential Moving Average."""
    if len(prices) < period:
        return np.full(len(prices), np.nan)
    
    ema = np.zeros(len(prices))
    ema[:period] = np.nan
    
    # Initial SMA
    ema[period-1] = np.mean(prices[:period])
    
    # EMA calculation
    multiplier = 2 / (period + 1)
    for i in range(period, len(prices)):
        ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema


def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD, Signal line, and Histogram.
    
    Returns: (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line[~np.isnan(macd_line)], signal)
    
    # Pad signal line to match length
    padded_signal = np.full(len(prices), np.nan)
    start_idx = len(prices) - len(signal_line)
    padded_signal[start_idx:] = signal_line
    
    histogram = macd_line - padded_signal
    
    return macd_line, padded_signal, histogram


def calculate_stochastic(high: np.ndarray, low: np.ndarray, close: np.ndarray, 
                         k_period: int = 14, d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator (%K and %D).
    
    Returns: (stoch_k, stoch_d)
    """
    n = len(close)
    stoch_k = np.full(n, np.nan)
    
    for i in range(k_period - 1, n):
        highest = np.max(high[i - k_period + 1:i + 1])
        lowest = np.min(low[i - k_period + 1:i + 1])
        
        if highest != lowest:
            stoch_k[i] = 100 * (close[i] - lowest) / (highest - lowest)
        else:
            stoch_k[i] = 50  # Neutral if no range
    
    # %D is SMA of %K
    stoch_d = calculate_sma(stoch_k, d_period)
    
    return stoch_k, stoch_d


def calculate_bollinger_bands(prices: np.ndarray, period: int = 20, 
                              std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands.
    
    Returns: (upper_band, middle_band, lower_band)
    """
    middle = calculate_sma(prices, period)
    
    std = np.full(len(prices), np.nan)
    for i in range(period - 1, len(prices)):
        std[i] = np.std(prices[i - period + 1:i + 1])
    
    upper = middle + (std_dev * std)
    lower = middle - (std_dev * std)
    
    return upper, middle, lower


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range."""
    if len(high) < period + 1:
        return np.full(len(high), np.nan)
    
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(len(high))
    atr[:period] = np.nan
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(high)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


def find_swing_points(high: np.ndarray, low: np.ndarray, lookback: int = 5) -> Tuple[List[int], List[int]]:
    """
    Find swing high and swing low indices.
    
    Returns: (swing_high_indices, swing_low_indices)
    """
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(high) - lookback):
        # Swing High: highest in range
        if high[i] == np.max(high[i - lookback:i + lookback + 1]):
            swing_highs.append(i)
        
        # Swing Low: lowest in range
        if low[i] == np.min(low[i - lookback:i + lookback + 1]):
            swing_lows.append(i)
    
    return swing_highs, swing_lows


# ═══════════════════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════════════════

def detect_regime(close: np.ndarray, atr: np.ndarray, 
                  sma_fast: np.ndarray, sma_slow: np.ndarray,
                  threshold: float = 0.7) -> str:
    """
    Detect market regime (trending vs ranging).
    
    Uses ATR percentile and MA alignment.
    
    Returns: 'trend' or 'range'
    """
    if len(atr) < 50 or np.isnan(atr[-1]):
        return 'trend'  # Default to trend
    
    # Calculate ATR percentile (current vs last 50 bars)
    recent_atr = atr[-50:]
    recent_atr = recent_atr[~np.isnan(recent_atr)]
    if len(recent_atr) < 20:
        return 'trend'
    
    current_atr = atr[-1]
    atr_percentile = np.sum(recent_atr < current_atr) / len(recent_atr)
    
    # Check MA alignment
    if np.isnan(sma_fast[-1]) or np.isnan(sma_slow[-1]):
        return 'trend'
    
    ma_spread = abs(sma_fast[-1] - sma_slow[-1]) / close[-1] * 100
    
    # Trending: High ATR percentile AND wide MA spread
    if atr_percentile > threshold and ma_spread > 0.5:
        return 'trend'
    
    # Ranging: Low ATR percentile OR narrow MA spread
    if atr_percentile < (1 - threshold) or ma_spread < 0.2:
        return 'range'
    
    return 'trend'  # Default


# ═══════════════════════════════════════════════════════════════════════════
# CONFLUENCE CALCULATION
# ═══════════════════════════════════════════════════════════════════════════

def compute_confluence(
    direction: str,
    close: float,
    indicators: Dict[str, Any],
    params: StrategyParams
) -> Tuple[int, List[str]]:
    """
    Compute confluence score based on 5 independent pillars.
    
    PILLARS:
    1. TREND BIAS - MA alignment (SMA fast/slow/trend)
    2. MOMENTUM - MACD + Stochastic agreement
    3. VOLATILITY - Bollinger Band position
    4. STRUCTURE - Swing break or retest
    5. CONFIRMATION - Price action
    
    Returns: (confluence_score, list_of_reasons)
    """
    confluence = 0
    reasons = []
    
    # Extract indicator values
    sma_fast = indicators.get('sma_fast', np.array([np.nan]))[-1]
    sma_slow = indicators.get('sma_slow', np.array([np.nan]))[-1]
    sma_trend = indicators.get('sma_trend', np.array([np.nan]))[-1]
    
    macd_line = indicators.get('macd_line', np.array([np.nan]))[-1]
    macd_signal = indicators.get('macd_signal', np.array([np.nan]))[-1]
    macd_hist = indicators.get('macd_hist', np.array([np.nan]))[-1]
    
    stoch_k = indicators.get('stoch_k', np.array([np.nan]))[-1]
    stoch_d = indicators.get('stoch_d', np.array([np.nan]))[-1]
    
    bb_upper = indicators.get('bb_upper', np.array([np.nan]))[-1]
    bb_middle = indicators.get('bb_middle', np.array([np.nan]))[-1]
    bb_lower = indicators.get('bb_lower', np.array([np.nan]))[-1]
    
    atr = indicators.get('atr', np.array([np.nan]))[-1]
    
    # Previous values for momentum
    prev_macd_hist = indicators.get('macd_hist', np.array([np.nan, np.nan]))[-2] if len(indicators.get('macd_hist', [])) > 1 else np.nan
    prev_close = indicators.get('close', np.array([np.nan, np.nan]))[-2] if len(indicators.get('close', [])) > 1 else close
    
    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 1: TREND BIAS (MA alignment)
    # ═══════════════════════════════════════════════════════════════════════
    
    if not np.isnan(sma_fast) and not np.isnan(sma_slow):
        if direction == 'bullish':
            # Bullish: price > fast > slow (ideally > trend)
            if close > sma_fast > sma_slow:
                confluence += 1
                reasons.append("📈 Bullish MA alignment")
        else:
            # Bearish: price < fast < slow (ideally < trend)
            if close < sma_fast < sma_slow:
                confluence += 1
                reasons.append("📉 Bearish MA alignment")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 2: MOMENTUM (MACD + Stochastic)
    # ═══════════════════════════════════════════════════════════════════════
    
    macd_bullish = False
    macd_bearish = False
    stoch_bullish = False
    stoch_bearish = False
    
    # MACD check
    if not np.isnan(macd_line) and not np.isnan(macd_signal):
        if macd_line > macd_signal and macd_hist > 0:
            macd_bullish = True
        elif macd_line < macd_signal and macd_hist < 0:
            macd_bearish = True
    
    # Stochastic check
    if not np.isnan(stoch_k) and not np.isnan(stoch_d):
        if stoch_k > stoch_d and stoch_k < params.stoch_overbought:
            stoch_bullish = True
        elif stoch_k < stoch_d and stoch_k > params.stoch_oversold:
            stoch_bearish = True
    
    # Award confluence if both agree
    if direction == 'bullish' and (macd_bullish or stoch_bullish):
        confluence += 1
        if macd_bullish and stoch_bullish:
            reasons.append("🚀 Strong bullish momentum (MACD + Stoch)")
        elif macd_bullish:
            reasons.append("📊 MACD bullish")
        else:
            reasons.append("📊 Stochastic bullish")
    elif direction == 'bearish' and (macd_bearish or stoch_bearish):
        confluence += 1
        if macd_bearish and stoch_bearish:
            reasons.append("🔻 Strong bearish momentum (MACD + Stoch)")
        elif macd_bearish:
            reasons.append("📊 MACD bearish")
        else:
            reasons.append("📊 Stochastic bearish")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 3: VOLATILITY (Bollinger Band position)
    # ═══════════════════════════════════════════════════════════════════════
    
    if not np.isnan(bb_upper) and not np.isnan(bb_lower) and not np.isnan(bb_middle):
        bb_width = bb_upper - bb_lower
        
        if direction == 'bullish':
            # Bullish: price in lower half of BB (room to run up)
            if close < bb_middle and close > bb_lower:
                confluence += 1
                reasons.append("📍 Price in lower BB zone (upside room)")
        else:
            # Bearish: price in upper half of BB (room to run down)
            if close > bb_middle and close < bb_upper:
                confluence += 1
                reasons.append("📍 Price in upper BB zone (downside room)")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 4: STRUCTURE (Recent swing break/retest)
    # ═══════════════════════════════════════════════════════════════════════
    
    swing_highs = indicators.get('swing_highs', [])
    swing_lows = indicators.get('swing_lows', [])
    high_arr = indicators.get('high', np.array([]))
    low_arr = indicators.get('low', np.array([]))
    
    if len(swing_highs) > 0 and len(swing_lows) > 0 and len(high_arr) > 0:
        # Get most recent swing levels
        if len(swing_highs) > 0:
            last_swing_high = high_arr[swing_highs[-1]] if swing_highs[-1] < len(high_arr) else np.nan
        else:
            last_swing_high = np.nan
            
        if len(swing_lows) > 0:
            last_swing_low = low_arr[swing_lows[-1]] if swing_lows[-1] < len(low_arr) else np.nan
        else:
            last_swing_low = np.nan
        
        structure_tolerance = atr * params.structure_break_atr if not np.isnan(atr) else 0
        
        if direction == 'bullish' and not np.isnan(last_swing_low):
            # Bullish structure: price above last swing low (support held)
            if close > last_swing_low + structure_tolerance:
                confluence += 1
                reasons.append("🏗️ Above swing low support")
        elif direction == 'bearish' and not np.isnan(last_swing_high):
            # Bearish structure: price below last swing high (resistance held)
            if close < last_swing_high - structure_tolerance:
                confluence += 1
                reasons.append("🏗️ Below swing high resistance")
    
    # ═══════════════════════════════════════════════════════════════════════
    # PILLAR 5: PRICE ACTION CONFIRMATION
    # ═══════════════════════════════════════════════════════════════════════
    
    if not np.isnan(prev_close):
        # Momentum building
        if direction == 'bullish':
            if close > prev_close:  # Higher close
                confluence += 1
                reasons.append("✅ Bullish price action")
        else:
            if close < prev_close:  # Lower close
                confluence += 1
                reasons.append("✅ Bearish price action")
    
    return confluence, reasons


# ═══════════════════════════════════════════════════════════════════════════
# MAIN STRATEGY LOGIC
# ═══════════════════════════════════════════════════════════════════════════

def compute_indicators(candles: List[Dict], params: StrategyParams) -> Dict[str, Any]:
    """
    Compute all technical indicators from candle data.
    
    Args:
        candles: List of OHLC candles with 'open', 'high', 'low', 'close'
        params: Strategy parameters
    
    Returns:
        Dictionary of indicator arrays
    """
    if not candles or len(candles) < 50:
        return {}
    
    # Extract price arrays
    close = np.array([c['close'] for c in candles])
    high = np.array([c['high'] for c in candles])
    low = np.array([c['low'] for c in candles])
    open_price = np.array([c['open'] for c in candles])
    
    # SMA indicators
    sma_fast = calculate_sma(close, params.sma_fast_period)
    sma_slow = calculate_sma(close, params.sma_slow_period)
    sma_trend = calculate_sma(close, params.sma_trend_period)
    
    # MACD
    macd_line, macd_signal, macd_hist = calculate_macd(
        close, params.macd_fast, params.macd_slow, params.macd_signal
    )
    
    # Stochastic
    stoch_k, stoch_d = calculate_stochastic(
        high, low, close, params.stoch_k_period, params.stoch_d_period
    )
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(
        close, params.bb_period, params.bb_std_dev
    )
    
    # ATR
    atr = calculate_atr(high, low, close, params.atr_period)
    
    # Swing points
    swing_highs, swing_lows = find_swing_points(high, low, params.swing_lookback)
    
    indicators = {
        'close': close,
        'high': high,
        'low': low,
        'open': open_price,
        'sma_fast': sma_fast,
        'sma_slow': sma_slow,
        'sma_trend': sma_trend,
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'stoch_k': stoch_k,
        'stoch_d': stoch_d,
        'bb_upper': bb_upper,
        'bb_middle': bb_middle,
        'bb_lower': bb_lower,
        'atr': atr,
        'swing_highs': swing_highs,
        'swing_lows': swing_lows,
    }
    
    return indicators


def generate_signal(
    symbol: str,
    candles: List[Dict],
    params: StrategyParams,
    signal_time: Optional[datetime] = None
) -> Optional[Signal]:
    """
    Generate trading signal based on confluence scoring.
    
    STRATEGY:
    1. Determine direction from MA alignment
    2. Compute 5-pillar confluence score
    3. Check regime (trend vs range)
    4. Generate signal if confluence >= threshold
    
    Args:
        symbol: Trading symbol
        candles: Historical candles (most recent last)
        params: Strategy parameters
        signal_time: Time of signal (default: now)
    
    Returns:
        Signal object or None if no valid setup
    """
    if signal_time is None:
        signal_time = datetime.utcnow()
    
    # Compute indicators
    ind = compute_indicators(candles, params)
    if not ind:
        return None
    
    # Get latest values
    close = ind['close'][-1]
    sma_fast = ind['sma_fast'][-1]
    sma_slow = ind['sma_slow'][-1]
    sma_trend = ind['sma_trend'][-1]
    atr = ind['atr'][-1]
    
    # Check for valid indicator values
    if np.isnan(sma_fast) or np.isnan(sma_slow) or np.isnan(atr):
        return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 1: DETERMINE DIRECTION FROM MA ALIGNMENT
    # ═══════════════════════════════════════════════════════════════════════
    
    direction = None
    
    # Strong bullish: price above both MAs, fast > slow
    if close > sma_fast and sma_fast > sma_slow:
        direction = 'bullish'
    # Strong bearish: price below both MAs, fast < slow
    elif close < sma_fast and sma_fast < sma_slow:
        direction = 'bearish'
    else:
        return None  # No clear direction
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 2: DETECT REGIME (if enabled)
    # ═══════════════════════════════════════════════════════════════════════
    
    regime = 'trend'
    if params.use_regime_filter:
        regime = detect_regime(
            ind['close'], ind['atr'], ind['sma_fast'], ind['sma_slow'],
            params.range_atr_threshold
        )
        
        # In range mode, require stronger confirmation
        # (This makes the strategy more selective in choppy markets)
        if regime == 'range':
            # Skip if price is in middle of range (wait for extreme)
            bb_middle = ind['bb_middle'][-1]
            bb_range = ind['bb_upper'][-1] - ind['bb_lower'][-1]
            
            if not np.isnan(bb_range) and bb_range > 0:
                price_position = (close - ind['bb_lower'][-1]) / bb_range
                
                # In range: only trade at extremes (< 0.25 or > 0.75)
                if 0.25 < price_position < 0.75:
                    return None  # Not at extreme
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 3: COMPUTE CONFLUENCE SCORE
    # ═══════════════════════════════════════════════════════════════════════
    
    confluence, reasons = compute_confluence(direction, close, ind, params)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 4: CHECK MINIMUM CONFLUENCE
    # ═══════════════════════════════════════════════════════════════════════
    
    min_required = params.min_confluence
    
    # Require higher confluence in range mode
    if regime == 'range':
        min_required = max(min_required, 4)  # At least 4 in range
    
    if confluence < min_required:
        return None
    
    # ═══════════════════════════════════════════════════════════════════════
    # STEP 5: CALCULATE ENTRY, STOP LOSS, TAKE PROFITS
    # ═══════════════════════════════════════════════════════════════════════
    
    entry = close
    risk = atr * params.atr_sl_multiplier
    
    if direction == 'bullish':
        stop_loss = entry - risk
        tp1 = entry + risk * params.tp1_r_multiple
        tp2 = entry + risk * params.tp2_r_multiple
        tp3 = entry + risk * params.tp3_r_multiple
    else:
        stop_loss = entry + risk
        tp1 = entry - risk * params.tp1_r_multiple
        tp2 = entry - risk * params.tp2_r_multiple
        tp3 = entry - risk * params.tp3_r_multiple
    
    return Signal(
        symbol=symbol,
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        confluence=confluence,
        signal_time=signal_time,
        regime=regime,
        reason=" | ".join(reasons)
    )


def get_default_params() -> StrategyParams:
    """Get default strategy parameters."""
    return StrategyParams()


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def validate_signal(signal: Signal, params: StrategyParams) -> bool:
    """
    Validate a signal before execution.
    
    Returns True if signal passes all validation checks.
    """
    if signal is None:
        return False
    
    # Check direction
    if signal.direction not in ('bullish', 'bearish'):
        return False
    
    # Check risk/reward
    if signal.risk_reward_ratio < 1.0:
        return False
    
    # Check confluence
    if signal.confluence < params.min_confluence:
        return False
    
    return True


def calculate_lot_size(
    balance: float,
    risk_pct: float,
    entry: float,
    stop_loss: float,
    pip_value: float = 10.0,
    pip_size: float = 0.0001
) -> float:
    """
    Calculate position size based on risk.
    
    Args:
        balance: Account balance
        risk_pct: Risk percentage (0.5 = 0.5%)
        entry: Entry price
        stop_loss: Stop loss price
        pip_value: Value per pip per lot (default $10 for forex)
        pip_size: Size of 1 pip (0.0001 for most forex)
    
    Returns:
        Lot size (rounded to 0.01)
    """
    risk_amount = balance * (risk_pct / 100)
    sl_pips = abs(entry - stop_loss) / pip_size
    
    if sl_pips <= 0:
        return 0.01
    
    lot_size = risk_amount / (sl_pips * pip_value)
    return max(0.01, round(lot_size, 2))


# ═══════════════════════════════════════════════════════════════════════════
# TEST / DEMO
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("🎯 Confluence Strategy Template - Demo")
    print("=" * 60)
    
    # Create sample candles with trending data
    import random
    random.seed(42)
    
    base_price = 1.1000
    candles = []
    
    # Generate 200 candles with slight uptrend
    for i in range(200):
        trend = 0.00005 * i  # Slight uptrend
        noise = random.uniform(-0.0030, 0.0030)
        
        open_price = base_price + trend + noise
        high = open_price + random.uniform(0.0010, 0.0040)
        low = open_price - random.uniform(0.0010, 0.0040)
        close = random.uniform(low + 0.0005, high - 0.0005)
        base_price = close
        
        candles.append({
            'time': datetime(2024, 1, 1, i % 24),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close
        })
    
    # Generate signal
    params = get_default_params()
    signal = generate_signal("EURUSD", candles, params)
    
    if signal:
        print(f"\n✅ Signal Generated:")
        print(f"   Symbol: {signal.symbol}")
        print(f"   Direction: {signal.direction.upper()}")
        print(f"   Regime: {signal.regime}")
        print(f"   Entry: {signal.entry:.5f}")
        print(f"   Stop Loss: {signal.stop_loss:.5f}")
        print(f"   TP1: {signal.tp1:.5f} ({params.tp1_r_multiple}R)")
        print(f"   TP2: {signal.tp2:.5f} ({params.tp2_r_multiple}R)")
        print(f"   TP3: {signal.tp3:.5f} ({params.tp3_r_multiple}R)")
        print(f"   Confluence: {signal.confluence}/5")
        print(f"   Risk/Reward: {signal.risk_reward_ratio:.2f}")
        print(f"\n   Reasons:")
        for reason in signal.reason.split(" | "):
            print(f"   • {reason}")
    else:
        print("\n❌ No signal generated (confluence not met)")
    
    print("\n" + "=" * 60)
    print("📋 CONFLUENCE PILLARS:")
    print("   1. 📈 Trend Bias (MA alignment)")
    print("   2. 🚀 Momentum (MACD + Stochastic)")
    print("   3. 📍 Volatility (Bollinger Bands)")
    print("   4. 🏗️ Structure (Swing levels)")
    print("   5. ✅ Price Action confirmation")
    print("\n   Minimum required: 3/5 (configurable)")
    print("=" * 60)
