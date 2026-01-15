# 🤖 Trading Bot Template - 5ERS Challenge Ready

**A customizable trading bot template for Forex optimization and live trading.**

> ⚠️ **TEMPLATE**: Dit is een educatieve template met een eenvoudige EMA/RSI/ADX strategie.
> Je wordt aangemoedigd om je eigen strategie te ontwikkelen en de parameters te optimaliseren.

---

## 🚀 Quick Start (Codespaces)

### 1. Open in GitHub Codespaces
Klik op de groene "Code" knop → "Codespaces" → "Create codespace on main"

### 2. Installeer dependencies
```bash
pip install -r requirements.txt
```

### 3. Run TPE Optimization
```bash
python optimizer.py --trials 50 --balance 100000 --start 2023-01-01 --end 2025-12-31
```

### 4. Valideer met H1 Simulator
```bash
python simulator.py --params optimization_output/best_params.json
```

---

## 📁 Project Structuur

```
├── strategy_template.py    # 🎯 Jouw trading strategie (EMA/RSI/ADX)
├── optimizer.py            # 🔧 TPE optimizer (Optuna) + 5ERS risk rules
├── simulator.py            # 📊 H1 trade simulator + DDD tracking
├── live_bot.py             # 🤖 Live trading bot (MT5) + prop firm safety
├── data/ohlcv/             # 📈 Historische data (D1/H1)
├── optimization_output/    # 💾 Optimizer resultaten
└── simulation_output/      # 📊 Simulator resultaten
```

---

## 🎯 De Template Strategie

De meegeleverde strategie gebruikt een **5-Pillar Confluence System**:

### Confluence Pillars:

| # | Pillar | Bullish | Bearish |
|---|--------|---------|---------|
| 1 | 📈 **Trend Bias** | Price > SMA20 > SMA50 | Price < SMA20 < SMA50 |
| 2 | 🚀 **Momentum** | MACD + Stoch bullish | MACD + Stoch bearish |
| 3 | 📍 **Volatility** | Price in lower BB zone | Price in upper BB zone |
| 4 | 🏗️ **Structure** | Above swing low | Below swing high |
| 5 | ✅ **Price Action** | Higher close | Lower close |

**Minimum required: 3/5 confluence pillars**

### Parameters om te optimaliseren:
- `sma_fast_period`: 10-30
- `sma_slow_period`: 40-80
- `macd_fast/slow/signal`: 8-15, 20-30, 7-12
- `stoch_k/d_period`: 10-20, 3-5
- `bb_period/std_dev`: 15-25, 1.5-2.5
- `min_confluence`: 2-4

### Regime Detection:
De strategie past zich aan op marktcondities:
- **Trend Mode**: Normale confluence vereist (3/5)
- **Range Mode**: Hogere confluence vereist (4/5) + extremen

---

## 📊 Workflow

### Stap 1: Optimalisatie (TPE)
```bash
python optimizer.py --trials 100
```

Dit vindt de beste parameters door duizenden combinaties te testen.

**Output:**
- `optimization_output/best_params.json` - Beste parameters
- `optimization_output/optimization_history.csv` - Alle trials
- `optimization_output/optimization_results.json` - Samenvatting

### Stap 2: H1 Validatie
```bash
python simulator.py --params optimization_output/best_params.json
```

Simuleert trades op uurlijkse data voor realistische resultaten.

**Output:**
- `simulation_output/simulation_results.json` - Eindresultaten
- `simulation_output/trades.csv` - Alle trades
- `simulation_output/daily_snapshots.csv` - Dagelijkse balans

### Stap 3: Live Trading
```bash
python live_bot.py
```

Draait de bot live met MetaTrader 5 (alleen op Windows).

---

## 💰 5ERS Prop Firm Regels

De bot implementeert **strikte prop firm regels** om challenges te beschermen:

| Regel | Limiet | Actie |
|-------|--------|-------|
| Daily DD Warning | 2.0% | ⚠️ Alert |
| Daily DD Reduce | **3.2%** | **TRADE HALF** (50% position size) |
| Daily DD Halt | 4.0% | ⛔ Stop trading voor vandaag |
| Daily DD Limit | **5.0%** | 🚨 HARD LIMIT - Challenge failed |
| Total DD Warning | 5.0% | ⚠️ Alert |
| Total DD Reduce | 7.0% | Reduce to minimum risk |
| Total DD Limit | **10.0%** | 🚨 HARD LIMIT - Challenge failed |

### Kritieke Regels:

1. **TRADE HALF @ 3.2%**: Wanneer daily DD 3.2% bereikt, worden alle nieuwe trades met 50% positiegrootte geplaatst
2. **HALT @ 4%**: Geen nieuwe trades meer voor die dag
3. **Total DD is STATIC**: Berekend vanaf startbalans ($100K), niet trailing!

### Risk Management Flow:
```
0% DD → Normale risk (100%)
   ↓
2% DD → Warning (75% risk)
   ↓  
3.2% DD → TRADE HALF (50% risk)
   ↓
4% DD → HALT (0% - stop trading today)
   ↓
5% DD → CHALLENGE FAILED
```

---

## 🔧 Strategie Aanpassen

### 1. Open `strategy_template.py`

### 2. Pas `generate_signal()` aan:
```python
def generate_signal(symbol, candles, params, signal_time=None):
    # Jouw eigen logica hier
    # Voorbeeld: voeg MACD toe, support/resistance, etc.
    
    # Return Signal object of None
    return Signal(
        symbol=symbol,
        direction='bullish',  # of 'bearish'
        entry=entry_price,
        stop_loss=sl_price,
        tp1=tp1_price,
        tp2=tp2_price,
        tp3=tp3_price,
        confluence=score,
        signal_time=signal_time
    )
```

### 3. Voeg parameters toe aan `StrategyParams`:
```python
@dataclass
class StrategyParams:
    # Bestaande params...
    
    # Jouw nieuwe params:
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
```

### 4. Update `optimizer.py` suggest_params():
```python
def suggest_params(trial):
    return StrategyParams(
        # Bestaande...
        macd_fast=trial.suggest_int('macd_fast', 8, 15),
        macd_slow=trial.suggest_int('macd_slow', 20, 30),
        macd_signal=trial.suggest_int('macd_signal', 5, 12),
    )
```

---

## 📈 Performance & 5ERS Compliance

De meegeleverde EMA(20,50) strategie is een **template** - je moet zelf betere parameters vinden!

### Belangrijke Note:
De optimizer stopt automatisch bij DD overschrijding:
- Trading wordt geHALT bij 4% daily DD
- Challenge wordt gestopt bij 5% daily of 10% total DD

### Om 5ERS te passeren:
1. Optimaliseer met conservatieve risk (0.3-0.5%)
2. Valideer met H1 simulator
3. Check dat Max Daily DD **onder 5%** blijft
4. Check dat Max Total DD **onder 10%** blijft (static)

> 💡 **Tip**: De "trade half at 3.2%" regel beschermt je account.
> Als je vaak geHALT wordt, verlaag je base risk.

---

## ⚠️ Belangrijke Waarschuwingen

1. **Backtests ≠ Live resultaten** - Altijd forward testen!
2. **Geen financieel advies** - Alleen voor educatieve doeleinden
3. **Risico management** - Gebruik altijd stop losses
4. **Overfitting** - Pas op voor te veel parameters

---

## 🛠️ Vereisten

- Python 3.10+
- Optuna
- Pandas
- NumPy
- tqdm
- MetaTrader 5 (alleen voor live trading)

```bash
pip install optuna pandas numpy tqdm
```

---

## 📝 Commands Overzicht

```bash
# Optimalisatie
python optimizer.py --trials 100 --balance 100000

# Status bekijken
python optimizer.py --status

# H1 Simulatie
python simulator.py --params optimization_output/best_params.json

# Met custom instellingen
python simulator.py --balance 100000 --start 2023-01-01 --end 2025-12-31
```

---

## 🤝 Bijdragen

Voel je vrij om:
- Je eigen strategie te ontwikkelen
- Bugs te rapporteren
- Verbeteringen voor te stellen

---

## 📜 Licentie

MIT License - Gebruik op eigen risico.

---

**Happy Trading! 🚀**
