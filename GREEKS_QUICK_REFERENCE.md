# 🎯 SECOND-ORDER GREEKS - QUICK REFERENCE CARD

**Print this. Keep it at your desk. Use it every day.**

---

## 📋 THE GREEKS CHEAT SHEET

### 🔴 VANNA (Delta × Volatility)
**∂Delta/∂σ**

| When | Value | Means | Action |
|------|-------|-------|--------|
| OTM Call | Negative | Delta ↓ when IV ↑ | Buy when IV low |
| ITM Call | Positive | Delta ↑ when IV ↑ | Sell when IV low |
| Max | ATM | Highest impact | Best for trades |

**Trading Rule:**
```
BUY calls when:
  • IV < 30th percentile (CHEAP)
  • Strike = OTM (high negative Vanna)
  • Breakout setup (vol will spike)
  
RESULT: Negative Vanna = More delta as IV rises = Bigger profits
```

---

### 🔵 CHARM (Delta × Time)
**∂Delta/∂t**

| DTE | Charm Threshold | Action | Risk |
|-----|-----------------|--------|------|
| 45+ | High | Hold | Low |
| 30 | Medium | Monitor | Low |
| 21 | 0.05+ | Consider exit | Medium |
| 14 | 0.02+ | Probably exit | High |
| 7 | Any + | Must exit | Critical |
| 3 | Extreme | FORCED exit | Extreme |

**Trading Rule:**
```
EXIT when:
  • DTE < 21 AND Charm > 0.05
  • DTE < 14 AND Charm > 0.02
  • DTE < 7 (always exit)
  
REASON: Delta becomes unstable (gamma risk)
```

---

### 🟢 VOMMA (Vega × Volatility)
**∂Vega/∂σ**

| Vomma Level | Meaning | Action |
|-------------|---------|--------|
| > 0.5 | Very high | Buy vol plays |
| 0.2-0.5 | High | Consider vol plays |
| < 0 | Low/Negative | Avoid vol plays |

**Trading Rule:**
```
BUY STRADDLES when:
  • High Vomma (> 0.5)
  • Low IV (cheap premium)
  • Wait for vol expansion
  
RESULT: Vomma amplifies vega = Bigger vol profits
```

---

## 🚀 DAILY CHECKLIST

### ✅ MORNING (Before Market Open)

```
[ ] Check VIX and IV Percentile
    Low (< 20)? → Good for buying calls
    High (> 80)? → Good for selling

[ ] Calculate Vanna for potential entries
    Negative Vanna (OTM calls)? → Good setup
    
[ ] Scan for Charm warnings
    Charm > 0.10? → Monitor for exits
    
[ ] Identify technical setups
    Breakout ready? → Use with Vanna
    
[ ] Size positions based on Greeks
    High Greeks = Smaller position
```

### ⚡ DURING MARKET

```
[ ] Monitor open positions Charm daily
[ ] Exit if Charm exceeds thresholds
[ ] Track if Vanna is amplifying moves
[ ] Look for Vomma opportunities in vol spikes
```

### 📊 AFTER MARKET

```
[ ] Record Greeks values in journal
[ ] Update exit prices for Charm signals
[ ] Review if predictions were accurate
[ ] Plan tomorrow's Greeks-based setups
```

---

## 🎯 QUICK DECISION TREE

```
ENTRY DECISION:

        START
          ↓
    [Check IV Percentile]
          ↓
    ┌─────┴──────┐
    ↓            ↓
  LOW IV      HIGH IV
    ↓            ↓
 CALCULATE   SKIP
  GREEKS     OR SELL
    ↓
[Check Vanna]
    ↓
 ┌──────┬──────┐
 ↓      ↓      ↓
NEG    ZERO   POS
↓      ↓      ↓
BUY   SKIP   SELL
CALL  OR     PUT
      SKIP

EXIT DECISION:

    [Check DTE]
    ↓
 ┌──────┬──────┬──────┐
 ↓      ↓      ↓      ↓
<7   7-14  14-21  >21
 ↓      ↓      ↓      ↓
EXIT  CHECK  CHECK HOLD
      CHARM  CHARM MONITOR
      ↓      ↓
    EXIT IF  EXIT IF
    >0.02    >0.05
```

---

## 💰 PROFIT SCENARIOS

### Scenario 1: "Vanna Wins"
```
Entry:  Buy call (OTM, IV 15%, Vanna -0.05)
Stock:  +$5 move
IV:     Rises to 25% (vol spike)

Without Vanna awareness:
  Delta 0.50 × $5 = $2.50 profit

With Vanna optimization:
  Delta increases to 0.58 (Vanna effect!)
  0.58 × $5 = $2.90 profit
  
EXTRA: +$0.40 = +16% MORE PROFIT!
```

### Scenario 2: "Charm Saves"
```
Position: 12 DTE call
Charm: 0.08 (accelerating decay)

Choice 1 (ignoring Charm):
  Hold to 3 DTE
  Gamma crush = -60% loss
  
Choice 2 (respecting Charm):
  Exit at 14 DTE when Charm warning
  Keep 40% of profit
  
RESULT: +40% profit vs -60% loss = 100% swing!
```

### Scenario 3: "Vomma Explosion"
```
Entry:  Buy straddle (High Vomma, IV 20%)
Event:  Earnings announcement
Result: IV spikes to 50%

Vega profit: $5,000
Vomma amplification: +$3,000 (convexity)
Total: +$8,000

RESULT: Vomma added +60% extra profit!
```

---

## 🚨 DANGER ZONES

| Condition | Risk | Action |
|-----------|------|--------|
| Charm > 0.20 | EXTREME | EXIT NOW |
| Charm 0.10-0.20 | HIGH | Monitor close |
| Charm 0.05-0.10 | MEDIUM | Plan exit |
| DTE < 7 | GAMMA | Exit or roll |
| IV spiked >50% | REALIZED | Reduce size |
| Negative Vomma > -1 | VOL CRUSH | Close shorts |

---

## 📊 GREEK VALUES AT A GLANCE

### Call Options (Stock $450, Strike $455, 30 DTE, IV 25%)

| Greek | Value | Interpretation |
|-------|-------|-----------------|
| Vanna | 0.211 | ITM - Will lose delta if vol rises |
| Charm | -1.398 | Fast decay - High risk |
| Vomma | 1.663 | High convexity - Good for vol |

### Ideal Entry (Stock $450, Strike $455, 30 DTE, IV 15%)

| Greek | Value | Interpretation |
|-------|-------|-----------------|
| Vanna | -0.492 | OTM - Will GAIN delta if vol rises ✓ |
| Charm | 2.354 | Stable - Good to hold |
| Vomma | 25.828 | Very high convexity ✓ |

---

## 💡 PROFESSIONAL TIPS

1. **Recalculate Daily** - Greeks change every day
2. **Use Percentiles** - Compare Greeks to historical range
3. **Combine Signals** - Never trade on one Greek alone
4. **Journal Trades** - Record Greeks at entry/exit
5. **Backtest Thresholds** - Find YOUR optimal levels
6. **Scale Positions** - Bigger Greeks = Smaller position
7. **Stress Test** - Model 10%, 20%, 30% IV moves

---

## 🎓 REAL EXAMPLES

### Trade 1: Bull Breakout with Vanna
```
Setup:     Stock consolidating, breakout ready
Entry:     Buy OTM call when IV = 15%
Vanna:     -0.05 (OTM, good signal)
Charm:     -0.10 (stable decay)

Breakout:  Stock +$10, IV rises to 25%
Vanna win: Delta 0.50 → 0.56 (Vanna helped)
Profit:    $10 × 0.56 = $5.60 per contract
Result:    ✓ +25-40% return

Key:       Bought low IV with negative Vanna
```

### Trade 2: Exiting Before Gamma Trap
```
Entry:     30 DTE call, Charm -0.09
5 Days:    Charm still okay (-0.07)
14 Days:   Charm warning! (-0.08, rising)
Action:    Close position for 45% profit
Outcome:   ✓ Avoid gamma risk

vs Hold:   Would have held to 7 DTE
Charm:     +0.20 (extreme)
Stock:     Suddenly down $8
Loss:      -60% profit evaporated
Result:    ✗ Would have been crushed

Lesson:    Exit on Charm warning
```

### Trade 3: Vomma Vol Play
```
Setup:     High Vomma straddle (0.50+)
Wait for:  Catalyst (earnings, Fed)
IV move:   +30 VIX points
Vega:      $2.00 × 30 = $60 profit
Vomma:     Adds $36 (convexity)
Total:     $96 profit
Result:    ✓ +40% return in 1 day

Key:       High Vomma = High vol convexity
```

---

## 🔄 WEEKLY OPTIMIZATION

| Day | Focus |
|-----|-------|
| Mon | Review weekend analysis, identify setups |
| Tue | Execute high-probability Vanna setups |
| Wed | Monitor Charm on existing positions |
| Thu | Consider exiting before DTE decay |
| Fri | Prepare for weekend, reduce risk |

---

## 📱 COPY-PASTE PYTHON

```python
from second_order_greeks import SecondOrderGreeks

greek = SecondOrderGreeks()

# Quick calculation
greeks = greek.calculate_all_second_order(
    S=450, K=455, T=30/365, 
    r=0.05, sigma=0.25, option_type='call'
)

print(f"Vanna: {greeks['vanna']:.6f}")
print(f"Charm: {greeks['charm']:.6f}")
print(f"Vomma: {greeks['vomma']:.6f}")
```

---

## 🎯 BOTTOM LINE

| Greek | Best For | Key Action |
|-------|----------|------------|
| **Vanna** | Entry timing | Buy OTM when IV low |
| **Charm** | Exit management | Exit when > 0.10 |
| **Vomma** | Vol profiting | Hold through vol spikes |

**Master these three Greeks and you trade like a professional.** 🚀

---

**Keep this handy. Reference it daily. Make more money.**

**Print. Laminate. Keep at desk. 📋**

---

*Questions? Review SECOND_ORDER_GREEKS_GUIDE.md for full details.*

*Need examples? Run greeks_integration_examples.py for live demonstrations.*
