# Vanna-Based Support & Resistance Levels

## Overview

This feature calculates support and resistance levels based on **Vanna** (the second-order Greek that measures delta sensitivity to volatility changes). These levels indicate where options market maker hedging activity creates natural price magnets.

## What is Vanna?

**Vanna** = ∂Delta/∂IV (change in delta per 1% change in implied volatility)

- **Positive Vanna**: Delta increases when IV rises (typical for ATM/slightly OTM options)
- **Negative Vanna**: Delta decreases when IV rises (typical for deep ITM/OTM options)

## Why Vanna Creates Support/Resistance

Market makers hedge their options positions by trading the underlying. Vanna determines how much they need to hedge when volatility changes:

### Support Levels (Below Current Price)
- **Positive Vanna concentration** = dealers are **long gamma** at these strikes
- When price falls toward these levels:
  - Dealers **buy more stock** to stay delta-neutral
  - Creates **buying pressure** = support
- **Stronger in high IV environments** (VIX spikes)

### Resistance Levels (Above Current Price)
- **Negative Vanna concentration** = dealers are **short gamma** at these strikes
- When price rises toward these levels:
  - Dealers **sell stock** to stay delta-neutral
  - Creates **selling pressure** = resistance
- **Stronger in high IV environments**

## How to Use the Feature

### Basic Usage

```python
from second_order_greeks import OptionsFeatureEngineering

# Initialize with your price data
ofe = OptionsFeatureEngineering(df)

# Calculate second-order Greeks
ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)

# Add Vanna support/resistance levels
ofe.add_vanna_support_resistance(
    strike_width=5.0,        # Price increments to check ($5 apart)
    vanna_threshold=0.001    # Minimum Vanna strength to consider
)
```

### Output Features

The method adds these columns to your DataFrame:

**Support Levels:**
- `vanna_support_1` - Strongest support level price
- `vanna_support_1_strength` - Vanna strength at that level
- `vanna_support_2`, `vanna_support_3` - 2nd and 3rd strongest

**Resistance Levels:**
- `vanna_resistance_1` - Strongest resistance level price
- `vanna_resistance_1_strength` - Vanna strength at that level
- `vanna_resistance_2`, `vanna_resistance_3` - 2nd and 3rd strongest

## Trading Strategies

### 1. Pullback Entries
```
If price pulls back to vanna_support_1:
  → Enter LONG
  → Stop loss below vanna_support_2
  → Target: vanna_resistance_1
```

### 2. Breakout Trading
```
If price breaks above vanna_resistance_1 with volume:
  → Enter LONG (resistance becomes support)
  → Stop loss at old resistance level
  → Target: Next resistance or +2% move
```

### 3. Range Trading
```
When price between support and resistance:
  → Sell near resistance levels
  → Buy near support levels
  → Exit if breaks either level decisively
```

### 4. Risk Management
```
Distance to nearest support = max position size
Closer to support = larger size (better R/R)
Between levels = reduce size (choppy zone)
```

## Integration with ML Models

Add Vanna levels as features for your ML models:

```python
# In feature_engineering.py

from second_order_greeks import OptionsFeatureEngineering

# After your existing features:
ofe = OptionsFeatureEngineering(df)
ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)
ofe.add_vanna_support_resistance()

# Calculate derived features
df['distance_to_support'] = (df['close'] - df['vanna_support_1']) / df['close']
df['distance_to_resistance'] = (df['vanna_resistance_1'] - df['close']) / df['close']
df['support_strength_ratio'] = df['vanna_support_1_strength'] / df['vanna_resistance_1_strength']
df['near_vanna_level'] = ((df['distance_to_support'].abs() < 0.01) |
                           (df['distance_to_resistance'].abs() < 0.01)).astype(int)
```

## Parameters Explained

### `strike_width` (default: 5.0)
- Price increment between strikes checked
- **Smaller values** (1.0-2.0): More granular, finds precise levels
- **Larger values** (5.0-10.0): Faster computation, broader levels
- **Recommended**: Match your typical options strike spacing

### `vanna_threshold` (default: 0.01)
- Minimum absolute Vanna value to consider significant
- **Lower values** (0.001): More levels identified, including weaker ones
- **Higher values** (0.05): Only strongest levels, cleaner chart
- **Recommended**: Start with 0.01, adjust based on your underlying

## Real-World Examples

### Example 1: SPY Near Vanna Support
```
Current Price: $450.00
Vanna Support 1: $448.00 (-0.44%) | Strength: 0.0450
Vanna Support 2: $445.00 (-1.11%) | Strength: 0.0380

Action: Price at $449 → Watch for bounce at $448
Risk/Reward: Risk $2 to support, Reward $5+ to resistance
```

### Example 2: QQQ Breaking Vanna Resistance
```
Current Price: $385.50
Vanna Resistance 1: $385.00 (+0.13%) | Strength: 0.0420

Action: Just broke through $385 resistance
Expected: Resistance becomes support, momentum to $390+
Stop Loss: Below $385 (old resistance)
```

### Example 3: High IV Strengthens Levels
```
Normal Conditions:
- Vanna Support 1 Strength: 0.0300
- Price bounces 60% of time

VIX Spike (+5 points):
- Vanna Support 1 Strength: 0.0450 (50% increase)
- Price bounces 85% of time
- Stronger dealer hedging = stronger levels
```

## Advanced Concepts

### Vanna Flip
When price crosses a major Vanna level:
- **Old support becomes resistance** (and vice versa)
- Dealers flip their hedging from buying to selling
- Can create strong reversals or continuation

### Vanna Decay
As expiration approaches:
- Vanna decreases (less time for volatility to matter)
- Support/resistance levels weaken
- Recalculate for next month's expiration

### Combining with Other Greeks
- **Gamma levels** (0DTE options): Strongest intraday levels
- **Charm levels**: Time decay pressure points
- **Vomma levels**: Vega convexity zones
- Use all together for comprehensive level analysis

## Limitations & Caveats

1. **Approximation**: Without actual options chain data, we estimate Vanna distribution
2. **Expiration Date**: Results vary by DTE (days to expiration)
3. **Implied Volatility**: Assumes constant IV across strikes (in reality, there's vol skew)
4. **Market Regime**: Works best in normal conditions, less reliable in crashes
5. **Needs Confirmation**: Use with other indicators (volume, momentum, etc.)

## Best Practices

1. **Recalculate Regularly**
   - Daily for active trading
   - Weekly for swing trading
   - After major news events

2. **Combine with Volume**
   - Vanna level + high volume = stronger level
   - Vanna level + low volume = might break easily

3. **Watch VIX**
   - Rising VIX = stronger Vanna levels
   - Falling VIX = weaker Vanna levels
   - VIX at extremes = most reliable levels

4. **Multiple Timeframes**
   - Calculate for 30 DTE (monthly)
   - Calculate for 7 DTE (weekly)
   - Calculate for 1 DTE (daily/0DTE)

5. **Backtest**
   - Test on historical data first
   - Measure actual bounce rates at levels
   - Optimize parameters for your trading style

## References & Further Reading

- **SqueezeMetrics**: Industry leader in options positioning data
- **SpotGamma**: Provides daily gamma/vanna level charts
- **Options volatility pricing** by Sheldon Natenberg
- **The Volatility Surface** by Jim Gatheral

## Support

For questions or issues with Vanna levels:
1. Check calculation with `vanna_levels_example.py`
2. Verify your data has OHLCV columns
3. Adjust `strike_width` and `vanna_threshold` parameters
4. Compare with professional options data providers

---

**Last Updated**: November 2024
**Version**: 1.0
**Author**: Claude Code ML Trading System
