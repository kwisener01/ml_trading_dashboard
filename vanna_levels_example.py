"""
Example script showing how to calculate and use Vanna-based support/resistance levels.

Vanna levels show where options positioning creates natural support/resistance:
- Support: Where market makers hedging will buy on dips (positive gamma/vanna)
- Resistance: Where market makers will sell on rallies (negative gamma/vanna)
"""

import pandas as pd
import numpy as np
from second_order_greeks import OptionsFeatureEngineering

# Create sample price data for SPY
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=100, freq='D')
prices = 450 + np.cumsum(np.random.randn(100) * 2)  # Random walk around 450

df = pd.DataFrame({
    'date': dates,
    'close': prices,
    'high': prices + np.random.rand(100) * 3,
    'low': prices - np.random.rand(100) * 3,
    'open': prices + np.random.randn(100),
    'volume': np.random.randint(1000000, 5000000, 100)
})

print("="*80)
print("VANNA SUPPORT & RESISTANCE LEVELS")
print("="*80)
print(f"\nCurrent Price: ${df['close'].iloc[-1]:.2f}")

# Initialize Options Feature Engineering
ofe = OptionsFeatureEngineering(df)

# Add second-order Greeks (required for Vanna calculation)
ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)

# Add Vanna-based support/resistance levels
ofe.add_vanna_support_resistance(strike_width=5.0, vanna_threshold=0.001)

# Display the levels
print("\n" + "="*80)
print("VANNA SUPPORT LEVELS (below current price)")
print("="*80)

support_cols = [col for col in ofe.df.columns if 'vanna_support' in col and 'strength' not in col]
for col in sorted(support_cols):
    if col in ofe.df.columns and not pd.isna(ofe.df[col].iloc[-1]):
        level = ofe.df[col].iloc[-1]
        strength_col = f"{col}_strength"
        strength = ofe.df[strength_col].iloc[-1] if strength_col in ofe.df.columns else 0
        distance = ((level - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
        print(f"{col.replace('vanna_', '').replace('_', ' ').title():20} ${level:7.2f} ({distance:+.2f}%) | Strength: {strength:.4f}")

print("\n" + "="*80)
print("VANNA RESISTANCE LEVELS (above current price)")
print("="*80)

resistance_cols = [col for col in ofe.df.columns if 'vanna_resistance' in col and 'strength' not in col]
for col in sorted(resistance_cols):
    if col in ofe.df.columns and not pd.isna(ofe.df[col].iloc[-1]):
        level = ofe.df[col].iloc[-1]
        strength_col = f"{col}_strength"
        strength = ofe.df[strength_col].iloc[-1] if strength_col in ofe.df.columns else 0
        distance = ((level - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
        print(f"{col.replace('vanna_', '').replace('_', ' ').title():20} ${level:7.2f} ({distance:+.2f}%) | Strength: {strength:.4f}")

print("\n" + "="*80)
print("TRADING INTERPRETATION")
print("="*80)
print("""
How to use Vanna levels:

1. SUPPORT LEVELS (Below Price):
   - Strong Vanna support = dealers will hedge by BUYING as price falls
   - Price likely to bounce at these levels
   - Use for entries on pullbacks

2. RESISTANCE LEVELS (Above Price):
   - Strong Vanna resistance = dealers will hedge by SELLING as price rises
   - Price likely to stall at these levels
   - Use for profit taking or shorts

3. VANNA STRENGTH:
   - Higher strength = more dealer hedging activity expected
   - Strongest levels are most likely to hold
   - Watch for breaks - can signal major moves

4. VOLATILITY IMPACT:
   - Rising IV strengthens these levels (more hedging needed)
   - Falling IV weakens these levels (less hedging needed)
   - VIX spikes can create strong Vanna walls
""")

print("\n" + "="*80)
print("INTEGRATION WITH YOUR TRADING SYSTEM")
print("="*80)
print("""
To add this to your ML pipeline:

1. In feature_engineering.py, add:
   from second_order_greeks import OptionsFeatureEngineering
   ofe = OptionsFeatureEngineering(df)
   ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)
   ofe.add_vanna_support_resistance()

2. New features available for ML models:
   - vanna_support_1, vanna_support_2, vanna_support_3
   - vanna_resistance_1, vanna_resistance_2, vanna_resistance_3
   - vanna_support_1_strength, etc.

3. Use in trading decisions:
   - Distance to nearest Vanna support/resistance
   - Strength of nearest levels
   - Number of levels nearby (congestion)
""")
