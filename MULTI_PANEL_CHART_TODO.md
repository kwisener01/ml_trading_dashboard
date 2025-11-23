# Multi-Panel Chart Implementation TODO

## ✅ Phase 1 - COMPLETED
- ✅ Enhanced predictor with IV, Charm, and dealer flow calculations
- ✅ Added plotly subplots import to dashboard
- ✅ Committed initial enhancements

## ✅ Phase 2 - COMPLETED
- ✅ Created VWAP calculation helper
- ✅ Built multi-panel chart function (create_options_flow_chart)
- ✅ Implemented Panel 1: Price with all options flow levels
- ✅ Implemented Panel 2: IV & Vanna indicators
- ✅ Implemented Panel 3: Dealer flow indicators
- ✅ Added background color shading for GEX regimes
- ✅ Replaced old chart with new 3-panel layout
- ✅ Updated chart legend with comprehensive guide
- ✅ Tested and committed changes
- ✅ Pushed to GitHub

## Implementation Complete! 🎉

The dashboard now features:
1. **Comprehensive price analysis** with Gamma Flip, GEX walls, Vanna walls, Put/Call walls, and VWAP
2. **IV & Vanna panel** showing implied volatility dynamics and dealer bias
3. **Dealer flow panel** with charm pressure, GEX pressure, and combined score
4. **Visual regime indicators** with color-coded backgrounds
5. **Clean, professional UI** with 900px height and dark theme

## Original Requirements (from Phase 2 Planning)

### 1. Dashboard Chart Restructuring
Location: `dashboard.py` lines ~600-1000

Replace single chart with 3-panel subplot layout:

```python
fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.05,
    row_heights=[0.6, 0.2, 0.2],
    subplot_titles=("Price with Options Flow Levels", "IV & Vanna Panel", "Dealer Flow Panel")
)
```

### 2. Panel 1: Enhanced Price Chart
**Keep existing:**
- Candlesticks
- Current price line (blue)
- Bounce/Rejection zones (green/red shading)

**Add new:**
- Gamma Flip line (label as "major pivot")
- GEX walls (label as "S/R zones")
- Vanna walls (label as "pressure zones")
- Put/Call walls (label as "magnets/barriers") - if available
- VWAP line (calculate from recent intraday data)
- Color shading for dealer flow regime (green for positive, red for negative)

### 3. Panel 2: IV & Vanna Indicators
Add 3 traces:
- **IV (Implied Volatility)**: Line chart showing IV level and percentile
- **Vanna**: Bar/line showing net vanna (dealer bias from IV)
- **Vanna × IV**: Line showing trend indication

### 4. Panel 3: Dealer Flow Panel
Add 3 traces:
- **Charm**: Time decay flows (dealer delta hedging pressure)
- **GEX Pressure**: Reaction strength indicator
- **Dealer Flow Score**: Combined score (-100 to +100)

### 5. VWAP Calculation
Add before chart creation:
```python
# Calculate VWAP from recent intraday data
if price_df is not None and not price_df.empty:
    price_df['vwap'] = (price_df['close'] * price_df['volume']).cumsum() / price_df['volume'].cumsum()
    current_vwap = price_df['vwap'].iloc[-1]
```

### 6. Color Shading Implementation
Add vertical shaded regions based on dealer flow regime:
- Positive GEX regime: Light green background
- Negative GEX regime: Light red background
- Transition zones: Yellow/orange

### 7. Update Chart Legend
Simplify and clarify:
- Gamma Flip = major pivot
- GEX walls = S/R zones
- Vanna walls = pressure zones
- Put/Call walls = magnets/barriers
- VWAP = dynamic balance

### 8. Testing Checklist
- [ ] Chart displays correctly with data
- [ ] All 3 panels show properly
- [ ] Indicators calculate without errors
- [ ] Legend is readable
- [ ] Mobile responsive
- [ ] Handles missing data gracefully

## Implementation Strategy
1. Create helper function `create_multi_panel_chart(pred, price_df)`
2. Build each panel separately
3. Combine with make_subplots
4. Replace existing chart display
5. Test thoroughly
6. Commit changes

## Notes
- Keep existing functionality intact
- Maintain clean, readable code
- Add comments for complex calculations
- Handle None/missing data gracefully
