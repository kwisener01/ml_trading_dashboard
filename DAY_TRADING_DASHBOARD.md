# Day Trading Dashboard - New Features

## ✅ What Was Added

The Streamlit dashboard now has **full day trading support** with the following features:

### 1. Trading Mode Selector (Sidebar)
- **Radio button** to choose between:
  - "Daily Trading" (swing trades, multi-day holds)
  - "Day Trading (Intraday)" (same-day trades, close by 4pm)

### 2. Intraday Chart Interval (Day Trading Mode Only)
- Dropdown to select chart interval:
  - **5min** (recommended - 78 bars/day)
  - **15min** (26 bars/day)
  - **1min** (390 bars/day for scalping)

### 3. Market Hours Status (Day Trading Mode Only)
Real-time monitoring of market hours:
- ⏸️ **Pre-market** - Shows minutes until market open
- 🔔 **Market Closed** - After 4:00 PM
- ⚠️ **Avoid Zone** - First 30 minutes (9:30-10:00 AM)
- ⚠️ **Closing Time** - Last hour (3:00-4:00 PM)
- ✅ **Optimal Hours** - Best trading window (10:00 AM-3:00 PM)

### 4. Live Intraday Candlestick Chart
When in Day Trading mode, chart shows:
- **Candlestick bars** - OHLC for selected interval
- **Volume bars** - On secondary y-axis
- **Vanna support level** - Green dashed line (dealer buying zone)
- **Vanna resistance level** - Red dashed line (dealer selling zone)
- **Profit target** - Green dotted line
- **Stop loss** - Red dotted line
- **Hover details** - Price, volume, time

### 5. Intraday Statistics
Four key metrics displayed below chart:
- **Bars Today** - Number of bars so far
- **Day High** - Highest price today
- **Day Low** - Lowest price today
- **Day Range** - High-Low as percentage

### 6. Context-Aware Information
The sidebar "About" section changes based on mode:
- **Day Trading Mode**: Shows intraday features
- **Daily Trading Mode**: Shows swing trading features

### 7. Welcome Screen Updates
When no prediction is generated yet:
- Shows day trading specific features if in intraday mode
- Lists all capabilities (charts, market hours, risk management)

---

## 🚀 How to Use Day Trading Mode

### Step 1: Open Dashboard
```bash
streamlit run dashboard.py
```
Dashboard opens at: **http://localhost:8501**

### Step 2: Select Day Trading Mode
In the sidebar:
1. Find "📊 Trading Mode" section
2. Click **"Day Trading (Intraday)"** radio button

### Step 3: Configure Interval
1. Select chart interval from dropdown:
   - **5min** (recommended for beginners)
   - **15min** (for slower pace)
   - **1min** (for advanced scalpers)

### Step 4: Check Market Hours
Sidebar shows current market status:
- If **pre-market** → Wait for market open
- If **avoid zone** → Wait until 10:00 AM
- If **optimal hours** → Green light to trade!
- If **closing time** → Prepare to close positions

### Step 5: Generate Prediction
1. Select symbol (SPY, QQQ, IWM, DIA)
2. Adjust trade quality threshold (60 = selective, 40 = more trades)
3. Click **"🔄 Generate Prediction"**

### Step 6: Analyze Intraday Chart
The candlestick chart appears showing:
- **Green candles** = Price closed higher than open
- **Red candles** = Price closed lower than open
- **Blue volume bars** = Trading activity
- **Horizontal lines** = Key support/resistance levels

### Step 7: Watch for Entry Signals
Look for:
1. **Trade Quality** > 60/100 (✅ TRADEABLE)
2. **Optimal Hours** status = green
3. **Price near Vanna support** = bullish entry
4. **Strong trend, low chop** = ideal conditions
5. **High volume** = confirmation

### Step 8: Execute Trade (Outside Dashboard)
Use your broker to:
- Enter at current price
- Set profit target (shown on chart)
- Set stop loss (shown on chart)
- Position size: 5-15% of capital

### Step 9: Monitor Throughout Day
- Refresh prediction every 15-30 minutes
- Watch for signals to exit early
- Close ALL positions by 3:30 PM

---

## 📊 Example Day Trading Session

### 9:00 AM - Pre-Market
- Dashboard shows: ⏸️ Pre-market (30 min to open)
- **Action**: Review overnight news, plan day

### 9:30 AM - Market Opens
- Dashboard shows: ⚠️ Avoid zone (first 30 min)
- **Action**: Watch, don't trade yet (volatile, wide spreads)

### 10:00 AM - Optimal Hours Begin
- Dashboard shows: ✅ Optimal trading hours
- Generate prediction for SPY
- **Result**:
  ```
  SPY @ $580.50
  Trade Quality: 72/100 ✅ TRADEABLE
  Win Probability: 65%
  Profit Target: $581.25 (+0.3%)
  Stop Loss: $579.85 (-0.2%)
  Vanna Support: $580.00 (STRONG)
  ```
- **Chart shows**: Price bouncing off Vanna support at $580
- **Action**: Enter LONG at $580.50

### 10:30 AM - In Position
- Refresh prediction
- Price at $580.80 (up $0.30)
- Still shows TRADEABLE
- **Action**: Hold position

### 11:15 AM - Profit Target Hit
- Price touches $581.25
- Profit = $0.75 per share = **0.3% gain**
- On $1,000 position = **$3 profit**
- **Action**: Exit at profit target

### 11:30 AM - Looking for Next Trade
- Generate new prediction
- **Result**: Trade Quality: 45/100 🚫 AVOID
- Reasons: "Market is too choppy"
- **Action**: Stay flat, no trade

### 2:00 PM - Second Setup
- Generate prediction for QQQ
- **Result**:
  ```
  QQQ @ $485.20
  Trade Quality: 68/100 ✅ TRADEABLE
  Win Probability: 62%
  ```
- **Action**: Enter LONG at $485.20

### 3:00 PM - Closing Hour
- Dashboard shows: ⚠️ Closing time (last hour)
- Position at $485.50 (up $0.30)
- **Action**: Close early to avoid end-of-day volatility
- Exit at $485.50
- Profit = **0.06%** = $0.60 on $1,000

### 3:30 PM - End of Day
- Close all remaining positions
- **Day Results**:
  - Trade 1: +0.3% = $3.00
  - Trade 2: +0.06% = $0.60
  - **Total**: +0.36% = **$3.60/day**
  - On $10,000 account = $36/day = **$180/week**

---

## 🎯 Day Trading Best Practices

### Timing
1. ✅ **Trade 10:00 AM - 3:00 PM** (optimal hours)
2. ❌ **Avoid 9:30-10:00 AM** (too volatile)
3. ❌ **Avoid 3:30-4:00 PM** (erratic moves)
4. ✅ **Close by 3:30 PM** (no overnight risk)

### Position Sizing
- **Conservative**: 5% of capital per trade
- **Moderate**: 10% of capital
- **Aggressive**: 15% of capital
- **Max**: Never more than 15%

### Trade Frequency
- **Conservative**: 1-2 trades/day
- **Moderate**: 2-3 trades/day
- **Aggressive**: 3-5 trades/day
- **Avoid**: Overtrading (>5 trades)

### Quality Threshold
- **High selectivity**: 70/100 (fewer trades, higher quality)
- **Balanced**: 60/100 (2-3 good setups/day)
- **Aggressive**: 50/100 (more trades, lower quality)

### Profit Targets
- **Safe**: 0.2-0.3% per trade
- **Standard**: 0.3-0.5% per trade
- **Aggressive**: 0.5-1.0% per trade

### Stop Losses
- **Always use stops**: 0.15-0.3%
- **Never wider** than 0.5%
- **Respect the stop** - don't move it

---

## 🔧 Technical Details

### Data Refresh
- **Intraday chart**: Fetches live data when prediction generated
- **Price updates**: Click "Generate Prediction" to refresh
- **Auto-refresh**: Not enabled (to avoid API rate limits)

### API Calls
Each prediction makes:
- 1 call for current quote
- 1 call for intraday bars (if day trading mode)
- Total: ~2 API calls per prediction

### Performance
- **Chart loading**: 1-2 seconds
- **Prediction**: 2-3 seconds
- **Total**: ~5 seconds per update

### Limitations
- **Market hours only**: No data when market closed
- **Delays**: API data has ~1-2 second lag
- **Sandbox limitations**: May have delayed data

---

## 🆚 Day Trading vs Daily Trading Comparison

| Feature | Day Trading Mode | Daily Trading Mode |
|---------|-----------------|-------------------|
| **Chart Type** | Candlesticks (5/15/1min) | Price target visualization |
| **Timeframe** | Intraday (same day) | Multi-day swings |
| **Profit Target** | 0.3% (tight) | 0.5% (wider) |
| **Stop Loss** | 0.2% (tight) | 0.3% (wider) |
| **Position Holds** | Close by 4pm daily | Can hold overnight |
| **Vanna Levels** | 0DTE (same day exp) | Monthly expiration |
| **Market Hours** | Real-time monitoring | Not critical |
| **Trades per Day** | 2-5 trades | 1-2 trades/week |
| **Best For** | Active traders, $25k+ | Swing traders, any capital |

---

## 📝 Tips for Success

### 1. Start Small
- Use 5% position size first week
- Only 1 trade/day for first week
- Focus on learning, not profits

### 2. Follow the Rules
- Only trade optimal hours (10am-3pm)
- Respect quality threshold (>60)
- Always use stops
- Close by 3:30pm

### 3. Track Everything
- Dashboard logs all predictions
- Keep your own trade journal
- Review what worked/didn't

### 4. Use Vanna Levels
- **Near Vanna support** = bullish entry
- **Near Vanna resistance** = take profits
- **Breaking Vanna level** = strong momentum

### 5. Know When to Stop
- 2 losing trades in a row → stop for day
- Quality < 60 all day → stay flat
- You feel emotional → step away

---

## 🐛 Troubleshooting

### Chart Not Showing
- **Cause**: Market is closed or no data available
- **Fix**: Check market hours status in sidebar

### "No intraday data available"
- **Cause**: Weekend, holiday, or pre-market
- **Fix**: Wait for market hours (9:30am-4pm ET)

### Empty Candlesticks
- **Cause**: Selected interval too granular (1min on slow day)
- **Fix**: Switch to 5min or 15min interval

### Vanna Levels Missing
- **Cause**: Not calculated yet or NaN values
- **Fix**: Generate fresh prediction

### Dashboard Not Updating
- **Cause**: Streamlit cache
- **Fix**: Refresh browser (F5) or restart dashboard

---

## 🚀 What's Next?

### Future Enhancements (Potential)
1. **Auto-refresh** every 1-5 minutes
2. **Alerts** when trade quality crosses threshold
3. **Multi-symbol watchlist** with rankings
4. **Real-time P&L tracking** for day
5. **Trade execution** via broker API
6. **Historical chart** with yesterday's trades
7. **Sound alerts** for entry signals
8. **Mobile responsive** layout

---

## 📌 Quick Reference

### Best Day Trading Setup
- **Mode**: Day Trading (Intraday)
- **Interval**: 5min
- **Symbol**: SPY (most liquid)
- **Quality Threshold**: 60
- **Hours**: 10:00 AM - 3:00 PM
- **Position Size**: 10% of capital
- **Profit Target**: 0.3%
- **Stop Loss**: 0.2%
- **Max Trades**: 3 per day

### Success Formula
```
Quality > 60
+ Optimal Hours
+ Vanna Support Nearby
+ Strong Trend + Low Chop
+ High Volume
= TAKE THE TRADE
```

---

**Dashboard is ready for day trading!** 🎯

Just refresh your browser at **http://localhost:8501** to see all the new features.

Happy trading! 📈
