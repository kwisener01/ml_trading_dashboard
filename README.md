# ML Trading System with Day Trading Dashboard

A complete machine learning trading system for both daily and intraday (day trading) with real-time predictions, Vanna-based support/resistance levels, and a Streamlit dashboard.

## Features

- **Dual Trading Modes**: Daily swing trading + Intraday day trading
- **ML Models**: 4 XGBoost models for trade quality, price targets, support/resistance
- **Vanna Levels**: Options Greeks-based support/resistance (0DTE for day trading)
- **DOE Optimization**: Latin Hypercube Sampling for hyperparameter tuning
- **Backtesting**: Walk-forward testing with performance metrics
- **Live Dashboard**: Streamlit web interface with candlestick charts
- **Market Hours Monitoring**: Real-time status for optimal trading windows

## Quick Start

### 1. Install Dependencies

```bash
pip install pandas numpy scikit-learn xgboost streamlit plotly requests python-dotenv scipy
```

### 2. Set Up API Token

Create a `.env` file:
```
TRADIER_API_TOKEN="your_api_token_here"
USE_SANDBOX=false
```

### 3. Collect Data

```bash
# Daily data (swing trading)
python data_collector.py

# OR Intraday data (day trading)
python intraday_collector.py --symbol SPY --days 10 --interval 5min
```

### 4. Train Models

```bash
python train_models.py
```

### 5. Launch Dashboard

```bash
streamlit run dashboard.py
```

Open http://localhost:8501 in your browser.

## Dashboard Usage

### Daily Trading Mode
1. Select "Daily Trading" in sidebar
2. Choose symbol (SPY, QQQ, IWM, DIA)
3. Click "Generate Prediction"
4. View predictions, targets, market conditions

### Day Trading Mode
1. Select "Day Trading (Intraday)" in sidebar
2. Choose interval (5min recommended)
3. Check market hours status
4. Generate prediction
5. View live candlestick chart with Vanna levels

## File Structure

```
files_ml_system/
├── data_collector.py              # Daily data collection
├── intraday_collector.py          # Intraday/day trading data
├── feature_engineering.py         # 72+ ML features
├── second_order_greeks.py         # Vanna calculations
├── train_models.py                # Train 4 ML models
├── hyperparameter_optimizer.py    # DOE optimization
├── backtest.py                    # Historical testing
├── predictor.py                   # Generate predictions
├── dashboard.py                   # Streamlit UI
├── quickstart.py                  # Run full pipeline
├── .env                           # API token (YOU create this)
├── SYSTEM_SUMMARY.md              # Complete documentation
├── DAY_TRADING_DASHBOARD.md       # Dashboard guide
└── VANNA_LEVELS_GUIDE.md          # Vanna levels explained
```

## Documentation

- **SYSTEM_SUMMARY.md** - Complete system documentation
- **DAY_TRADING_DASHBOARD.md** - Dashboard usage guide
- **VANNA_LEVELS_GUIDE.md** - Vanna levels explanation

## Requirements

- Python 3.11+
- Tradier API account (free)
- $25,000+ for day trading (PDT rule)

## What's missing for confident live trading

- **Broker execution & risk controls**: This repo stops at generating signals; you'll need broker connectivity, position sizing, max-loss/position limits, and pre-/post-trade risk checks to protect capital.
- **Fill quality & latency monitoring**: Add real-time slippage/latency tracking plus alerts so you know when the model assumptions no longer match live fills.
- **Paper trading & guardrails**: Run in a paper account first, compare expected vs. realized P&L, and add kill-switch conditions (e.g., stop after consecutive losses or slippage spikes).
- **Alternative data (dark pools)**: Dark pool prints/levels are not collected here; you'd need a feed (institutional tape, dark pool summaries) and a parser to overlay those levels on the dashboard.
- **0DTE options depth/flow**: The dashboard only computes Vanna levels from Tradier quotes; it does not track zero-DTE gamma walls or large option strikes. Integrating a high-granularity options order-flow/oi feed and summarizing the top strikes by expiry would be required.
- **Gamma walls & large strikes (where they would show)**: The current UI never renders gamma walls or the biggest call/put strikes—`calculate_options_flow_data` in `predictor.py` leaves those fields as placeholders because the Tradier chain feed lacks the depth needed for reliable walls. You would need to add an options flow provider (OI + volume by strike/expiry) and then plot those levels alongside the Vanna lines on the intraday chart.

## How accurate are the gamma and vanna levels?

- **Dependent on Tradier depth and Greeks**: Both gamma and vanna exposures are derived from a single nearest-dated Tradier options chain. If Tradier omits Greeks the code falls back to a fixed 0.25 IV assumption, so exposures can diverge from actual market hedging flows.【F:gex_calculator.py†L98-L139】【F:gex_calculator.py†L201-L274】
- **Single-expiration snapshot**: The calculator only looks at the closest 0DTE/1DTE expiry, so it misses size resting in later expirations that often anchor price; walls may move intraday as OI updates.【F:gex_calculator.py†L98-L139】【F:gex_calculator.py†L201-L274】
- **UI fallbacks when data fails**: The dashboard example panel substitutes simple recent highs/lows as “vanna” levels if it can’t load options data, which are illustrative only—not dealer positioning.【F:dashboard.py†L1173-L1234】
- **No dark pool or flow enrichment**: Put/Call walls and dark-pool-derived levels remain placeholders in `calculate_options_flow_data`, so there’s no confirmation from flow or hidden liquidity to validate the computed vanna/gamma levels.【F:predictor.py†L154-L219】

Treat the current gamma/vanna markers as rough guides; verifying them against a richer options flow feed (with per-strike OI/volume across expirations) is needed before trading on them.

## Disclaimer

**Not financial advice.** Use at your own risk. Past performance doesn't guarantee future results.

Test with paper trading first before using real capital.

## License

MIT License
