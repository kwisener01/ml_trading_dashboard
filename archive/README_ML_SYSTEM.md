# 🤖 ML Trading System - Predict Levels, Targets & When NOT to Trade

A machine learning system that predicts:
1. **Support/Resistance Levels** - Where price is likely to go
2. **Price Targets** - Upside potential and downside risk
3. **When NOT to Trade** - Filter out low-quality setups (MOST IMPORTANT)

Built specifically for 0DTE options trading on SPY/QQQ using Tradier API.

---

## 🎯 System Overview

### Three Core Models

1. **Trade Quality Classifier** (Priority #1)
   - Predicts if a setup is worth trading (0-100 score)
   - Filters out choppy markets, low volume, bad times
   - Prevents losses by avoiding bad setups

2. **Profit Target Classifier**
   - Predicts probability of hitting profit target
   - Based on historical patterns and market regime

3. **Price Level Regressor**
   - Predicts future high/low levels
   - Sets realistic targets and stops

### Key Features That Drive Predictions

- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR
- **Market Regime**: Trend strength, choppiness, volatility
- **Time Features**: Time of day, day of week, avoid first/last 30min
- **Support/Resistance**: Key price levels and pivot points
- **Volume Analysis**: Relative volume and volume surges

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_ml.txt
```

### 2. Add Your Tradier API Token

Edit each Python file and replace:
```python
API_TOKEN = "YOUR_TRADIER_API_TOKEN_HERE"
```

Get your token at: https://documentation.tradier.com/brokerage-api/getting-started

### 3. Run Complete Pipeline

```bash
# Step 1: Collect historical data
python data_collector.py

# Step 2: Engineer features
python feature_engineering.py

# Step 3: Train models
python train_models.py

# Step 4: Make predictions
python predictor.py
```

---

## 📁 File Structure

```
data_collector.py          - Fetch historical & real-time data from Tradier
feature_engineering.py     - Create ML features (indicators, regime, time)
train_models.py           - Train all ML models
predictor.py              - Real-time predictions & signals
tradier_options_test.py   - Test Tradier API connection
requirements_ml.txt       - Python dependencies
```

---

## 🔄 Complete Workflow

### Phase 1: Data Collection

```python
from data_collector import TradierDataCollector

collector = TradierDataCollector(API_TOKEN)

# Collect 60 days of data
data = collector.collect_training_data('SPY', days_back=60)
collector.save_data(data, 'spy_training_data')
```

**Outputs:**
- `spy_training_data_daily.csv` - Daily price data
- `spy_training_data_intraday.csv` - 5-minute intraday data
- `spy_training_data_metadata.json` - Collection info

### Phase 2: Feature Engineering

```python
from feature_engineering import FeatureEngineering
import pandas as pd

# Load data
df = pd.read_csv('spy_training_data_intraday.csv')

# Create features
fe = FeatureEngineering(df)
fe.add_technical_indicators()
fe.add_support_resistance_levels()
fe.add_market_regime_features()
fe.add_time_features()
fe.create_target_labels()

# Save dataset
features, full_data = fe.get_features_for_ml()
full_data.to_csv('spy_ml_dataset.csv', index=False)
```

**Features Created:**
- 50+ technical indicators
- Market regime classifiers
- Time-based features (avoid bad hours)
- Support/resistance levels
- Target labels for supervised learning

### Phase 3: Model Training

```python
from train_models import TradingMLModel

# Load dataset
df = pd.read_csv('spy_ml_dataset.csv')

# Train all models
trainer = TradingMLModel()
models = trainer.train_all_models(df)

# Save models
trainer.save_models('spy_trading_model')
```

**Models Trained:**
1. Trade Quality Classifier (XGBoost)
2. Profit Target Classifier (XGBoost)
3. Future High Regressor (XGBoost)
4. Future Low Regressor (XGBoost)

**Outputs:**
- Model files (.pkl)
- Performance metrics (.json)
- Feature importance rankings (.csv)

### Phase 4: Real-Time Predictions

```python
from predictor import TradingPredictor

# Initialize predictor
predictor = TradingPredictor(API_TOKEN)

# Get prediction
predictions = predictor.predict('SPY')

# Display signal
print(predictor.format_signal(predictions))
```

**Prediction Output:**
```
================================================================================
TRADING SIGNAL: SPY @ $681.50
Time: 2025-11-12T14:30:00
================================================================================

🎯 TRADE QUALITY: 78.5/100
✅ SIGNAL: TRADEABLE SETUP

📊 Win Probability: 65.2%

🎯 PRICE TARGETS:
  ↗️  Upside Target: $684.20 (+0.40%)
  ↘️  Downside Stop:  $679.80 (-0.25%)

💰 Risk/Reward Ratio: 1.60:1

📈 MARKET CONDITIONS:
  - Trend Strength: Moderate (32.5)
  - Choppiness: Low (28.3)
  - Volatility: Moderate (0.45)

================================================================================
```

---

## 🎛️ Configuration & Customization

### Adjust Delta Ranges

In `tradier_options_test.py`:
```python
# More aggressive (higher delta)
calls = scanner.filter_options_by_delta(
    chain_data, 
    min_delta=0.50,  # Changed from 0.30
    max_delta=0.80,  # Changed from 0.70
    option_type='call'
)
```

### Change Profit/Loss Thresholds

In `feature_engineering.py`:
```python
fe.create_target_labels(
    forward_periods=12,      # How many bars ahead to look
    profit_threshold=0.005,  # 0.5% profit target
    loss_threshold=-0.003    # 0.3% stop loss
)
```

### Modify Trade Quality Threshold

In `predictor.py`:
```python
predictions['should_trade'] = quality_proba > 0.6  # 60% threshold
```

Lower = more trades (less selective)
Higher = fewer trades (more selective)

### Add More Symbols

```python
symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'TSLA', 'AAPL']
```

---

## 📊 Understanding the Signals

### Trade Quality Score (0-100)

- **80-100**: Excellent setup, high confidence
- **60-79**: Good setup, trade with normal size
- **40-59**: Marginal, consider skipping
- **0-39**: Poor setup, DO NOT TRADE

### When System Says "AVOID"

The model avoids trades when:
- ❌ Choppy market (Choppiness > 50)
- ❌ Weak trend (Trend Strength < 20)
- ❌ Low volume (Volume Ratio < 0.8)
- ❌ Bad time (First/last 30 min of market)
- ❌ Tight range (Range compression)

**This is the most valuable feature** - avoiding bad trades prevents losses!

### Win Probability

- **>65%**: High confidence
- **50-65%**: Moderate confidence
- **<50%**: Flip a coin, probably skip

### Risk/Reward Ratio

- **>2:1**: Excellent
- **1.5-2:1**: Good
- **<1.5:1**: Not worth it

---

## 🔗 Integration with Lindy Agent

### Option 1: API Endpoint

Create a simple Flask API:

```python
from flask import Flask, jsonify
from predictor import TradingPredictor

app = Flask(__name__)
predictor = TradingPredictor(API_TOKEN)

@app.route('/signal/<symbol>')
def get_signal(symbol):
    pred = predictor.predict(symbol)
    return jsonify(pred)

if __name__ == '__main__':
    app.run(port=5000)
```

Call from Lindy:
```
GET http://localhost:5000/signal/SPY
```

### Option 2: CSV Export

```python
# In predictor.py
predictions = predictor.predict('SPY')
predictor.save_prediction(predictions, 'lindy_signals.csv')
```

Lindy reads `lindy_signals.csv` on a schedule.

### Option 3: Webhook

Send predictions directly to Lindy webhook:

```python
import requests

def send_to_lindy(predictions):
    webhook_url = "YOUR_LINDY_WEBHOOK_URL"
    requests.post(webhook_url, json=predictions)
```

---

## 📈 Model Performance Metrics

After training, check these files:
- `spy_trading_model_metrics_*.json` - All model accuracies
- `spy_trading_model_importance_*.csv` - Top features driving predictions

**Key Metrics to Watch:**

1. **Trade Quality Model**
   - Precision: How often "trade" signals are actually good
   - Recall: How many good setups are caught

2. **Profit Target Model**
   - Accuracy: Overall correctness
   - F1 Score: Balance of precision/recall

3. **Price Level Models**
   - MAE: Average dollar error
   - Mean % Error: Average percentage error

---

## 🔍 Troubleshooting

### "No models found"
Run `train_models.py` first to train models.

### "Insufficient data for prediction"
Need at least 50 bars. Market might be closed or just opened.

### "Error 401 from Tradier"
Check your API token is correct.

### Low accuracy after training
- Collect more data (increase `days_back`)
- Adjust profit/loss thresholds
- Add more features
- Try different model parameters

### Too many "AVOID" signals
- Lower trade quality threshold from 0.6 to 0.5
- Adjust market regime thresholds in feature engineering

---

## 🎓 Next Steps

### 1. Backtest the System

Create `backtest.py`:
```python
# Load historical predictions
# Compare to actual outcomes
# Calculate win rate, profit factor, max drawdown
```

### 2. Paper Trade

Run predictor every 5 minutes and log signals without real money.

### 3. Optimize Parameters

Use grid search to find optimal:
- Profit/loss thresholds
- Trade quality threshold
- Feature combinations

### 4. Add Option-Specific Features

- Open interest
- Implied volatility rank
- Put/call ratio
- Unusual option activity

### 5. Deploy to Cloud

Run on AWS/GCP for 24/7 predictions:
```bash
# Schedule with cron
*/5 * * * * /usr/bin/python /path/to/predictor.py
```

---

## 💡 Pro Tips

1. **Start Conservative**: Use high trade quality threshold (0.7) until you trust the system

2. **Track Performance**: Log every prediction and actual outcome to improve models

3. **Retrain Regularly**: Market regimes change, retrain weekly/monthly

4. **Respect the "AVOID"**: The system's best feature is telling you NOT to trade

5. **Position Sizing**: Trade smaller when quality score is 60-70, larger when 80+

6. **Combine with Your Analysis**: Use ML as confirmation, not sole decision maker

---

## 📞 Support

Issues? Check:
1. All dependencies installed (`pip install -r requirements_ml.txt`)
2. API token is valid
3. Market is open when testing real-time predictions
4. Models are trained before making predictions

---

## 🔐 Security Note

**Never commit your API token to version control!**

Use environment variables:
```python
import os
API_TOKEN = os.environ.get('TRADIER_API_TOKEN')
```

---

## 📄 License

Built for personal trading use. Use at your own risk. Not financial advice.

---

**Remember: The best trade is often the one you DON'T take! 🎯**
