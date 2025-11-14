# 🎯 ML TRADING SYSTEM - QUICK REFERENCE

## 📦 What You Got

A complete machine learning system that predicts:
1. ✅ **When TO trade** (quality score 0-100)
2. 🚫 **When NOT to trade** (filters bad setups) - MOST VALUABLE
3. 🎯 **Where price will go** (targets & stops)
4. 📊 **Win probability** (success likelihood)

## ⚡ Quick Start (5 Minutes)

### 1. Install Everything
```bash
pip install -r requirements_ml.txt
```

### 2. Setup API Token (One-Time)
```bash
# Copy the template
cp .env.example .env

# Edit .env and add your token
# Change: TRADIER_API_TOKEN=your_api_token_here
# To:     TRADIER_API_TOKEN=your_actual_token
```

See [ENV_SETUP_GUIDE.md](ENV_SETUP_GUIDE.md) for detailed setup instructions.

### 3. Run Complete Pipeline
```bash
python quickstart.py
```
Choose option 1 for complete pipeline.

### 4. Launch Dashboard
```bash
streamlit run dashboard.py
```
Opens at http://localhost:8501

## 📁 Files Explained

| File | What It Does |
|------|-------------|
| `quickstart.py` | 🚀 Run everything with one command |
| `data_collector.py` | 📊 Fetch data from Tradier API |
| `feature_engineering.py` | 🔧 Create ML features |
| `train_models.py` | 🤖 Train prediction models |
| `predictor.py` | 🎯 Make real-time predictions |
| `dashboard.py` | 📈 Visual interface (Streamlit) |
| `README_ML_SYSTEM.md` | 📖 Complete documentation |

## 🎮 Usage Examples

### Command Line Prediction
```bash
python predictor.py
```
Outputs formatted trading signal.

### Dashboard (Visual)
```bash
streamlit run dashboard.py
```
Nice UI with charts and history.

### Programmatic Use
```python
from predictor import TradingPredictor

predictor = TradingPredictor('YOUR_TOKEN')
pred = predictor.predict('SPY')

print(f"Trade Quality: {pred['trade_quality_score']:.1f}/100")
print(f"Should Trade: {pred['should_trade']}")
print(f"Target: ${pred['predicted_high']:.2f}")
```

## 📊 Understanding the Output

### Trade Quality Score
- **80-100**: 🟢 Excellent - Trade with confidence
- **60-79**: 🟡 Good - Standard position size
- **40-59**: 🟠 Marginal - Probably skip
- **0-39**: 🔴 Poor - DEFINITELY SKIP

### Signal: TRADEABLE vs AVOID
- **✅ TRADEABLE**: All systems go
- **🚫 AVOID**: Market conditions are poor

### Why System Says "AVOID"
1. ❌ Choppy market (no clear direction)
2. ❌ Weak trend (low momentum)
3. ❌ Low volume (illiquid)
4. ❌ Bad timing (first/last 30min)
5. ❌ Compressed range (too tight)

## 🔗 Lindy Integration Options

### Option 1: REST API
Create Flask endpoint:
```python
# api.py
from flask import Flask, jsonify
from predictor import TradingPredictor

app = Flask(__name__)
predictor = TradingPredictor(API_TOKEN)

@app.route('/signal/<symbol>')
def signal(symbol):
    return jsonify(predictor.predict(symbol))

app.run(port=5000)
```

Lindy calls: `GET http://localhost:5000/signal/SPY`

### Option 2: CSV Export
```python
predictor.save_prediction(pred, 'signals.csv')
```
Lindy reads `signals.csv` on schedule.

### Option 3: Direct Integration
Copy predictor code into Lindy Python action.

## 🎯 Pro Tips

1. **Start Conservative**: Use 70+ quality threshold until you trust it
2. **Track Everything**: Log all predictions vs outcomes
3. **Respect "AVOID"**: Best feature is telling you NOT to trade
4. **Retrain Weekly**: Markets change, models need updates
5. **Position Sizing**: Smaller on 60-70, bigger on 80+

## 🔍 Customization

### Change Thresholds
In `predictor.py`:
```python
predictions['should_trade'] = quality_proba > 0.6  # Change 0.6
```

### Adjust Target Levels
In `feature_engineering.py`:
```python
fe.create_target_labels(
    forward_periods=12,     # Bars ahead
    profit_threshold=0.005,  # 0.5% profit
    loss_threshold=-0.003    # 0.3% stop
)
```

### Add More Symbols
In `predictor.py` or `data_collector.py`:
```python
symbols = ['SPY', 'QQQ', 'IWM', 'DIA', 'AAPL', 'TSLA']
```

## ⚠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| "No models found" | Run `python train_models.py` first |
| "401 from Tradier" | Check API token is correct |
| "Insufficient data" | Market might be closed or just opened |
| Low accuracy | Collect more data, adjust thresholds |
| Too many AVOIDs | Lower quality threshold (0.6 → 0.5) |

## 🚀 Next Steps

1. **Paper Trade**: Run for 1 week, track results
2. **Optimize**: Adjust thresholds based on performance
3. **Backtest**: Validate on historical data
4. **Deploy**: Set up automated scanning
5. **Scale**: Add more symbols, timeframes

## 📞 Support Resources

- **Full Docs**: README_ML_SYSTEM.md
- **Tradier API**: https://documentation.tradier.com
- **Model Metrics**: Check generated JSON files
- **Feature Importance**: Check generated CSV files

## 💡 Key Insight

**The ML model's superpower is telling you WHEN NOT TO TRADE.**

Avoiding bad setups prevents losses. That's more valuable than catching every winner.

---

## 🎓 Learning the System

**Day 1**: Run quickstart, generate predictions, explore dashboard
**Day 2**: Review model metrics, understand feature importance
**Day 3**: Paper trade with signals, track accuracy
**Week 2**: Optimize thresholds based on your results
**Week 3**: Integrate with Lindy or deploy automation
**Week 4**: Add custom features, retrain models

---

## ⚡ One-Liner

Train everything and see results:
```bash
python quickstart.py
```
Choose option 1, wait 5-10 minutes, done.

---

**Remember: Best trade is often the one you DON'T take! 🎯**
