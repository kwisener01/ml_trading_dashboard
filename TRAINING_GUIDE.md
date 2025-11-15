# Training Guide - How Much Data & When to Train

## Manual Training

### Train Right Now

```bash
# Complete pipeline (recommended)
python automated_training.py

# Or step by step:
python data_collector.py    # Collect data
python train_models.py      # Train models
python s3_storage.py        # Upload to S3
```

### From Dashboard

You can also train from the Streamlit dashboard by clicking "Retrain Models" button (if you add one).

## Scheduling Options

### Daily at 10 AM EST (Trading Days Only)

**Updated workflow** now runs:
- Monday-Friday at 10:00 AM EST
- 30 minutes after market open (9:30 AM)
- Skips weekends and holidays automatically

**Pros:**
- Fresh data every trading day
- Models adapt quickly to market changes
- Ready for afternoon trading

**Cons:**
- More API usage (20 runs/month vs 4)
- More compute time (still free)

**Best for:** Active day traders, volatile markets

### After Market Close (4:00 PM EST)

Change cron to:
```yaml
cron: '0 21 * * 1-5'  # 4 PM EST = 9 PM UTC
```

**Pros:**
- Full trading day data included
- Better for EOD strategies

**Cons:**
- No intraday updates

**Best for:** Swing traders, overnight positions

### Weekly (Sunday Evening)

Change cron to:
```yaml
cron: '0 23 * * 0'  # 6 PM EST Sunday
```

**Pros:**
- Less frequent (minimal API usage)
- Full week of data
- Still fresh enough

**Cons:**
- May miss Monday opportunities
- Slower to adapt

**Best for:** Long-term strategies, low-frequency traders

## How Many Days of Training Data?

### Current: 365 Days (Recommended)

```python
df = collector.collect_training_data(
    symbol='SPY',
    days_back=365,
    interval='15min'
)
```

**What you get:**
- ~6,500 samples (15-min bars)
- Full year of market conditions
- Multiple market regimes (bull, bear, sideways)
- Seasonal patterns

**Pros:**
- Balanced dataset
- Good generalization
- Not too much historical noise
- Fast training (~2-3 min)

**Cons:**
- May include outdated market conditions
- Older regime changes

**Best for:** Most traders

### Option: 180 Days (6 Months)

```python
days_back=180
```

**What you get:**
- ~3,250 samples
- Recent market behavior only
- Faster training (~1-2 min)

**Pros:**
- More current market conditions
- Faster to adapt
- Less historical noise

**Cons:**
- Fewer samples (less robust)
- May overfit to recent regime
- Less seasonal data

**Best for:** Fast-moving markets, recent regime shifts

### Option: 730 Days (2 Years)

```python
days_back=730
```

**What you get:**
- ~13,000 samples
- Multiple market cycles
- More robust models

**Pros:**
- More training data
- Better generalization
- Captures rare events

**Cons:**
- Slower training (~5-6 min)
- May include outdated patterns
- Old market regimes less relevant

**Best for:** Conservative strategies, stress testing

### Option: 90 Days (3 Months)

```python
days_back=90
```

**What you get:**
- ~1,625 samples
- Very recent behavior only
- Very fast training (~1 min)

**Pros:**
- Most current market regime
- Very fast training
- Highly adaptive

**Cons:**
- Small dataset (may underfit)
- Missing seasonal patterns
- Less robust

**Best for:** Scalping, high-frequency, algorithm testing

## Recommendations by Trading Style

### Day Trader
```python
days_back=180  # 6 months
interval='5min'  # More granular
schedule='0 15 * * 1-5'  # Daily 10 AM
```
- Fresh data daily
- Recent patterns only
- 5-min bars for precision

### Swing Trader (You)
```python
days_back=365  # 1 year (current)
interval='15min'  # Good balance
schedule='0 15 * * 1-5'  # Daily 10 AM
```
- Full year of patterns
- Daily updates
- 15-min bars optimal

### Position Trader
```python
days_back=730  # 2 years
interval='1hour'  # Less noise
schedule='0 23 * * 0'  # Weekly Sunday
```
- Long-term patterns
- Weekly updates sufficient
- Hourly bars for big picture

## Data Collection Settings

### Current Settings (in automated_training.py)

```python
df = collector.collect_training_data(
    symbol='SPY',
    days_back=365,      # 1 year
    interval='15min'    # 15-minute bars
)
```

### To Change

Edit `automated_training.py` line ~20:

```python
# For more recent data
days_back=180,

# For more historical data
days_back=730,

# For finer granularity
interval='5min',

# For less noise
interval='1hour',
```

## Training Frequency Guidelines

### How Often Should You Retrain?

**High volatility (VIX > 25):**
- Train daily
- Use 90-180 days data
- Adapt quickly to changing conditions

**Normal volatility (VIX 15-25):**
- Train daily or every 2-3 days
- Use 365 days data (current)
- Balance between fresh and stable

**Low volatility (VIX < 15):**
- Train weekly
- Use 365-730 days data
- Models stay relevant longer

## Sample Size Requirements

### Minimum Samples per Model

For reliable ML models:
- **Minimum**: 1,000 samples
- **Good**: 5,000 samples
- **Optimal**: 10,000+ samples

### Your Current Setup

With 365 days @ 15min bars:
- **Total samples**: ~6,500
- **Status**: Good for reliable models
- **Training split**: 80/20 = 5,200 train, 1,300 test

## API Usage

### Tradier API Limits

**Sandbox (testing):**
- 120 requests/minute
- 10,000 requests/day

**Production:**
- 120 requests/minute
- 60,000 requests/day

### Your Usage

**Daily training at 10 AM:**
- 1 workflow run = ~10 API calls
- 20 trading days = 200 calls/month
- Well within limits

**Data collection:**
- 365 days of 15min data = ~5-10 API calls
- Pagination handles large datasets
- No issues with limits

## Recommended Setup for You

Based on your trading style (swing/day trading with Vanna levels):

```python
# Edit automated_training.py

df = collector.collect_training_data(
    symbol='SPY',
    days_back=365,      # 1 year - good balance
    interval='15min'    # Captures intraday without too much noise
)
```

**Schedule:**
```yaml
# .github/workflows/scheduled_training.yml
cron: '0 15 * * 1-5'  # Daily 10 AM EST, Mon-Fri
```

**Why:**
- 365 days captures seasonal patterns + recent data
- 15min bars good for swing trades (not too noisy)
- Daily 10 AM gives you fresh predictions for afternoon
- ~6,500 samples = robust models
- 5 min training time = fast enough for daily runs

## Testing Different Settings

To test optimal settings for your strategy:

```bash
# Test with 6 months
# Edit automated_training.py: days_back=180
python automated_training.py

# Test with 2 years
# Edit automated_training.py: days_back=730
python automated_training.py

# Compare model performance in prediction_log.csv
# Look at trade_quality_score - higher = better
```

## Summary

**Current Setup (Recommended):**
- 365 days of data
- 15-minute bars
- Daily training at 10 AM EST
- ~6,500 samples
- 2-3 minute training time

**Alternative for More Recent Focus:**
- 180 days of data
- 5-minute bars
- Daily training at 10 AM EST
- ~5,000 samples
- 1-2 minute training time

**Alternative for More Stability:**
- 730 days of data
- 30-minute bars
- Weekly training (Sunday evening)
- ~10,000 samples
- 5-6 minute training time

Your current 365-day setup is a good sweet spot for most trading styles!
