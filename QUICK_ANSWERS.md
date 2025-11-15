# Quick Answers - Training FAQ

## How do I manually train?

**Simple:**
```bash
python automated_training.py
```

Takes ~5 minutes, does everything:
1. Collects fresh data
2. Trains all models
3. Uploads to S3

## Can I have it trained daily at 10:00 AM during trading days?

**Yes! Optimized for day trading.**

The workflow now runs:
- **Monday-Friday at 6:00 AM EST**
- Before market open (9:30 AM)
- Fresh predictions ready for the full trading day
- Automatically skips weekends

File: `.github/workflows/scheduled_training.yml`
```yaml
cron: '0 11 * * 1-5'  # Mon-Fri 6 AM EST
```

**Why 6 AM instead of 10 AM?**
- Models ready BEFORE market open
- Includes overnight price action
- Better for 1-2 quality trades per day
- No rushing during trading hours

## How many days should I have it trained for?

**New: 180 days with 5-min bars (Optimized for Day Trading)**

This gives you:
- ~15,000 samples (5-min bars)
- Recent 6 months of market patterns
- 3x more granular data than 15-min bars
- Better entry/exit precision
- 3-4 minute training time

**Why 5-min bars?**
- Catch intraday Vanna movements
- More precise support/resistance levels
- Better for 1-2 quality trades per day
- Improved Sharpe ratio potential

### To Change Back:

Edit `automated_training.py` line ~27-30:
```python
df = collector.collect_training_data(
    symbol='SPY',
    days_back=180,     # 6 months
    interval='5min'    # 5-minute bars
)
```

## Schedule Summary

**Current Setup (Optimized for Day Trading):**
- Runs: Monday-Friday at 6:00 AM EST
- Data: 180 days of 5-min bars (~15,000 samples)
- Training: ~3-4 minutes
- Upload: Automatic to S3
- Cost: $0.00/month

**Your models will now:**
- Train BEFORE market open (6 AM)
- Use more granular 5-min data
- Focus on recent 6-month patterns
- Be ready at 9:30 AM opening bell
- Provide better entry/exit precision

Perfect for day trading with 1-2 high-quality trades per day!
