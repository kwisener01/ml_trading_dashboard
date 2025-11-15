# AWS Lambda Deployment Guide

Deploy your automated training pipeline to AWS Lambda for serverless, scheduled execution.

## Overview

AWS Lambda will:
- Run on a schedule (e.g., weekly)
- Collect fresh market data
- Train ML models
- Upload to S3
- Run completely independently of your computer

## Cost Estimate

**Lambda Free Tier:**
- 1 million requests/month FREE
- 400,000 GB-seconds compute/month FREE

**Your usage:**
- 4 runs/month (weekly)
- ~5 minutes per run
- ~2 GB memory
- **Total cost: $0.00** (well within free tier)

## Deployment Steps

### Option 1: AWS Lambda (Recommended)

#### 1. Install AWS SAM CLI

```bash
# Windows
pip install aws-sam-cli

# Mac/Linux
brew install aws-sam-cli
```

#### 2. Create Lambda Layer for Dependencies

```bash
# Create layer directory
mkdir python
pip install -r requirements.txt -t python/

# Package layer
zip -r lambda_layer.zip python/
```

#### 3. Create Lambda Function

Go to AWS Console > Lambda > Create function:

- **Name**: ml-trading-trainer
- **Runtime**: Python 3.11
- **Memory**: 2048 MB
- **Timeout**: 15 minutes
- **Execution role**: Create new with S3 permissions

#### 4. Upload Code

```bash
# Package your code
zip -r function.zip automated_training.py data_collector.py train_models.py s3_storage.py

# Upload via AWS CLI
aws lambda update-function-code \
  --function-name ml-trading-trainer \
  --zip-file fileb://function.zip
```

#### 5. Set Environment Variables

In Lambda console > Configuration > Environment variables:

```
TRADIER_API_TOKEN=your_token
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION=us-east-1
S3_BUCKET_NAME=ml-trading-models-kwise
```

#### 6. Create EventBridge Schedule

AWS Console > EventBridge > Rules:

- **Name**: weekly-training
- **Schedule**: `cron(0 23 * * SUN *)` (every Sunday 11 PM UTC = 6 PM EST)
- **Target**: Lambda function (ml-trading-trainer)

### Option 2: GitHub Actions (Easier, Still Free)

#### 1. Push Code to GitHub

```bash
git init
git add .
git commit -m "Add automated training"
git remote add origin https://github.com/YOUR_USERNAME/ml-trading-system.git
git push -u origin main
```

#### 2. Add GitHub Secrets

Go to GitHub repo > Settings > Secrets and variables > Actions:

Add these secrets:
- `TRADIER_API_TOKEN`
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET_NAME`

#### 3. Enable GitHub Actions

The workflow file `.github/workflows/scheduled_training.yml` is already created!

It will automatically run:
- Every Sunday at 6 PM EST
- Or manually via "Actions" tab > "Run workflow"

**GitHub Actions Free Tier:**
- 2,000 minutes/month for private repos
- Unlimited for public repos
- Your usage: ~20 minutes/month
- **Cost: $0.00**

### Option 3: Heroku Scheduler (Easiest)

#### 1. Create Heroku Account

Go to [heroku.com](https://heroku.com) and sign up (free tier available)

#### 2. Install Heroku CLI

```bash
# Download from heroku.com/cli
```

#### 3. Create App

```bash
heroku create ml-trading-trainer
```

#### 4. Add Scheduler Add-on

```bash
heroku addons:create scheduler:standard
```

#### 5. Configure Schedule

```bash
heroku addons:open scheduler
```

Add job:
- **Schedule**: Every day at 6:00 PM EST
- **Command**: `python automated_training.py`

#### 6. Deploy

```bash
# Create Procfile
echo "web: python automated_training.py" > Procfile

# Deploy
git push heroku main
```

## Recommended Schedule

**Weekly training** (recommended):
- **When**: Sunday 6 PM EST (after markets close Friday)
- **Why**: Fresh data from full week, not too frequent
- **GitHub Actions cron**: `0 23 * * 0`
- **EventBridge cron**: `cron(0 23 * * SUN *)`

**Daily training** (if you want more frequent updates):
- **When**: Every day at 6 PM EST (after market close)
- **Why**: Most recent data, faster model updates
- **GitHub Actions cron**: `0 23 * * *`
- **EventBridge cron**: `cron(0 23 * * ? *)`

## Testing

Test locally first:

```bash
python automated_training.py
```

You should see:
1. Data collection
2. Model training
3. S3 upload
4. Success message

## Monitoring

### GitHub Actions
- View logs: GitHub repo > Actions tab
- Email notifications on failure

### AWS Lambda
- View logs: CloudWatch > Log groups > /aws/lambda/ml-trading-trainer
- Set up CloudWatch alarms for failures

### Heroku
- View logs: `heroku logs --tail`
- Dashboard shows last run status

## Comparison

| Feature | GitHub Actions | AWS Lambda | Heroku Scheduler |
|---------|---------------|------------|------------------|
| **Setup Time** | 5 min | 20 min | 10 min |
| **Cost** | Free | Free | Free tier available |
| **Reliability** | High | Very High | Medium |
| **Logs** | Easy to view | CloudWatch | Command line |
| **Manual trigger** | Yes (UI button) | Yes (console) | No |
| **Best for** | Already using GitHub | AWS ecosystem | Simple setup |

## Recommended: GitHub Actions

Easiest and most reliable for your use case:

1. Already have the workflow file
2. Free for public/private repos
3. Easy to view logs and manually trigger
4. No additional services needed

Just push to GitHub and add secrets!

## Auto-Download in Dashboard

Update dashboard.py to download latest models from S3 on startup:

```python
# Add at top of dashboard.py
import os
from s3_storage import S3StorageManager

# Download latest models if not present
if not os.path.exists('spy_trading_model_trade_quality_20251113_230157.pkl'):
    st.info("Downloading latest models from S3...")
    storage = S3StorageManager()
    storage.download_models(decompress=True)
    st.success("Models downloaded!")
```

This way your dashboard always uses the latest trained models.

## Next Steps

1. Choose deployment method (GitHub Actions recommended)
2. Set up scheduled runs
3. Test first manual run
4. Verify models upload to S3
5. Update dashboard to auto-download models

Your system will now train itself automatically every week!
