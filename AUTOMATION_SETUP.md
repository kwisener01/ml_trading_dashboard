# Automated Training Setup - Quick Start Guide

Your ML trading system can now update itself automatically on a schedule without your computer running!

## What You've Set Up So Far

1. **S3 Storage** - Models backed up in AWS S3 cloud
2. **Automated Training Script** - `automated_training.py` does everything automatically
3. **GitHub Actions Workflow** - Ready to run on schedule

## Quick Setup (5 minutes)

### Option A: GitHub Actions (Recommended - FREE & Easy)

#### Step 1: Create GitHub Repository

```bash
# Initialize git (if not already)
cd C:\Trading\files_ml_system
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - ML trading system with automated training"
```

#### Step 2: Create GitHub Repo

1. Go to [github.com/new](https://github.com/new)
2. Name: `ml-trading-system`
3. Privacy: Your choice (free tier works for both)
4. Click "Create repository"

#### Step 3: Push Code

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/ml-trading-system.git

# Push code
git push -u origin main
```

#### Step 4: Add Secrets to GitHub

1. Go to your repo on GitHub
2. Click "Settings" > "Secrets and variables" > "Actions"
3. Click "New repository secret" and add each:

**Required secrets:**
- `TRADIER_API_TOKEN` = your Tradier API token
- `AWS_ACCESS_KEY_ID` = from your .env file
- `AWS_SECRET_ACCESS_KEY` = from your .env file
- `AWS_REGION` = us-east-1
- `S3_BUCKET_NAME` = ml-trading-models-kwise

#### Step 5: Enable GitHub Actions

1. Go to "Actions" tab in your GitHub repo
2. You'll see "Automated ML Training" workflow
3. Click "Enable workflow"

#### Step 6: Test It!

1. Go to "Actions" tab
2. Click "Automated ML Training"
3. Click "Run workflow" dropdown
4. Click green "Run workflow" button
5. Watch it run! (takes ~5 minutes)

**That's it!** Your system will now automatically:
- Run every Sunday at 6 PM EST
- Collect fresh market data
- Train new models
- Upload to S3

### Option B: AWS Lambda (More Control)

See `lambda_deployment.md` for detailed Lambda setup.

Benefits:
- More control over scheduling
- Longer run time limits
- Integration with other AWS services

## What Happens Automatically

Every week (or whatever schedule you set):

```
1. Collect Data (1-2 min)
   - Fetch latest 365 days of SPY data
   - Calculate all features (Vanna, VIX, Greeks)
   - Save to spy_ml_dataset.csv

2. Train Models (2-3 min)
   - Train trade quality classifier
   - Train price target regressors
   - Train profit/loss predictors
   - Save all models locally

3. Upload to S3 (30 sec)
   - Compress models with gzip
   - Upload to S3 bucket
   - Upload training data

4. Done!
   - Models in S3 are now latest version
   - Dashboard downloads them automatically
```

## Monitoring Your Automated Training

### GitHub Actions

**View logs:**
1. Go to your repo > Actions tab
2. Click on any run
3. View detailed logs for each step

**Get notifications:**
- GitHub sends email if workflow fails
- Green checkmark in Actions tab = success
- Red X = failure (check logs)

### AWS Lambda

**View logs:**
1. AWS Console > CloudWatch
2. Log Groups > /aws/lambda/ml-trading-trainer
3. View detailed execution logs

**Set up alerts:**
- CloudWatch Alarms for failures
- SNS notifications to email/SMS

## Customizing the Schedule

Edit `.github/workflows/scheduled_training.yml`:

**Weekly (Sunday 6 PM EST):**
```yaml
cron: '0 23 * * 0'
```

**Daily (6 PM EST):**
```yaml
cron: '0 23 * * *'
```

**Every Monday and Friday (6 PM EST):**
```yaml
cron: '0 23 * * 1,5'
```

**Twice weekly (Wednesday and Sunday 6 PM EST):**
```yaml
cron: '0 23 * * 0,3'
```

**Cron format:** `minute hour day month weekday`
- 0-59 for minute
- 0-23 for hour (UTC time)
- 1-7 for weekday (0 or 7 = Sunday)

**Note:** GitHub Actions uses UTC time. EST = UTC - 5, so 6 PM EST = 11 PM UTC (23:00)

## Testing Locally First

Before setting up automation, test locally:

```bash
# Test the automated pipeline
python automated_training.py
```

You should see:
```
================================================================================
AUTOMATED TRAINING PIPELINE
Started: 2025-11-15 18:00:00
================================================================================

[1/4] Collecting fresh market data...
[OK] Collected 6500 samples

[2/4] Training ML models...
[OK] Models trained and saved

[3/4] Uploading to S3...
[OK] All files uploaded to S3

[4/4] Summary
================================================================================
AUTOMATED TRAINING COMPLETE!
Finished: 2025-11-15 18:05:23
================================================================================
```

## Dashboard Auto-Download

Make your dashboard always use the latest models from S3:

Add this to the top of `dashboard.py` (after imports):

```python
# Auto-download latest models from S3
@st.cache_resource
def download_latest_models():
    """Download latest models from S3 on first run"""
    import glob
    from s3_storage import S3StorageManager

    # Check if we have any models locally
    local_models = glob.glob('spy_trading_model*.pkl')

    if len(local_models) == 0:
        st.info("📥 Downloading latest models from S3...")
        storage = S3StorageManager()
        storage.download_models(decompress=True)
        st.success("✓ Models downloaded!")

    return True

# Download models on startup
download_latest_models()
```

This way:
- Dashboard checks for models on startup
- Downloads from S3 if not present
- Always uses latest trained version

## Cost Breakdown

**GitHub Actions (Recommended):**
- Cost: $0.00 (2,000 free minutes/month)
- Your usage: ~20 minutes/month (4 runs × 5 min)
- Reliability: High
- Setup time: 5 minutes

**AWS Lambda:**
- Cost: $0.00 (well within free tier)
- Your usage: ~4 invocations/month
- Reliability: Very high
- Setup time: 20 minutes

**AWS S3 Storage:**
- Cost: $0.00 (free tier covers it)
- Your usage: ~1 MB compressed
- Already set up!

**Total monthly cost: $0.00**

## Troubleshooting

### "Workflow doesn't appear in Actions tab"

Make sure `.github/workflows/scheduled_training.yml` is in your repo:
```bash
git add .github/workflows/scheduled_training.yml
git commit -m "Add workflow"
git push
```

### "Secrets not found"

Double-check secret names exactly match:
- `TRADIER_API_TOKEN` (not TRADIER_TOKEN)
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_REGION`
- `S3_BUCKET_NAME`

### "Training fails with import error"

Make sure `requirements.txt` is in repo:
```bash
git add requirements.txt
git commit -m "Add requirements"
git push
```

### "S3 upload fails"

1. Check AWS credentials in GitHub secrets
2. Verify S3 bucket name is correct
3. Check IAM permissions include S3 access

## Next Steps

1. **Test locally:** `python automated_training.py`
2. **Push to GitHub** (if using GitHub Actions)
3. **Add secrets** to GitHub repo
4. **Run workflow manually** to test
5. **Let it run automatically** every week!

Your ML trading system now trains itself automatically. You can focus on trading while the models stay updated with fresh data!

## Support

If you need help:
- GitHub Actions: [docs.github.com/actions](https://docs.github.com/actions)
- AWS Lambda: See `lambda_deployment.md`
- General questions: Check workflow logs for errors
