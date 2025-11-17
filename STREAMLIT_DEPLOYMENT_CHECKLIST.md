# Streamlit Cloud Deployment Checklist

Quick checklist to deploy your ML Trading Dashboard to Streamlit Cloud.

## Pre-Deployment Checklist

✅ **1. GitHub Repository**
- [x] Code pushed to: `https://github.com/kwisener01/ml_trading_dashboard`
- [x] Branch: `main`
- [x] Main file: `dashboard.py`
- [x] Requirements: `requirements.txt`
- [x] `.env` file in `.gitignore` (secrets protected)

✅ **2. AWS S3 Setup**
- [x] S3 bucket created: `ml-trading-models-kwise`
- [x] Models uploaded: 25 files
- [x] IAM credentials configured
- [x] S3 connection tested and working

✅ **3. GitHub Actions**
- [x] Workflow file: `.github/workflows/scheduled_training.yml`
- [x] Scheduled: Monday-Friday at 6:00 AM EST
- [x] Last run: Completed successfully
- [ ] Secrets added to GitHub (verify below)

## GitHub Secrets (Required for Actions)

Go to: https://github.com/kwisener01/ml_trading_dashboard/settings/secrets/actions

Add these secrets:

| Secret Name | Value | Status |
|-------------|-------|--------|
| `TRADIER_API_TOKEN` | Your Tradier token | ❓ Verify |
| `AWS_ACCESS_KEY_ID` | Your AWS access key | ❓ Verify |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | ❓ Verify |
| `AWS_REGION` | `us-east-1` | ❓ Verify |
| `S3_BUCKET_NAME` | `ml-trading-models-kwise` | ❓ Verify |

## Streamlit Cloud Deployment (5 Minutes)

### Step 1: Sign In
1. Go to: https://share.streamlit.io
2. Click "Sign in with GitHub"
3. Authorize Streamlit to access your repos

### Step 2: Create New App
1. Click "New app" button (top right)
2. Fill in deployment details:
   - **Repository:** `kwisener01/ml_trading_dashboard`
   - **Branch:** `main`
   - **Main file path:** `dashboard.py`
   - **App URL:** (optional) customize if you want

3. Click "Deploy!"

### Step 3: Add Secrets
**IMPORTANT:** Your app will fail without these secrets!

1. Once deployed, click your app in the dashboard
2. Click ⚙️ **Settings** (bottom left of app page)
3. Click **Secrets** tab
4. Paste this (replace with your actual values):

```toml
# Tradier API
TRADIER_API_TOKEN = "your_tradier_api_token_here"

# AWS S3 Credentials
AWS_ACCESS_KEY_ID = "your_aws_access_key_id_here"
AWS_SECRET_ACCESS_KEY = "your_aws_secret_access_key_here"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "ml-trading-models-kwise"
```

**IMPORTANT:** Replace the placeholder values with your actual credentials from your `.env` file.

5. Click **Save**
6. App will automatically restart

### Step 4: Test Your App

Your app will be live at:
```
https://kwisener01-ml-trading-dashboard.streamlit.app
```

**Test checklist:**
- [ ] App loads without errors
- [ ] API token shows as loaded in sidebar
- [ ] Click "Generate Prediction" button
- [ ] Models download from S3 automatically
- [ ] Prediction displays with chart
- [ ] Vanna levels visible on chart
- [ ] Test on mobile browser

## Troubleshooting

### App shows "ModuleNotFoundError"
**Fix:** Make sure `requirements.txt` is in your repo and pushed to GitHub.

```bash
git add requirements.txt
git commit -m "fix: ensure requirements.txt is in repo"
git push
```

### Models not downloading from S3
**Check:**
1. Secrets are added to Streamlit Cloud (Settings → Secrets)
2. AWS credentials are correct (no extra spaces)
3. S3 bucket name is correct

**Debug:** Add to `dashboard.py` temporarily:
```python
st.write("AWS_ACCESS_KEY_ID:", os.getenv('AWS_ACCESS_KEY_ID', 'NOT SET')[:8] + "...")
```

### "No API token found"
**Fix:** Add `TRADIER_API_TOKEN` to Streamlit Secrets (Settings → Secrets)

### App is slow/sleeping
**Normal behavior:**
- First load: 10-15 seconds (downloading models)
- After inactivity: ~30 seconds to wake up (free tier)
- Prediction: 3-5 seconds (API calls)

## Post-Deployment

### Daily Workflow
1. Open app on any device: `https://kwisener01-ml-trading-dashboard.streamlit.app`
2. Click "Generate Prediction"
3. Models auto-download from S3 (latest from 6 AM training)
4. Get fresh predictions with live market data

### Updating Your App
```bash
# Make changes to code
git add .
git commit -m "your changes"
git push

# Streamlit auto-detects and redeploys in ~2 minutes
```

### Add to Mobile Home Screen
**iOS:**
1. Open app in Safari
2. Tap Share → Add to Home Screen
3. Now accessible like a native app!

**Android:**
1. Open app in Chrome
2. Menu → Add to Home Screen

## Complete System Flow

```
Every Day at 6:00 AM EST:
  GitHub Actions runs automated_training.py
    ↓
  Trains models with latest 180 days SPY data
    ↓
  Uploads compressed models to S3
    ↓
  (Models ready in cloud)

Anytime You Open Dashboard:
  Open: kwisener01-ml-trading-dashboard.streamlit.app
    ↓
  Streamlit app downloads latest models from S3
    ↓
  Click "Generate Prediction"
    ↓
  Gets live SPY price, VIX, options data
    ↓
  Calculates current Vanna levels
    ↓
  📊 Shows fresh prediction!
```

## Costs
- Streamlit Cloud: **FREE**
- GitHub Actions: **FREE** (2,000 min/month)
- AWS S3: **< $0.01/month**
- **Total: $0.00/month**

## Support Links
- Streamlit Docs: https://docs.streamlit.io/streamlit-community-cloud
- Your App (after deploy): https://kwisener01-ml-trading-dashboard.streamlit.app
- GitHub Repo: https://github.com/kwisener01/ml_trading_dashboard
- S3 Console: https://console.aws.amazon.com/s3/buckets/ml-trading-models-kwise

---

**Ready to deploy?** Go to https://share.streamlit.io and follow Step 2 above!
