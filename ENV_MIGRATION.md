# 🔐 API Token Migration to .env File

## What Changed

All scripts now use a `.env` file for API token storage instead of hardcoding.

### Benefits
✅ **More Secure** - Token never goes in version control
✅ **Easier** - Set token once, use everywhere
✅ **Professional** - Industry standard practice
✅ **Cleaner** - No token in your code files

---

## Quick Setup

### One-Time Setup (30 seconds)

```bash
# 1. Copy template
cp .env.example .env

# 2. Edit .env file
# Add your actual API token:
TRADIER_API_TOKEN=your_actual_token_here

# 3. Done! All scripts now work automatically
```

---

## Files Updated

### Core Scripts
- ✅ `data_collector.py` - Reads from .env
- ✅ `collect_more_data.py` - Reads from .env
- ✅ `predictor.py` - Reads from .env
- ✅ `tradier_options_test.py` - Reads from .env
- ✅ `dashboard.py` - Reads from .env (with UI indicator)

### New Files
- ✅ `.env.example` - Template file (safe to commit)
- ✅ `.gitignore` - Protects your .env file
- ✅ `ENV_SETUP_GUIDE.md` - Complete setup instructions

### Updated Dependencies
- ✅ `requirements_ml.txt` - Added `python-dotenv==1.0.0`

### Updated Documentation
- ✅ `START_HERE.md` - Step 0 now shows .env setup
- ✅ `QUICK_START_GUIDE.md` - Updated setup instructions

---

## How It Works

### Before (Old Way)
```python
# Had to edit every file
API_TOKEN = "YOUR_TRADIER_API_TOKEN_HERE"  # ← Edit this
```

**Problems:**
- ❌ Easy to commit token by accident
- ❌ Must edit multiple files
- ❌ Token visible in code
- ❌ Hard to change

### After (New Way)
```python
# Automatically loads from .env
from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.getenv('TRADIER_API_TOKEN')
```

**.env file:**
```env
TRADIER_API_TOKEN=abc123yourtoken789
```

**Benefits:**
- ✅ Never committed (in .gitignore)
- ✅ One file to edit
- ✅ Token hidden
- ✅ Easy to change

---

## Usage Examples

### All Scripts Auto-Load

```bash
# Just run any script - token loads automatically
python data_collector.py
python collect_more_data.py
python predictor.py
python tradier_options_test.py
```

### Dashboard Shows Status

When you run `streamlit run dashboard.py`:
- ✅ Green: "API token loaded from .env"
- ⚠️ Yellow: "No .env file found" (with input field)

### Error Handling

If `.env` missing or token not set:
```
❌ ERROR: TRADIER_API_TOKEN not found!

📋 SETUP REQUIRED:
1. Copy .env.example to .env
2. Edit .env and add your API token
3. Run this script again
```

---

## Migration Checklist

If you were using the old version:

- [ ] Install updated dependency: `pip install python-dotenv`
- [ ] Copy `.env.example` to `.env`
- [ ] Add your API token to `.env`
- [ ] Delete any tokens from code files (optional - they're ignored now)
- [ ] Test with: `python tradier_options_test.py`
- [ ] Verify `.env` is in `.gitignore`

---

## Security Best Practices

### DO ✅
- Keep `.env` file local only
- Add `.env` to `.gitignore`
- Commit `.env.example` (without real token)
- Use different tokens for dev/production
- Rotate tokens periodically

### DON'T ❌
- Commit `.env` to Git
- Share `.env` file
- Email tokens
- Screenshot tokens
- Hardcode tokens in code

---

## Backwards Compatibility

### Still Works (But Not Recommended)

```bash
# Environment variable still works
export TRADIER_API_TOKEN='your_token'
python data_collector.py
```

The scripts check:
1. First: `.env` file
2. Fallback: Environment variable
3. Error: If neither exists

---

## Troubleshooting

### "TOKEN not found" Error

**Check 1:** Does `.env` file exist?
```bash
ls -la .env
```

**Check 2:** Is token in the file?
```bash
cat .env
# Should show: TRADIER_API_TOKEN=...
```

**Check 3:** Is file in correct directory?
```bash
pwd  # Should be your project directory
```

### Scripts Work But Dashboard Doesn't

The dashboard might be cached. Restart it:
```bash
# Stop dashboard (Ctrl+C)
# Start fresh
streamlit run dashboard.py
```

### "401 Unauthorized" Error

Token is set but wrong:
1. Double-check token in `.env`
2. No spaces around `=`
3. No quotes around token
4. Token not expired

**Correct format:**
```env
TRADIER_API_TOKEN=abc123xyz789
```

**Wrong formats:**
```env
TRADIER_API_TOKEN = abc123xyz789      ❌
TRADIER_API_TOKEN="abc123xyz789"      ❌
TRADIER_API_TOKEN='abc123xyz789'      ❌
```

---

## Testing Your Setup

### Quick Test

```bash
# Test token loads
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ Token found!' if os.getenv('TRADIER_API_TOKEN') else '❌ Token missing')"
```

### Full Test

```bash
# Should show SPY/QQQ option data
python tradier_options_test.py
```

---

## Advanced Options

### Multiple Environments

Create separate files:
```
.env.development
.env.production
```

Load specific one:
```python
load_dotenv('.env.production')
```

### Sandbox Mode

Add to `.env`:
```env
TRADIER_API_TOKEN=your_token
USE_SANDBOX=true
```

---

## Summary

**Old Way:** Edit 5+ files with your token
**New Way:** Edit 1 file (`.env`) once

**Result:** 
- 🔐 More secure
- 🚀 Easier to use
- 💼 Professional setup
- ✅ Industry standard

---

## Need Help?

📖 **Full Guide:** See `ENV_SETUP_GUIDE.md`
🚀 **Quick Start:** See `START_HERE.md`
🔧 **Problems:** See `TROUBLESHOOTING.md`

---

**Your API token is now secure and easy to manage!** 🎉
