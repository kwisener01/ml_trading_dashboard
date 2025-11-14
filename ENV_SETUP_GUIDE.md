# 🔐 SECURE API TOKEN SETUP

## Why Use .env File?

✅ **Security**: Never commit API tokens to version control
✅ **Convenience**: Set once, use everywhere
✅ **Best Practice**: Industry standard for sensitive credentials
✅ **Easy Updates**: Change token in one place

---

## Quick Setup (30 seconds)

### Step 1: Create .env File

Copy the example file:

```bash
# Mac/Linux
cp .env.example .env

# Windows Command Prompt
copy .env.example .env

# Windows PowerShell
Copy-Item .env.example .env
```

### Step 2: Add Your API Token

Open `.env` in any text editor and replace `your_api_token_here` with your actual token:

```env
# Before
TRADIER_API_TOKEN=your_api_token_here

# After
TRADIER_API_TOKEN=aBcDeFgHiJkLmNoPqRsTuVwXyZ123456789
```

### Step 3: Save and You're Done!

All scripts now automatically read from `.env` file.

---

## Getting Your Tradier API Token

1. Go to https://documentation.tradier.com/brokerage-api/getting-started
2. Sign up for a developer account
3. Generate an API token
4. Copy the token (long string of letters/numbers)
5. Paste into your `.env` file

---

## Using the .env File

### All Scripts Automatically Load It

Every script now includes:
```python
from dotenv import load_dotenv
load_dotenv()

API_TOKEN = os.getenv('TRADIER_API_TOKEN')
```

No need to edit multiple files!

### Test It Works

```bash
python tradier_options_test.py
```

If you see:
- ✅ Data returns → Token works!
- ❌ "TOKEN not found" → Check .env file exists
- ❌ "401 Unauthorized" → Token is wrong

---

## File Structure

After setup, your directory should look like:

```
your-project/
├── .env                          ← Your actual token (DO NOT COMMIT!)
├── .env.example                  ← Template (safe to commit)
├── .gitignore                    ← Should include .env
├── data_collector.py
├── predictor.py
└── ... other files
```

---

## Security Best Practices

### ✅ DO:
- Add `.env` to `.gitignore`
- Keep `.env` file local only
- Never commit `.env` to Git
- Use different tokens for dev/production
- Rotate tokens periodically

### ❌ DON'T:
- Commit `.env` file to version control
- Share your `.env` file with others
- Put tokens in code comments
- Email tokens in plain text
- Post tokens in screenshots

---

## .gitignore Setup

Create or update `.gitignore` file:

```gitignore
# Environment variables
.env

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Data files (optional - comment out if you want to commit data)
*.csv
*.pkl
*.json

# IDE
.vscode/
.idea/
*.swp
```

---

## Alternative Methods (Not Recommended)

### Method 1: Environment Variable (Session Only)

```bash
# Mac/Linux
export TRADIER_API_TOKEN='your_token_here'

# Windows PowerShell
$env:TRADIER_API_TOKEN = 'your_token_here'

# Windows Command Prompt
set TRADIER_API_TOKEN=your_token_here
```

**Cons:** Lost when terminal closes

### Method 2: Hardcode in Files (DON'T DO THIS)

```python
API_TOKEN = "your_token_here"  # ❌ BAD - visible in code
```

**Cons:** 
- Visible in version control
- Must update multiple files
- Security risk

---

## Troubleshooting

### "TRADIER_API_TOKEN not found"

**Solution 1:** Check .env file exists
```bash
# Check if file exists
ls -la .env      # Mac/Linux
dir .env         # Windows
```

**Solution 2:** Check .env content
```bash
# View file (Mac/Linux)
cat .env

# View file (Windows)
type .env
```

Should contain:
```
TRADIER_API_TOKEN=your_actual_token
```

**Solution 3:** Check file location
The `.env` file must be in the same directory where you run the scripts.

### "401 Unauthorized" After Setup

**Possible Issues:**
1. Token is incorrect (typo when copying)
2. Token expired (regenerate on Tradier)
3. Extra spaces in .env file
4. Using sandbox token with production URL (or vice versa)

**Fix:**
1. Copy token again carefully
2. Remove any spaces around the `=` sign
3. Make sure no quotes around the token value

**Correct format:**
```env
TRADIER_API_TOKEN=abc123xyz789
```

**Incorrect formats:**
```env
TRADIER_API_TOKEN = abc123xyz789      ❌ Spaces
TRADIER_API_TOKEN="abc123xyz789"      ❌ Quotes
TRADIER_API_TOKEN='abc123xyz789'      ❌ Quotes
```

### Scripts Still Ask for Token

**Possible Issues:**
1. .env not in the same directory
2. python-dotenv not installed
3. .env has wrong format

**Fix:**
```bash
# 1. Install dependency
pip install python-dotenv

# 2. Check .env location
pwd                    # See current directory
ls .env                # Check if .env exists here

# 3. Verify .env format
cat .env               # Should show: TRADIER_API_TOKEN=...
```

---

## Testing Your Setup

### Quick Test Script

Create `test_env.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

token = os.getenv('TRADIER_API_TOKEN')

if token:
    print("✅ SUCCESS!")
    print(f"   Token found: {token[:4]}...{token[-4:]}")
else:
    print("❌ FAILED!")
    print("   Check your .env file")
```

Run:
```bash
python test_env.py
```

### Test API Connection

```bash
python tradier_options_test.py
```

Should show SPY/QQQ option chains.

---

## Advanced: Multiple Environments

For dev/staging/production:

### Create Multiple .env Files

```
.env.development
.env.staging
.env.production
```

### Load Specific File

```python
from dotenv import load_dotenv

# Load specific environment
load_dotenv('.env.production')
```

### Or Use Environment Variable

```bash
export ENV=production
```

```python
import os
from dotenv import load_dotenv

env = os.getenv('ENV', 'development')
load_dotenv(f'.env.{env}')
```

---

## Optional: Sandbox Mode

For testing without real trading:

Add to your `.env`:

```env
TRADIER_API_TOKEN=your_token_here
USE_SANDBOX=true
```

Then in your code:

```python
use_sandbox = os.getenv('USE_SANDBOX', 'false').lower() == 'true'
base_url = "https://sandbox.tradier.com/v1" if use_sandbox else "https://api.tradier.com/v1"
```

---

## Summary

1. ✅ Copy `.env.example` to `.env`
2. ✅ Add your API token to `.env`
3. ✅ Add `.env` to `.gitignore`
4. ✅ Run any script - it automatically loads the token
5. ✅ Sleep well knowing your token is secure!

**One File, All Scripts. Simple. Secure. Professional.** 🔐
