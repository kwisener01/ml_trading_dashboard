"""
Test Connections - Verify all secrets are working
Run this to check if Tradier API and AWS S3 are configured correctly
"""
import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 80)
print("CONNECTION TEST")
print("=" * 80)

# Test 1: Check environment variables
print("\n[1/4] Checking Environment Variables...")
tradier_token = os.getenv('TRADIER_API_TOKEN')
aws_key = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
s3_bucket = os.getenv('S3_BUCKET_NAME')

if tradier_token:
    print(f"  [OK] TRADIER_API_TOKEN found: {tradier_token[:4]}...{tradier_token[-4:]}")
else:
    print("  [ERROR] TRADIER_API_TOKEN missing")

if aws_key:
    print(f"  [OK] AWS_ACCESS_KEY_ID found: {aws_key[:4]}...{aws_key[-4:]}")
else:
    print("  [ERROR] AWS_ACCESS_KEY_ID missing")

if aws_secret:
    print(f"  [OK] AWS_SECRET_ACCESS_KEY found: {aws_secret[:4]}...****")
else:
    print("  [ERROR] AWS_SECRET_ACCESS_KEY missing")

if aws_region:
    print(f"  [OK] AWS_REGION: {aws_region}")
else:
    print("  [ERROR] AWS_REGION missing")

if s3_bucket:
    print(f"  [OK] S3_BUCKET_NAME: {s3_bucket}")
else:
    print("  [ERROR] S3_BUCKET_NAME missing")

# Test 2: Test Tradier API connection
print("\n[2/4] Testing Tradier API Connection...")
try:
    from data_collector import TradierDataCollector
    collector = TradierDataCollector()

    # Try to get current quote
    quote = collector.get_quote('SPY')
    if quote and 'last' in quote:
        print(f"  [OK] Tradier API working! SPY: ${quote['last']:.2f}")
    else:
        print("  [ERROR] Tradier API returned unexpected data")
except Exception as e:
    print(f"  [ERROR] Tradier API failed: {e}")

# Test 3: Test AWS S3 connection
print("\n[3/4] Testing AWS S3 Connection...")
try:
    from s3_storage import S3StorageManager
    storage = S3StorageManager()

    # Try to list files
    files = storage.list_files()
    print(f"  [OK] S3 connection working! Found {len(files)} files in bucket")

    # Show some files
    if len(files) > 0:
        print(f"  Latest files:")
        for f in files[:3]:
            size_mb = f['size'] / (1024 * 1024)
            print(f"    - {f['key']} ({size_mb:.2f} MB)")

except Exception as e:
    print(f"  [ERROR] S3 connection failed: {e}")

# Test 4: Test Model Loading
print("\n[4/4] Testing Model Files...")
try:
    import glob
    models = glob.glob('spy_trading_model*.pkl')

    if len(models) > 0:
        print(f"  [OK] Found {len(models)} model files locally")
        # Show latest models
        for m in models[:3]:
            print(f"    - {m}")
    else:
        print("  [WARN] No models found locally (will download from S3)")

except Exception as e:
    print(f"  [ERROR] Model check failed: {e}")

# Summary
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

all_good = tradier_token and aws_key and aws_secret and aws_region and s3_bucket

if all_good:
    print("[OK] All secrets configured!")
    print("[OK] Ready for Streamlit Cloud deployment")
    print("\nNext steps:")
    print("1. Push code to GitHub: git push")
    print("2. Add these same secrets to Streamlit Cloud")
    print("3. Deploy and test!")
else:
    print("[ERROR] Some secrets are missing")
    print("Check your .env file or Streamlit secrets")

print("=" * 80)
