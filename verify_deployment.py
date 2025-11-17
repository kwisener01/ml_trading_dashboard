"""
Streamlit Cloud Deployment Verification Script
Checks if your ML Trading Dashboard is deployed and accessible
"""
import requests
import sys
from datetime import datetime

def check_streamlit_deployment():
    """Check if Streamlit app is deployed and accessible"""

    print("="*80)
    print("STREAMLIT CLOUD DEPLOYMENT VERIFICATION")
    print("="*80)
    print()

    # Possible URLs for the app
    urls = [
        "https://kwisener01-ml-trading-dashboard.streamlit.app",
        "https://ml-trading-dashboard.streamlit.app",
        "https://share.streamlit.io/kwisener01/ml_trading_dashboard/main/dashboard.py"
    ]

    deployment_found = False

    for url in urls:
        print(f"Checking: {url}")
        try:
            response = requests.get(url, timeout=10, allow_redirects=True)

            if response.status_code == 200:
                # Check if it's actually the Streamlit app
                content = response.text.lower()

                if 'ml trading' in content or 'streamlit' in content:
                    print(f"  [OK] FOUND! App is deployed at: {url}")
                    print(f"  Status: {response.status_code}")
                    print(f"  Size: {len(response.content)} bytes")
                    deployment_found = True
                    break
                else:
                    print(f"  [WARN] URL responds but content doesn't match expected app")

            elif response.status_code == 404:
                print(f"  [FAIL] Not found (404) - App not deployed at this URL")

            else:
                print(f"  [WARN] Unexpected status: {response.status_code}")

        except requests.exceptions.Timeout:
            print(f"  [TIMEOUT] App may be sleeping (try again)")

        except requests.exceptions.ConnectionError:
            print(f"  [FAIL] Connection error - URL doesn't exist")

        except Exception as e:
            print(f"  [ERROR] {str(e)}")

        print()

    print("="*80)

    if deployment_found:
        print("[SUCCESS] DEPLOYMENT VERIFIED!")
        print()
        print("Your app is live and accessible.")
        print()
        print("Next steps:")
        print("1. Test prediction generation")
        print("2. Verify models download from S3")
        print("3. Test on mobile device")
        print("4. Add to home screen for quick access")

    else:
        print("[NOTICE] APP NOT DEPLOYED YET")
        print()
        print("To deploy your app:")
        print()
        print("1. Go to: https://share.streamlit.io")
        print("2. Sign in with GitHub")
        print("3. Click 'New app'")
        print("4. Configure:")
        print("   - Repository: kwisener01/ml_trading_dashboard")
        print("   - Branch: main")
        print("   - Main file: dashboard.py")
        print("5. Click 'Deploy'")
        print("6. Add secrets (Settings → Secrets)")
        print()
        print("See STREAMLIT_DEPLOYMENT_CHECKLIST.md for detailed instructions")

    print("="*80)

    return deployment_found


def check_github_repo():
    """Check if GitHub repo is accessible"""
    print("\nChecking GitHub repository...")

    try:
        response = requests.get("https://github.com/kwisener01/ml_trading_dashboard", timeout=10)
        if response.status_code == 200:
            print("  [OK] GitHub repo is accessible")
            return True
        else:
            print(f"  [WARN] GitHub repo status: {response.status_code}")
            return False
    except Exception as e:
        print(f"  [ERROR] Error accessing GitHub: {str(e)}")
        return False


def check_s3_connection():
    """Check if S3 bucket is accessible"""
    print("\nChecking AWS S3 connection...")

    try:
        from s3_storage import S3StorageManager
        storage = S3StorageManager()
        files = storage.list_files('models/')

        if files:
            print(f"  [OK] S3 connection working! Found {len(files)} model files")

            # Show latest file
            latest = sorted(files, key=lambda x: x['last_modified'], reverse=True)[0]
            print(f"  Latest upload: {latest['last_modified']}")
            return True
        else:
            print("  [WARN] S3 connection works but no models found")
            return False

    except Exception as e:
        print(f"  [ERROR] S3 error: {str(e)}")
        return False


def main():
    """Run all verification checks"""
    print(f"\nVerification Time: {datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}\n")

    # Check GitHub
    github_ok = check_github_repo()

    # Check S3
    s3_ok = check_s3_connection()

    # Check Streamlit deployment
    print()
    streamlit_ok = check_streamlit_deployment()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"GitHub Repository: {'[OK]' if github_ok else '[ISSUE]'}")
    print(f"AWS S3 Storage:    {'[OK]' if s3_ok else '[ISSUE]'}")
    print(f"Streamlit Cloud:   {'[DEPLOYED]' if streamlit_ok else '[NOT DEPLOYED]'}")
    print("="*80)

    if github_ok and s3_ok and not streamlit_ok:
        print("\n[READY] Everything is ready - just need to deploy to Streamlit Cloud!")
        print("        Run: python verify_deployment.py --help-deploy")

    return 0 if streamlit_ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--help-deploy":
        print("\nQuick Deploy Instructions:")
        print("1. Visit: https://share.streamlit.io")
        print("2. Sign in with GitHub")
        print("3. Click 'New app'")
        print("4. Repository: kwisener01/ml_trading_dashboard")
        print("5. Branch: main")
        print("6. Main file: dashboard.py")
        print("7. Click 'Deploy'")
        print("\nSee STREAMLIT_DEPLOYMENT_CHECKLIST.md for full instructions\n")
    else:
        sys.exit(main())
