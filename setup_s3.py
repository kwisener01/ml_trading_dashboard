"""
Interactive S3 Setup Helper
Guides you through AWS S3 configuration step-by-step
"""
import os
from dotenv import load_dotenv, set_key

load_dotenv()

def setup_aws_credentials():
    """Interactive setup for AWS credentials"""
    print("="*80)
    print("AWS S3 SETUP WIZARD")
    print("="*80)
    print("\nThis will help you configure AWS S3 for storing ML models and data.")
    print("\nPrerequisites:")
    print("  1. AWS account created")
    print("  2. IAM user created with S3 access")
    print("  3. Access keys downloaded")
    print("\nIf you haven't done these yet, see AWS_S3_SETUP.md for instructions.\n")

    input("Press Enter when ready to continue...")

    # Check if .env exists
    env_file = '.env'
    if not os.path.exists(env_file):
        print(f"\n[!] Creating {env_file} file...")
        with open(env_file, 'w') as f:
            f.write("# AWS S3 Configuration\n")

    print("\n" + "="*80)
    print("STEP 1: AWS Credentials")
    print("="*80)

    # Get credentials
    access_key = input("\nEnter your AWS Access Key ID (starts with AKIA): ").strip()
    if not access_key.startswith('AKIA'):
        print("[!] Warning: Access Key ID should start with 'AKIA'")

    secret_key = input("Enter your AWS Secret Access Key: ").strip()

    print("\n" + "="*80)
    print("STEP 2: AWS Region")
    print("="*80)
    print("\nCommon regions:")
    print("  us-east-1      - US East (N. Virginia) - Cheapest, most services")
    print("  us-west-2      - US West (Oregon)")
    print("  eu-west-1      - Europe (Ireland)")
    print("  ap-southeast-1 - Asia Pacific (Singapore)")

    region = input("\nEnter AWS region [us-east-1]: ").strip() or 'us-east-1'

    print("\n" + "="*80)
    print("STEP 3: S3 Bucket Name")
    print("="*80)
    print("\nBucket name must be:")
    print("  - Globally unique")
    print("  - 3-63 characters")
    print("  - Lowercase letters, numbers, hyphens only")
    print("  - No spaces or special characters")

    default_bucket = f"ml-trading-models-{os.getenv('USER', 'user').lower()}"
    bucket_name = input(f"\nEnter S3 bucket name [{default_bucket}]: ").strip() or default_bucket

    # Save to .env
    print("\n" + "="*80)
    print("SAVING CONFIGURATION")
    print("="*80)

    try:
        set_key(env_file, 'AWS_ACCESS_KEY_ID', access_key)
        set_key(env_file, 'AWS_SECRET_ACCESS_KEY', secret_key)
        set_key(env_file, 'AWS_REGION', region)
        set_key(env_file, 'S3_BUCKET_NAME', bucket_name)

        print(f"\n✓ Credentials saved to {env_file}")
        print("\n[!] IMPORTANT: Never commit .env to Git!")

    except Exception as e:
        print(f"\n✗ Error saving credentials: {e}")
        return False

    return True, bucket_name, region


def test_connection():
    """Test AWS S3 connection"""
    print("\n" + "="*80)
    print("TESTING CONNECTION")
    print("="*80)

    try:
        from s3_storage import S3StorageManager

        print("\nInitializing S3 client...")
        storage = S3StorageManager()

        print("Creating bucket if it doesn't exist...")
        storage.create_bucket_if_not_exists()

        print("\nListing existing files...")
        files = storage.list_files()
        print(f"Found {len(files)} files in bucket")

        print("\n✓ Connection successful!")
        return True

    except Exception as e:
        print(f"\n✗ Connection failed: {e}")
        print("\nCommon issues:")
        print("  - Check your Access Key ID and Secret Key")
        print("  - Verify IAM user has S3 permissions")
        print("  - Ensure bucket name is globally unique")
        return False


def upload_initial_data():
    """Upload models and data to S3"""
    print("\n" + "="*80)
    print("UPLOADING TO S3")
    print("="*80)

    response = input("\nUpload models and training data to S3? (y/n): ").lower()
    if response != 'y':
        print("Skipping upload.")
        return

    try:
        from s3_storage import S3StorageManager
        import glob

        storage = S3StorageManager()

        # Count files
        model_files = glob.glob('spy_trading_model*.pkl')
        print(f"\nFound {len(model_files)} model files")

        if model_files:
            print("\nUploading models (with compression)...")
            storage.upload_models(compress=True)

        # Upload training data
        if os.path.exists('spy_ml_dataset.csv'):
            print("\nUploading training data...")
            storage.upload_training_data('spy_ml_dataset.csv', compress=True)

        # Upload prediction log
        if os.path.exists('prediction_log.csv'):
            print("\nUploading prediction log...")
            storage.upload_file('prediction_log.csv', 'training_data/prediction_log.csv', compress=True)

        print("\n✓ Upload complete!")

        # Show summary
        print("\n" + "="*80)
        print("S3 BUCKET CONTENTS")
        print("="*80)

        files = storage.list_files()
        total_size = sum(f['size'] for f in files)

        print(f"\nTotal files: {len(files)}")
        print(f"Total size: {total_size / 1024 / 1024:.2f} MB")
        print(f"Monthly cost: ~${total_size / 1024 / 1024 * 0.023:.4f}")

        print("\nBy category:")
        models = [f for f in files if f['key'].startswith('models/')]
        data = [f for f in files if f['key'].startswith('training_data/')]

        print(f"  Models: {len(models)} files ({sum(f['size'] for f in models) / 1024 / 1024:.2f} MB)")
        print(f"  Data: {len(data)} files ({sum(f['size'] for f in data) / 1024 / 1024:.2f} MB)")

    except Exception as e:
        print(f"\n✗ Upload failed: {e}")
        return


def main():
    """Main setup flow"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "AWS S3 SETUP WIZARD" + " "*39 + "║")
    print("╚" + "="*78 + "╝")
    print("\n")

    # Check if already configured
    if os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('S3_BUCKET_NAME'):
        print("[!] AWS credentials already configured in .env")
        response = input("\nReconfigure? (y/n): ").lower()
        if response != 'y':
            print("\nSkipping configuration...")
            if test_connection():
                upload_initial_data()
            return

    # Setup credentials
    result = setup_aws_credentials()
    if not result:
        print("\n✗ Setup failed. Please try again.")
        return

    # Reload environment
    load_dotenv(override=True)

    # Test connection
    if test_connection():
        upload_initial_data()
    else:
        print("\n[!] Please fix the connection issues and run this script again.")
        return

    print("\n" + "="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print("\nYour ML models are now backed up in AWS S3!")
    print("\nNext steps:")
    print("  1. View your bucket: https://console.aws.amazon.com/s3")
    print("  2. Download models anytime: python -c \"from s3_storage import S3StorageManager; S3StorageManager().download_models()\"")
    print("  3. Deploy to Streamlit Cloud with S3 backing")
    print("\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n\n✗ Unexpected error: {e}")
        print("\nFor help, see AWS_S3_SETUP.md")
