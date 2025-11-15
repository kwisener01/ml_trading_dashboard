"""
S3 Storage Manager for ML Trading System
Handles uploading/downloading models and training data to/from AWS S3
"""
import boto3
from botocore.exceptions import ClientError
import os
from pathlib import Path
from dotenv import load_dotenv
import gzip
import shutil
from datetime import datetime

load_dotenv()


class S3StorageManager:
    """
    Manages storage of ML models and training data in AWS S3
    """

    def __init__(self, bucket_name=None):
        """Initialize S3 client"""
        self.bucket_name = bucket_name or os.getenv('S3_BUCKET_NAME', 'ml-trading-models')

        # Initialize S3 client with credentials from env
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_REGION', 'us-east-1')
        )

        print(f"[S3] Initialized with bucket: {self.bucket_name}")

    def create_bucket_if_not_exists(self):
        """Create S3 bucket if it doesn't exist"""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            print(f"[S3] Bucket '{self.bucket_name}' already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                region = os.getenv('AWS_REGION', 'us-east-1')
                if region == 'us-east-1':
                    self.s3_client.create_bucket(Bucket=self.bucket_name)
                else:
                    self.s3_client.create_bucket(
                        Bucket=self.bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region}
                    )
                print(f"[S3] Created bucket: {self.bucket_name}")
            else:
                raise

    def upload_file(self, local_path, s3_key, compress=False):
        """
        Upload file to S3

        Args:
            local_path: Path to local file
            s3_key: S3 key (path) for the file
            compress: Whether to gzip compress before upload
        """
        try:
            file_to_upload = local_path

            if compress:
                # Compress file before upload
                compressed_path = f"{local_path}.gz"
                with open(local_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_to_upload = compressed_path
                s3_key = f"{s3_key}.gz"

            # Upload to S3
            self.s3_client.upload_file(file_to_upload, self.bucket_name, s3_key)

            # Get file size
            file_size = os.path.getsize(file_to_upload)
            size_mb = file_size / (1024 * 1024)

            print(f"[S3] OK Uploaded: {s3_key} ({size_mb:.2f} MB)")

            # Clean up compressed file
            if compress and os.path.exists(compressed_path):
                os.remove(compressed_path)

            return True

        except Exception as e:
            print(f"[S3] ERROR Upload failed: {e}")
            return False

    def download_file(self, s3_key, local_path, decompress=False):
        """
        Download file from S3

        Args:
            s3_key: S3 key (path) for the file
            local_path: Path to save file locally
            decompress: Whether to decompress after download
        """
        try:
            # Create directory if needed
            Path(local_path).parent.mkdir(parents=True, exist_ok=True)

            # Download from S3
            download_path = local_path
            if decompress and s3_key.endswith('.gz'):
                download_path = f"{local_path}.gz"

            self.s3_client.download_file(self.bucket_name, s3_key, download_path)

            # Decompress if needed
            if decompress and download_path.endswith('.gz'):
                with gzip.open(download_path, 'rb') as f_in:
                    with open(local_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(download_path)  # Remove compressed file

            file_size = os.path.getsize(local_path)
            size_mb = file_size / (1024 * 1024)

            print(f"[S3] OK Downloaded: {s3_key} ({size_mb:.2f} MB)")
            return True

        except Exception as e:
            print(f"[S3] ERROR Download failed: {e}")
            return False

    def list_files(self, prefix=''):
        """List files in S3 bucket with given prefix"""
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )

            if 'Contents' in response:
                files = []
                for obj in response['Contents']:
                    files.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                return files
            else:
                return []

        except Exception as e:
            print(f"[S3] ERROR List failed: {e}")
            return []

    def delete_file(self, s3_key):
        """Delete file from S3"""
        try:
            self.s3_client.delete_object(Bucket=self.bucket_name, Key=s3_key)
            print(f"[S3] OK Deleted: {s3_key}")
            return True
        except Exception as e:
            print(f"[S3] ERROR Delete failed: {e}")
            return False

    def upload_models(self, model_prefix='spy_trading_model', compress=True):
        """
        Upload all model files to S3

        Args:
            model_prefix: Prefix for model files (default: 'spy_trading_model')
            compress: Whether to compress models before upload
        """
        import glob

        model_files = glob.glob(f"{model_prefix}*.pkl")

        if not model_files:
            print(f"[S3] No model files found with prefix: {model_prefix}")
            return

        print(f"[S3] Uploading {len(model_files)} model files...")

        success_count = 0
        for model_file in model_files:
            s3_key = f"models/{model_file}"
            if self.upload_file(model_file, s3_key, compress=compress):
                success_count += 1

        print(f"[S3] OK Uploaded {success_count}/{len(model_files)} models")

    def download_models(self, model_prefix='spy_trading_model', decompress=True):
        """
        Download all models from S3

        Args:
            model_prefix: Prefix for model files
            decompress: Whether to decompress after download
        """
        files = self.list_files(prefix='models/')

        if not files:
            print("[S3] No models found in S3")
            return

        print(f"[S3] Downloading {len(files)} model files...")

        success_count = 0
        for file_info in files:
            s3_key = file_info['key']
            local_filename = os.path.basename(s3_key)

            # Remove .gz extension if decompressing
            if decompress and local_filename.endswith('.gz'):
                local_filename = local_filename[:-3]

            if self.download_file(s3_key, local_filename, decompress=decompress):
                success_count += 1

        print(f"[S3] OK Downloaded {success_count}/{len(files)} models")

    def upload_training_data(self, data_file, compress=True):
        """Upload training data CSV to S3"""
        if not os.path.exists(data_file):
            print(f"[S3] File not found: {data_file}")
            return False

        s3_key = f"training_data/{os.path.basename(data_file)}"
        return self.upload_file(data_file, s3_key, compress=compress)

    def download_training_data(self, data_file, decompress=True):
        """Download training data from S3"""
        s3_key = f"training_data/{os.path.basename(data_file)}"
        if decompress and not s3_key.endswith('.gz'):
            s3_key = f"{s3_key}.gz"

        return self.download_file(s3_key, data_file, decompress=decompress)


def main():
    """Example usage"""
    print("="*80)
    print("S3 Storage Manager - Example Usage")
    print("="*80)

    # Initialize storage manager
    storage = S3StorageManager()

    # Create bucket if needed
    storage.create_bucket_if_not_exists()

    print("\n--- Upload Models ---")
    storage.upload_models(compress=True)

    print("\n--- Upload Training Data ---")
    if os.path.exists('spy_ml_dataset.csv'):
        storage.upload_training_data('spy_ml_dataset.csv', compress=True)

    print("\n--- List Files ---")
    models = storage.list_files(prefix='models/')
    print(f"Models in S3: {len(models)}")
    for model in models[:5]:  # Show first 5
        size_mb = model['size'] / (1024 * 1024)
        print(f"  - {model['key']} ({size_mb:.2f} MB)")

    data_files = storage.list_files(prefix='training_data/')
    print(f"\nTraining data in S3: {len(data_files)}")
    for data in data_files:
        size_mb = data['size'] / (1024 * 1024)
        print(f"  - {data['key']} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
