# AWS S3 Storage Setup Guide

Complete guide to set up AWS S3 for storing ML trading models and training data.

## Table of Contents
1. [AWS Account Setup](#aws-account-setup)
2. [Create S3 Bucket](#create-s3-bucket)
3. [Create IAM User](#create-iam-user)
4. [Configure Credentials](#configure-credentials)
5. [Usage Examples](#usage-examples)
6. [Cost Estimation](#cost-estimation)

---

## AWS Account Setup

### 1. Create AWS Account (if you don't have one)
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Click "Create an AWS Account"
3. Follow the registration process
4. **Note**: Free tier includes 5GB S3 storage for 12 months

---

## Create S3 Bucket

### Option A: Using AWS Console (Recommended for first-time)

1. **Go to S3 Console**
   - Navigate to [console.aws.amazon.com/s3](https://console.aws.amazon.com/s3)

2. **Create Bucket**
   - Click "Create bucket"
   - **Bucket name**: `ml-trading-models-YOUR-NAME` (must be globally unique)
   - **Region**: Choose closest region (e.g., `us-east-1`)
   - **Block Public Access**: Keep all boxes CHECKED (security)
   - Click "Create bucket"

### Option B: Using CLI
```bash
aws s3 mb s3://ml-trading-models-YOUR-NAME --region us-east-1
```

---

## Create IAM User

### 1. Go to IAM Console
- Navigate to [console.aws.amazon.com/iam](https://console.aws.amazon.com/iam)

### 2. Create User
1. Click "Users" → "Add users"
2. **User name**: `ml-trading-bot`
3. **Access type**: Check "Programmatic access"
4. Click "Next: Permissions"

### 3. Set Permissions
**Option A: Attach Existing Policy (Simple)**
1. Click "Attach existing policies directly"
2. Search for `AmazonS3FullAccess`
3. Check the box next to it
4. Click "Next: Tags" → "Next: Review" → "Create user"

**Option B: Create Custom Policy (More Secure)**
1. Click "Create policy"
2. Choose JSON tab
3. Paste this policy:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "s3:PutObject",
                "s3:GetObject",
                "s3:DeleteObject",
                "s3:ListBucket"
            ],
            "Resource": [
                "arn:aws:s3:::ml-trading-models-YOUR-NAME",
                "arn:aws:s3:::ml-trading-models-YOUR-NAME/*"
            ]
        }
    ]
}
```

4. Name it: `MLTradingBotS3Access`
5. Create policy
6. Attach to your user

### 4. Save Credentials
**IMPORTANT**: Download and save the CSV file with:
- **Access Key ID**: `AKIA...`
- **Secret Access Key**: `wJa...`

**⚠️ WARNING**: You can only view the Secret Access Key once!

---

## Configure Credentials

### Method 1: Environment Variables (Recommended)

Add to your `.env` file:

```bash
# AWS S3 Configuration
AWS_ACCESS_KEY_ID=AKIA****************
AWS_SECRET_ACCESS_KEY=wJa*********************************
AWS_REGION=us-east-1
S3_BUCKET_NAME=ml-trading-models-YOUR-NAME
```

### Method 2: AWS CLI Configuration

```bash
aws configure
```

Enter when prompted:
- AWS Access Key ID
- AWS Secret Access Key
- Default region: `us-east-1`
- Default output format: `json`

### Method 3: Streamlit Secrets (for Streamlit Cloud)

Create `.streamlit/secrets.toml`:

```toml
# AWS Credentials
AWS_ACCESS_KEY_ID = "AKIA****************"
AWS_SECRET_ACCESS_KEY = "wJa*********************************"
AWS_REGION = "us-east-1"
S3_BUCKET_NAME = "ml-trading-models-YOUR-NAME"
```

---

## Usage Examples

### Upload Models to S3

```python
from s3_storage import S3StorageManager

# Initialize
storage = S3StorageManager()

# Create bucket (first time only)
storage.create_bucket_if_not_exists()

# Upload all models (compressed)
storage.upload_models(compress=True)

# Upload training data
storage.upload_training_data('spy_ml_dataset.csv', compress=True)
```

### Download Models from S3

```python
from s3_storage import S3StorageManager

storage = S3StorageManager()

# Download all models
storage.download_models(decompress=True)

# Download specific training data
storage.download_training_data('spy_ml_dataset.csv', decompress=True)
```

### List Files in S3

```python
storage = S3StorageManager()

# List all models
models = storage.list_files(prefix='models/')
for model in models:
    print(f"{model['key']} - {model['size']/1024/1024:.2f} MB")

# List training data
data = storage.list_files(prefix='training_data/')
for item in data:
    print(f"{item['key']} - {item['size']/1024/1024:.2f} MB")
```

### Command Line Usage

```bash
# Upload models
python s3_storage.py

# Or use specific functions
python -c "from s3_storage import S3StorageManager; S3StorageManager().upload_models()"
```

---

## Cost Estimation

### S3 Pricing (us-east-1)

**Storage**:
- First 50 TB: $0.023 per GB/month
- Your usage (~6MB models): **$0.00014/month** (essentially free)

**Requests**:
- PUT/COPY/POST: $0.005 per 1,000 requests
- GET/SELECT: $0.0004 per 1,000 requests
- Your usage (~10 uploads/month): **$0.00005/month**

**Data Transfer**:
- Upload: FREE
- Download (first 100 GB/month): FREE
- Your usage: **$0.00/month**

### Total Estimated Monthly Cost: **< $0.01/month** (FREE tier covers this)

---

## File Structure in S3

```
ml-trading-models-YOUR-NAME/
├── models/
│   ├── spy_trading_model_trade_quality_20251113_230157.pkl.gz
│   ├── spy_trading_model_future_high_20251113_230157.pkl.gz
│   ├── spy_trading_model_future_low_20251113_230157.pkl.gz
│   └── spy_trading_model_profit_target_20251113_230157.pkl.gz
└── training_data/
    ├── spy_ml_dataset.csv.gz
    └── prediction_log.csv.gz
```

---

## Security Best Practices

1. **Never commit credentials to Git**
   - Add `.env` to `.gitignore`
   - Use environment variables

2. **Use least privilege IAM policies**
   - Only grant S3 access
   - Limit to specific bucket

3. **Enable S3 bucket versioning** (optional)
   ```bash
   aws s3api put-bucket-versioning \
       --bucket ml-trading-models-YOUR-NAME \
       --versioning-configuration Status=Enabled
   ```

4. **Enable bucket encryption** (optional)
   ```bash
   aws s3api put-bucket-encryption \
       --bucket ml-trading-models-YOUR-NAME \
       --server-side-encryption-configuration \
       '{"Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]}'
   ```

---

## Troubleshooting

### Error: "Bucket name already exists"
- Bucket names are globally unique
- Add your name/number: `ml-trading-models-john-123`

### Error: "Access Denied"
- Check IAM policy includes your bucket name
- Verify credentials in `.env` file
- Ensure no typos in bucket name

### Error: "No credentials found"
- Make sure `.env` file exists
- Check `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` are set
- Run `python-dotenv` to load env vars

### Error: "Region not specified"
- Add `AWS_REGION=us-east-1` to `.env`
- Or specify when creating bucket

---

## Integration with Streamlit Cloud

### 1. Add Secrets
In Streamlit Cloud dashboard:
- Go to your app settings
- Click "Secrets"
- Add your AWS credentials

### 2. Auto-download Models on Startup

Add to `dashboard.py`:

```python
import os
from s3_storage import S3StorageManager

# Download models if not present locally
if not os.path.exists('spy_trading_model_trade_quality_20251113_230157.pkl'):
    print("Downloading models from S3...")
    storage = S3StorageManager()
    storage.download_models(decompress=True)
```

---

## Next Steps

1. ✓ Create AWS account
2. ✓ Create S3 bucket
3. ✓ Create IAM user
4. ✓ Save credentials to `.env`
5. ✓ Run `python s3_storage.py` to upload models
6. ✓ Verify files in S3 console

**You're done!** Your models are now backed up in S3.

---

## Support

For issues:
- AWS Documentation: [docs.aws.amazon.com/s3](https://docs.aws.amazon.com/s3)
- boto3 Documentation: [boto3.amazonaws.com/v1/documentation/api/latest/index.html](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
