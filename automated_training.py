"""
Automated Training Pipeline
Runs data collection, training, and S3 upload automatically
Can be scheduled with GitHub Actions, AWS Lambda, or cron
"""
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def run_automated_training():
    """Complete automated training pipeline"""

    print("=" * 80)
    print("AUTOMATED TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Step 1: Collect fresh data
    print("\n[1/4] Collecting fresh market data...")
    try:
        from data_collector import TradierDataCollector

        # Get API token from environment
        api_token = os.getenv('TRADIER_API_TOKEN')
        if not api_token:
            raise ValueError("TRADIER_API_TOKEN not found in environment variables")

        collector = TradierDataCollector(api_token)
        data_dict = collector.collect_training_data(
            symbol='SPY',
            days_back=180,      # 6 months - more recent for day trading
            interval='5min'     # 5-min bars for precise entries
        )

        # Extract DataFrame from the returned dictionary
        # Use 'daily' data for training (more reliable than intraday)
        df = data_dict.get('daily', data_dict.get('intraday'))

        if df is None or df.empty:
            raise ValueError("No data collected")

        # Save dataset
        df.to_csv('spy_ml_dataset.csv', index=False)
        print(f"[OK] Collected {len(df)} samples")

    except Exception as e:
        print(f"[ERROR] Data collection failed: {e}")
        return False

    # Step 2: Feature Engineering
    print("\n[2/5] Creating ML features...")
    try:
        from feature_engineering import FeatureEngineering
        import pandas as pd

        df = pd.read_csv('spy_ml_dataset.csv')

        # Create features
        fe = FeatureEngineering(df)
        fe.add_technical_indicators()
        fe.add_support_resistance_levels()
        fe.add_market_regime_features()
        fe.add_target_variables(lookahead=1)  # Predict 1 day ahead

        # Get processed data
        df_processed = fe.get_data()

        # Save processed dataset
        df_processed.to_csv('spy_ml_dataset.csv', index=False)
        print(f"[OK] Created {len(df_processed.columns)} features")

    except Exception as e:
        print(f"[ERROR] Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Train models
    print("\n[3/5] Training ML models...")
    try:
        from train_models import TradingMLModel
        import pandas as pd

        df = pd.read_csv('spy_ml_dataset.csv')
        trainer = TradingMLModel()

        # Train all models
        trainer.train_all_models(df)

        # Save models locally
        trainer.save_models('spy_trading_model')
        print("[OK] Models trained and saved")

    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 4: Upload to S3
    print("\n[4/5] Uploading to S3...")
    try:
        from s3_storage import S3StorageManager

        storage = S3StorageManager()

        # Upload models
        storage.upload_models(compress=True)

        # Upload training data
        storage.upload_training_data('spy_ml_dataset.csv', compress=True)

        print("[OK] All files uploaded to S3")

    except Exception as e:
        print(f"[ERROR] S3 upload failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Summary
    print("\n[5/5] Summary")
    print("=" * 80)
    print("AUTOMATED TRAINING COMPLETE!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nFiles updated:")
    print("  - spy_ml_dataset.csv (latest market data)")
    print("  - spy_trading_model_*.pkl (trained models)")
    print("  - All files backed up to S3")

    return True


def lambda_handler(event, context):
    """
    AWS Lambda handler
    Deploy this to AWS Lambda for serverless scheduled training
    """
    success = run_automated_training()

    return {
        'statusCode': 200 if success else 500,
        'body': 'Training completed successfully' if success else 'Training failed'
    }


if __name__ == "__main__":
    success = run_automated_training()
    sys.exit(0 if success else 1)
