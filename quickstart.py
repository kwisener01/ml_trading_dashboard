#!/usr/bin/env python3
"""
Quick Start Script for ML Trading System
Runs the complete pipeline: Data Collection → Feature Engineering → Training → Prediction
"""

import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def print_header(text):
    """Print a nice header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def check_api_token():
    """Check if API token is set"""
    token = os.environ.get('TRADIER_API_TOKEN')
    if not token:
        print("[WARNING] TRADIER_API_TOKEN environment variable not set!")
        print("\nPlease set it using:")
        print("  export TRADIER_API_TOKEN='your_token_here'")
        print("\nOr edit the Python files directly and add your token.")
        return input("\nContinue anyway? (y/n): ").lower() == 'y'
    return True

def run_step(step_name, script_name, required_files=None):
    """Run a pipeline step"""
    print_header(f"STEP: {step_name}")
    
    # Check required files
    if required_files:
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print(f"[ERROR] Missing required files: {', '.join(missing)}")
            print(f"   Please run previous steps first.")
            return False
    
    # Run script (using same Python interpreter as this script)
    print(f"Running: {sys.executable} {script_name}")
    result = os.system(f'"{sys.executable}" {script_name}')
    
    if result == 0:
        print(f"\n[SUCCESS] {step_name} completed successfully!")
        return True
    else:
        print(f"\n[ERROR] {step_name} failed with exit code {result}")
        return False

def main():
    """Main pipeline"""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  ML TRADING SYSTEM - QUICK START PIPELINE".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    # Check API token
    if not check_api_token():
        print("\n[ERROR] Aborted.")
        sys.exit(1)
    
    # Menu
    print("\nWhat would you like to do?\n")
    print("1. Complete Pipeline (Data -> Features -> Train -> Predict)")
    print("2. Data Collection Only")
    print("3. Feature Engineering Only")
    print("4. Model Training Only")
    print("5. Generate Predictions Only")
    print("6. Launch Dashboard")
    print("7. Exit")
    
    choice = input("\nEnter choice (1-7): ").strip()
    
    if choice == '1':
        # Complete pipeline
        print("\n[RUNNING] Complete pipeline...")
        
        # Step 1: Data Collection
        if not run_step("Data Collection", "data_collector.py"):
            return
        
        # Step 2: Feature Engineering
        required = ["spy_training_data_intraday.csv"]
        if not run_step("Feature Engineering", "feature_engineering.py", required):
            return
        
        # Step 3: Model Training
        required = ["spy_ml_dataset.csv"]
        if not run_step("Model Training", "train_models.py", required):
            return
        
        # Step 4: Generate Prediction
        print_header("Generating Sample Prediction")
        if run_step("Prediction", "predictor.py"):
            print("\n[COMPLETE] PIPELINE COMPLETE!")
            print("\nNext steps:")
            print("  1. Review the prediction output above")
            print("  2. Launch dashboard: streamlit run dashboard.py")
            print("  3. Check model metrics in generated JSON files")
    
    elif choice == '2':
        run_step("Data Collection", "data_collector.py")
    
    elif choice == '3':
        required = ["spy_training_data_intraday.csv"]
        run_step("Feature Engineering", "feature_engineering.py", required)
    
    elif choice == '4':
        required = ["spy_ml_dataset.csv"]
        run_step("Model Training", "train_models.py", required)
    
    elif choice == '5':
        # Check if models exist
        import glob
        if not glob.glob('*_trading_model_*.pkl'):
            print("[ERROR] No trained models found!")
            print("   Run option 4 (Model Training) first.")
            return
        run_step("Prediction Generation", "predictor.py")
    
    elif choice == '6':
        # Launch dashboard
        print_header("Launching Dashboard")
        print("Starting Streamlit dashboard...")
        print("Dashboard will open in your browser at http://localhost:8501")
        print("\nPress Ctrl+C to stop the dashboard.")
        os.system(f'"{sys.executable}" -m streamlit run dashboard.py')
    
    elif choice == '7':
        print("\nGoodbye!")
        sys.exit(0)

    else:
        print("[ERROR] Invalid choice!")
        return

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
