"""
Enhanced Data Collector - Gets more historical data
Use this if data_collector.py isn't getting enough data
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get API token from environment variable
API_TOKEN = os.getenv('TRADIER_API_TOKEN')
SYMBOL = 'SPY'
DAYS_BACK = 365  # Get a full year of data

def get_daily_bars(symbol, start_date, end_date):
    """Get daily bars - most reliable method"""
    url = "https://api.tradier.com/v1/markets/history"
    
    headers = {
        'Authorization': f'Bearer {API_TOKEN}',
        'Accept': 'application/json'
    }
    
    params = {
        'symbol': symbol,
        'start': start_date,
        'end': end_date,
        'interval': 'daily'
    }
    
    print(f"Fetching {symbol} from {start_date} to {end_date}...")
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        
        # Try both possible keys
        history_data = None
        if 'history' in data and data['history']:
            history_data = data['history'].get('day') or data['history'].get('data')
        
        if history_data:
            df = pd.DataFrame(history_data)
            
            # Ensure date column exists
            if 'date' not in df.columns and 'time' in df.columns:
                df['date'] = df['time']
            
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            print(f"✓ Got {len(df)} bars")
            return df
        else:
            print(f"✗ No data in response")
            print(f"Response keys: {data.keys()}")
    else:
        print(f"✗ Error {response.status_code}")
        print(f"Response: {response.text[:500]}")
    
    return pd.DataFrame()

def main():
    print("\n" + "="*80)
    print("ENHANCED DATA COLLECTOR")
    print("="*80 + "\n")
    
    if not API_TOKEN:
        print("❌ ERROR: TRADIER_API_TOKEN not found!")
        print("\n📋 SETUP REQUIRED:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your API token:")
        print("   TRADIER_API_TOKEN=your_actual_token_here")
        print("3. Run this script again")
        print("\nAlternatively, set environment variable:")
        print("   export TRADIER_API_TOKEN='your_token_here'")
        return
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=DAYS_BACK)
    
    # Get data
    df = get_daily_bars(
        SYMBOL,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    if df.empty:
        print("\n❌ FAILED to collect data")
        print("\n📋 Troubleshooting:")
        print("1. Check your API token is correct")
        print("2. Check your internet connection")
        print("3. Verify the symbol exists (SPY, QQQ, etc.)")
        return
    
    # Ensure all required columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        print("\nCannot proceed - data format unexpected")
        return
    
    # Save files
    print(f"\nSaving data...")
    
    # Save as daily
    df.to_csv(f'{SYMBOL.lower()}_training_data_daily.csv', index=False)
    print(f"✓ Saved: {SYMBOL.lower()}_training_data_daily.csv")
    
    # Also save as intraday (for compatibility with pipeline)
    df_intraday = df.copy()
    if 'time' not in df_intraday.columns:
        df_intraday['time'] = df_intraday['date']
    
    df_intraday.to_csv(f'{SYMBOL.lower()}_training_data_intraday.csv', index=False)
    print(f"✓ Saved: {SYMBOL.lower()}_training_data_intraday.csv")
    
    # Summary
    print(f"\n" + "="*80)
    print("SUCCESS!")
    print("="*80)
    print(f"\n✓ Collected {len(df)} bars of {SYMBOL} data")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Columns: {', '.join(df.columns.tolist())}")
    
    # Data quality checks
    print(f"\nData Quality:")
    print(f"  - Missing values: {df.isnull().sum().sum()}")
    print(f"  - Duplicate dates: {df.duplicated(subset=['date']).sum()}")
    print(f"  - Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
    
    print(f"\n📋 NEXT STEPS:")
    print("1. Run: python feature_engineering.py")
    print("2. Then: python train_models.py")
    print("3. Finally: python predictor.py")
    
    print("\n💡 TIP: If you want more symbols, edit this file and:")
    print("   - Change SYMBOL = 'QQQ'")
    print("   - Run again")

if __name__ == "__main__":
    main()
