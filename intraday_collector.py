"""
Intraday Data Collector for Day Trading

Collects 5-minute bars for intraday/day trading analysis.
The ML system works for both daily and intraday timeframes.
"""

import pandas as pd
import sys
from datetime import datetime, timedelta
from data_collector import TradierDataCollector
from feature_engineering import FeatureEngineering
import os
from dotenv import load_dotenv

load_dotenv()


def collect_intraday_data(symbol='SPY', days_back=5, interval='5min'):
    """
    Collect intraday data for day trading

    Args:
        symbol: Stock symbol
        days_back: How many days of intraday data to collect
        interval: Time interval (1min, 5min, 15min)

    Returns:
        DataFrame with intraday OHLCV data
    """
    print("="*80)
    print(f"COLLECTING INTRADAY DATA FOR DAY TRADING")
    print("="*80)
    print(f"Symbol: {symbol}")
    print(f"Interval: {interval}")
    print(f"Days Back: {days_back}")

    api_token = os.getenv('TRADIER_API_TOKEN')
    if not api_token:
        print("\n[ERROR] TRADIER_API_TOKEN not found in .env file")
        sys.exit(1)

    collector = TradierDataCollector(api_token)

    # Collect intraday data for each day
    all_data = []
    end_date = datetime.now()

    for i in range(days_back):
        day = end_date - timedelta(days=i)

        # Market hours: 9:30 AM - 4:00 PM ET
        start_time = day.replace(hour=9, minute=30, second=0)
        end_time = day.replace(hour=16, minute=0, second=0)

        print(f"\nFetching {day.date()}...")

        df = collector.get_intraday_quotes(
            symbol=symbol,
            start_time=start_time.strftime('%Y-%m-%d %H:%M'),
            end_time=end_time.strftime('%Y-%m-%d %H:%M'),
            interval=interval
        )

        if not df.empty:
            all_data.append(df)
            print(f"  [OK] {len(df)} bars")
        else:
            print(f"  [SKIP] No data (weekend/holiday)")

    if not all_data:
        print("\n[ERROR] No intraday data collected")
        return pd.DataFrame()

    # Combine all days
    combined_df = pd.concat(all_data, ignore_index=True)
    combined_df = combined_df.sort_values('time')

    print(f"\n[SUCCESS] Total bars collected: {len(combined_df)}")

    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in combined_df.columns for col in required_cols):
        print(f"[ERROR] Missing required columns")
        return pd.DataFrame()

    # Rename 'time' to 'date' for compatibility with feature engineering
    combined_df = combined_df.rename(columns={'time': 'date'})

    return combined_df


def prepare_intraday_dataset(symbol='SPY', days_back=5, interval='5min'):
    """
    Collect intraday data and create ML-ready dataset

    Args:
        symbol: Stock symbol
        days_back: Days of intraday data
        interval: Bar interval

    Returns:
        DataFrame with features and targets
    """
    # Collect data
    df = collect_intraday_data(symbol, days_back, interval)

    if df.empty:
        print("[ERROR] No data to process")
        return None

    # Feature engineering
    print("\n" + "="*80)
    print("CREATING INTRADAY FEATURES")
    print("="*80)

    fe = FeatureEngineering(df)

    fe.add_technical_indicators()
    print("[OK] Technical indicators")

    fe.add_support_resistance_levels(lookback=20)
    print("[OK] Support/Resistance levels")

    try:
        fe.add_vanna_levels(dte=1, iv_estimate=0.25, strike_width=1.0)
        print("[OK] Vanna levels (0DTE optimized)")
    except Exception as e:
        print(f"[SKIP] Vanna levels: {e}")

    fe.add_market_regime_features()
    print("[OK] Market regime features")

    fe.add_time_features()
    print("[OK] Time features")

    # For intraday, use shorter forward periods (e.g., 12 bars = 1 hour for 5min data)
    fe.create_target_labels(
        forward_periods=12,  # 1 hour ahead for 5min bars
        profit_threshold=0.003,  # 0.3% profit (tighter for intraday)
        loss_threshold=-0.002   # 0.2% stop (tighter for intraday)
    )
    print("[OK] Target labels (intraday optimized)")

    features, full_data = fe.get_features_for_ml()

    print(f"\n[SUCCESS] Intraday dataset created:")
    print(f"  Features: {len(features.columns)}")
    print(f"  Samples: {len(full_data)}")

    # Save
    filename = f'{symbol.lower()}_intraday_{interval}_dataset.csv'
    full_data.to_csv(filename, index=False)
    print(f"\n[OK] Saved to: {filename}")

    return full_data


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Collect intraday data for day trading')
    parser.add_argument('--symbol', default='SPY', help='Stock symbol')
    parser.add_argument('--days', type=int, default=5, help='Days of data to collect')
    parser.add_argument('--interval', default='5min', choices=['1min', '5min', '15min'],
                       help='Bar interval')

    args = parser.parse_args()

    # Collect and prepare intraday dataset
    df = prepare_intraday_dataset(
        symbol=args.symbol,
        days_back=args.days,
        interval=args.interval
    )

    if df is not None:
        print("\n" + "="*80)
        print("READY FOR DAY TRADING")
        print("="*80)
        print("\nThis dataset can be used to:")
        print("1. Train ML models specifically for intraday trading")
        print("2. Backtest day trading strategies")
        print("3. Generate real-time intraday signals")
        print("\nNext steps:")
        print("  python train_models.py  # Train on intraday data")
        print("  python backtest.py      # Backtest intraday strategies")
