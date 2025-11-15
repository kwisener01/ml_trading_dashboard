import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

class TradierDataCollector:
    """
    Collects historical and real-time data from Tradier API for ML model training
    """
    
    def __init__(self, api_token, use_sandbox=False):
        self.api_token = api_token
        self.base_url = "https://sandbox.tradier.com/v1" if use_sandbox else "https://api.tradier.com/v1"
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json'
        }
    
    def get_historical_quotes(self, symbol, start_date, end_date, interval='daily'):
        """
        Get historical price data
        interval: daily, weekly, monthly, or specific intraday (1min, 5min, 15min)
        """
        url = f"{self.base_url}/markets/history"
        params = {
            'symbol': symbol,
            'start': start_date,
            'end': end_date,
            'interval': interval
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'history' in data and data['history'] and 'day' in data['history']:
                df = pd.DataFrame(data['history']['day'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                return df
        
        print(f"Error getting historical data: {response.status_code}")
        return pd.DataFrame()
    
    def get_intraday_quotes(self, symbol, start_time, end_time, interval='5min'):
        """
        Get intraday price data (critical for 0DTE trading)
        interval: 1min, 5min, 15min
        """
        url = f"{self.base_url}/markets/timesales"
        params = {
            'symbol': symbol,
            'interval': interval,
            'start': start_time,
            'end': end_time,
            'session_filter': 'all'
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'series' in data and data['series'] and 'data' in data['series']:
                df = pd.DataFrame(data['series']['data'])
                df['time'] = pd.to_datetime(df['time'])
                return df
        
        print(f"Error getting intraday data: {response.status_code}")
        return pd.DataFrame()
    
    def get_realtime_quote(self, symbols):
        """Get real-time quote data"""
        url = f"{self.base_url}/markets/quotes"
        params = {'symbols': ','.join(symbols) if isinstance(symbols, list) else symbols}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'quotes' in data and 'quote' in data['quotes']:
                quotes = data['quotes']['quote']
                if not isinstance(quotes, list):
                    quotes = [quotes]
                return pd.DataFrame(quotes)
        
        return pd.DataFrame()
    
    def get_option_chain_with_greeks(self, symbol, expiration):
        """Get option chain with Greeks for a specific expiration"""
        url = f"{self.base_url}/markets/options/chains"
        params = {
            'symbol': symbol,
            'expiration': expiration,
            'greeks': 'true'
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if 'options' in data and 'option' in data['options']:
                df = pd.DataFrame(data['options']['option'])
                
                # Expand Greeks into separate columns
                if 'greeks' in df.columns:
                    greeks_df = pd.json_normalize(df['greeks'])
                    df = pd.concat([df.drop('greeks', axis=1), greeks_df], axis=1)
                
                return df
        
        return pd.DataFrame()
    
    def get_vix_data(self, start_date, end_date):
        """Get VIX data (volatility index)"""
        return self.get_historical_quotes('VIX', start_date, end_date)
    
    def calculate_market_internals(self, symbol='SPY'):
        """
        Calculate market internal indicators
        Returns: dict with key market health metrics
        """
        quote = self.get_realtime_quote([symbol])
        
        if quote.empty:
            return {}
        
        quote = quote.iloc[0]
        
        internals = {
            'price': quote.get('last', 0),
            'volume': quote.get('volume', 0),
            'bid': quote.get('bid', 0),
            'ask': quote.get('ask', 0),
            'spread': quote.get('ask', 0) - quote.get('bid', 0),
            'spread_pct': ((quote.get('ask', 0) - quote.get('bid', 0)) / quote.get('last', 1)) * 100,
            'change': quote.get('change', 0),
            'change_pct': quote.get('change_percentage', 0),
        }
        
        return internals
    
    def collect_training_data(self, symbol, days_back=60, interval='5min'):
        """
        Collect comprehensive training data including:
        - Price action
        - Volume
        - Option flow
        - Market internals
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        print(f"Collecting {days_back} days of data for {symbol}...")
        
        # Get daily data
        daily_data = self.get_historical_quotes(
            symbol, 
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        print(f"Collected {len(daily_data)} daily bars")
        
        # Get VIX data
        vix_data = self.get_vix_data(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d')
        )
        
        if not vix_data.empty:
            vix_data = vix_data[['date', 'close']].rename(columns={'close': 'vix'})
            daily_data = daily_data.merge(vix_data, on='date', how='left')
        
        # For intraday data, collect last 5 trading days
        intraday_data = []
        for i in range(5):
            date = end_date - timedelta(days=i)
            # Skip weekends
            if date.weekday() >= 5:
                continue
            
            day_start = date.replace(hour=9, minute=30, second=0)
            day_end = date.replace(hour=16, minute=0, second=0)
            
            print(f"Getting intraday data for {date.strftime('%Y-%m-%d')}...")
            
            intraday = self.get_intraday_quotes(
                symbol,
                day_start.strftime('%Y-%m-%d %H:%M'),
                day_end.strftime('%Y-%m-%d %H:%M'),
                interval
            )
            
            if not intraday.empty:
                intraday_data.append(intraday)
            
            time.sleep(0.5)  # Rate limiting
        
        if intraday_data:
            intraday_df = pd.concat(intraday_data, ignore_index=True)
            print(f"Collected {len(intraday_df)} intraday bars")
        else:
            intraday_df = pd.DataFrame()
        
        return {
            'daily': daily_data,
            'intraday': intraday_df,
            'symbol': symbol,
            'collected_at': datetime.now()
        }
    
    def save_data(self, data, filename):
        """Save collected data to files"""
        if 'daily' in data and not data['daily'].empty:
            data['daily'].to_csv(f"{filename}_daily.csv", index=False)
            print(f"Saved daily data: {filename}_daily.csv")
        
        if 'intraday' in data and not data['intraday'].empty:
            data['intraday'].to_csv(f"{filename}_intraday.csv", index=False)
            print(f"Saved intraday data: {filename}_intraday.csv")
        
        # Save metadata
        metadata = {
            'symbol': data['symbol'],
            'collected_at': data['collected_at'].isoformat(),
            'daily_rows': len(data['daily']) if 'daily' in data else 0,
            'intraday_rows': len(data['intraday']) if 'intraday' in data else 0
        }
        
        with open(f"{filename}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    # Example usage
    API_TOKEN = "zNpWOOIJQBafal07lLmKWMO1dLrJ"
    
    collector = TradierDataCollector(API_TOKEN)
    
    # Collect training data for SPY
    data = collector.collect_training_data('SPY', days_back=60)
    collector.save_data(data, 'spy_training_data')
    
    # Collect for QQQ
    data = collector.collect_training_data('QQQ', days_back=60)
    collector.save_data(data, 'qqq_training_data')
    
    print("\nData collection complete!")
