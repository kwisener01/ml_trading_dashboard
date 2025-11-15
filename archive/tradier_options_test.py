import requests
import json
from datetime import datetime, timedelta

class TradierOptionsScanner:
    def __init__(self, api_token):
        """Initialize with your Tradier API token"""
        self.api_token = api_token
        self.base_url = "https://api.tradier.com/v1"
        self.sandbox_url = "https://sandbox.tradier.com/v1"  # Use this for testing
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json'
        }
    
    def get_option_expirations(self, symbol):
        """Get available expiration dates for a symbol"""
        url = f"{self.base_url}/markets/options/expirations"
        params = {'symbol': symbol}
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('expirations', {}).get('date', [])
        else:
            print(f"Error getting expirations: {response.status_code}")
            print(response.text)
            return []
    
    def get_option_chain(self, symbol, expiration):
        """Get full option chain for a symbol and expiration"""
        url = f"{self.base_url}/markets/options/chains"
        params = {
            'symbol': symbol,
            'expiration': expiration,
            'greeks': 'true'  # This is key - gets delta values
        }
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting option chain: {response.status_code}")
            print(response.text)
            return None
    
    def filter_options_by_delta(self, options_data, min_delta=None, max_delta=None, option_type='call'):
        """Filter options by delta range"""
        if not options_data or 'options' not in options_data:
            return []
        
        filtered = []
        options_list = options_data['options']['option']
        
        for option in options_list:
            # Filter by option type (call or put)
            if option['option_type'] != option_type:
                continue
            
            # Check if greeks data exists
            if 'greeks' not in option or not option['greeks']:
                continue
            
            delta = option['greeks'].get('delta', 0)
            
            # Apply delta filters
            if min_delta is not None and abs(delta) < abs(min_delta):
                continue
            if max_delta is not None and abs(delta) > abs(max_delta):
                continue
            
            filtered.append(option)
        
        return filtered
    
    def display_option_data(self, options, symbol, expiration):
        """Display formatted option data"""
        print(f"\n{'='*100}")
        print(f"{symbol} Options - Expiration: {expiration}")
        print(f"{'='*100}")
        print(f"{'Strike':<10} {'Type':<6} {'Last':<10} {'Bid':<10} {'Ask':<10} {'Volume':<10} {'Delta':<10} {'Gamma':<10} {'Theta':<10} {'Vega':<10}")
        print(f"{'-'*100}")
        
        for opt in options:
            strike = opt.get('strike', 'N/A')
            opt_type = opt.get('option_type', 'N/A')
            last = opt.get('last', 0)
            bid = opt.get('bid', 0)
            ask = opt.get('ask', 0)
            volume = opt.get('volume', 0)
            
            greeks = opt.get('greeks', {})
            delta = greeks.get('delta', 0) if greeks else 0
            gamma = greeks.get('gamma', 0) if greeks else 0
            theta = greeks.get('theta', 0) if greeks else 0
            vega = greeks.get('vega', 0) if greeks else 0
            
            print(f"{strike:<10} {opt_type:<6} ${last:<9.2f} ${bid:<9.2f} ${ask:<9.2f} {volume:<10} {delta:<10.4f} {gamma:<10.4f} {theta:<10.4f} {vega:<10.4f}")

def main():
    # *** ENTER YOUR TRADIER API TOKEN HERE ***
    API_TOKEN = "zNpWOOIJQBafal07lLmKWMO1dLrJ"
    
    # Initialize scanner
    scanner = TradierOptionsScanner(API_TOKEN)
    
    # Symbols to scan
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        print(f"\n\n{'#'*100}")
        print(f"Scanning {symbol}...")
        print(f"{'#'*100}")
        
        # Get available expirations
        expirations = scanner.get_option_expirations(symbol)
        
        if not expirations:
            print(f"No expirations found for {symbol}")
            continue
        
        print(f"\nAvailable expirations: {expirations[:5]}")  # Show first 5
        
        # Use the nearest expiration (typically 0DTE or next available)
        nearest_exp = expirations[0]
        print(f"\nUsing expiration: {nearest_exp}")
        
        # Get option chain
        chain_data = scanner.get_option_chain(symbol, nearest_exp)
        
        if chain_data:
            # Filter for CALLS with delta between 0.30 and 0.70 (near the money)
            print("\n\nCALLS (Delta 0.30 - 0.70):")
            calls = scanner.filter_options_by_delta(
                chain_data, 
                min_delta=0.30, 
                max_delta=0.70, 
                option_type='call'
            )
            scanner.display_option_data(calls[:10], symbol, nearest_exp)  # Show top 10
            
            # Filter for PUTS with delta between -0.70 and -0.30
            print("\n\nPUTS (Delta -0.70 to -0.30):")
            puts = scanner.filter_options_by_delta(
                chain_data, 
                min_delta=-0.70, 
                max_delta=-0.30, 
                option_type='put'
            )
            scanner.display_option_data(puts[:10], symbol, nearest_exp)  # Show top 10

if __name__ == "__main__":
    main()
