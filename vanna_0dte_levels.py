"""
Calculate Support and Resistance Levels based on Vanna for 0DTE Options

Vanna measures the sensitivity of delta to changes in implied volatility.
For options positioning:
- High Call OI at strikes with positive vanna = resistance (dealers sell on rallies)
- High Put OI at strikes with positive vanna = support (dealers buy on dips)
"""

import requests
import json
from datetime import datetime
import os
from dotenv import load_dotenv
import numpy as np
from scipy.stats import norm

# Load environment variables
load_dotenv()


class VannaLevelsCalculator:
    def __init__(self, api_token):
        """Initialize with Tradier API token"""
        self.api_token = api_token
        self.base_url = "https://api.tradier.com/v1"
        self.headers = {
            'Authorization': f'Bearer {api_token}',
            'Accept': 'application/json'
        }

    def get_current_price(self, symbol):
        """Get current stock price"""
        url = f"{self.base_url}/markets/quotes"
        params = {'symbols': symbol}

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            quotes = data.get('quotes', {}).get('quote', {})
            return quotes.get('last', 0)
        return 0

    def get_option_expirations(self, symbol):
        """Get available expiration dates"""
        url = f"{self.base_url}/markets/options/expirations"
        params = {'symbol': symbol}

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            data = response.json()
            return data.get('expirations', {}).get('date', [])
        return []

    def get_option_chain(self, symbol, expiration):
        """Get option chain with Greeks"""
        url = f"{self.base_url}/markets/options/chains"
        params = {
            'symbol': symbol,
            'expiration': expiration,
            'greeks': 'true'
        }

        response = requests.get(url, headers=self.headers, params=params)

        if response.status_code == 200:
            return response.json()
        return None

    def calculate_vanna(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate Vanna (∂Delta/∂σ or ∂Vega/∂S)

        Vanna = (Vega/S) * (d2/σ)
        where d2 = [ln(S/K) + (r - σ²/2)T] / (σ√T)

        Parameters:
        S: Current stock price
        K: Strike price
        T: Time to expiration (years)
        r: Risk-free rate
        sigma: Implied volatility
        """
        if T <= 0 or sigma <= 0:
            return 0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        # Vanna formula
        vanna = -norm.pdf(d1) * d2 / sigma

        return vanna

    def calculate_vanna_exposure(self, options_data, current_price):
        """
        Calculate vanna exposure at each strike

        Returns:
        - Dictionary with strike levels as keys
        - Values contain vanna, open interest, and net exposure
        """
        if not options_data or 'options' not in options_data:
            return {}

        vanna_levels = {}
        options_list = options_data['options']['option']

        # Estimate time to expiration (0DTE = end of day)
        now = datetime.now()
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        time_remaining = (market_close - now).total_seconds() / (365 * 24 * 3600)
        time_remaining = max(time_remaining, 1/(365*24))  # Minimum 1 hour

        r = 0.05  # Risk-free rate (5%)

        for option in options_list:
            strike = option.get('strike', 0)
            option_type = option.get('option_type', 'call')
            open_interest = option.get('open_interest', 0)

            # Skip if no open interest
            if open_interest == 0:
                continue

            # Get Greeks
            greeks = option.get('greeks', {})
            if not greeks:
                continue

            # Get implied volatility (use mid_iv or estimate from greeks)
            iv = greeks.get('mid_iv', 0.25)  # Default to 25% if not available
            vega = greeks.get('vega', 0)
            delta = greeks.get('delta', 0)

            # Calculate vanna
            vanna = self.calculate_vanna(current_price, strike, time_remaining, r, iv, option_type)

            # Vanna exposure = Vanna * Open Interest * 100 (shares per contract)
            # Positive vanna on calls = resistance (dealers sell as price rises)
            # Positive vanna on puts = support (dealers buy as price falls)

            if option_type == 'call':
                # For calls, positive vanna at high OI = resistance
                exposure = vanna * open_interest * 100
                level_type = 'resistance' if strike > current_price else 'support'
            else:  # put
                # For puts, positive vanna at high OI = support
                exposure = vanna * open_interest * 100
                level_type = 'support' if strike < current_price else 'resistance'

            if strike not in vanna_levels:
                vanna_levels[strike] = {
                    'call_vanna': 0,
                    'put_vanna': 0,
                    'call_oi': 0,
                    'put_oi': 0,
                    'call_exposure': 0,
                    'put_exposure': 0,
                    'net_vanna_exposure': 0,
                    'level_type': level_type
                }

            # Accumulate by option type
            if option_type == 'call':
                vanna_levels[strike]['call_vanna'] += vanna
                vanna_levels[strike]['call_oi'] += open_interest
                vanna_levels[strike]['call_exposure'] += exposure
            else:
                vanna_levels[strike]['put_vanna'] += vanna
                vanna_levels[strike]['put_oi'] += open_interest
                vanna_levels[strike]['put_exposure'] += exposure

            # Net exposure
            vanna_levels[strike]['net_vanna_exposure'] = (
                vanna_levels[strike]['call_exposure'] + vanna_levels[strike]['put_exposure']
            )

        return vanna_levels

    def identify_key_levels(self, vanna_levels, current_price, top_n=5):
        """Identify the most significant support and resistance levels"""

        # Separate into support and resistance
        support_levels = []
        resistance_levels = []

        for strike, data in vanna_levels.items():
            net_exposure = abs(data['net_vanna_exposure'])

            if strike < current_price:
                # Below current price = potential support
                # Look for positive put vanna (dealers buy on dips)
                if data['put_vanna'] > 0:
                    support_levels.append({
                        'strike': strike,
                        'exposure': net_exposure,
                        'put_oi': data['put_oi'],
                        'call_oi': data['call_oi'],
                        'vanna': data['put_vanna']
                    })
            else:
                # Above current price = potential resistance
                # Look for positive call vanna (dealers sell on rallies)
                if data['call_vanna'] > 0:
                    resistance_levels.append({
                        'strike': strike,
                        'exposure': net_exposure,
                        'put_oi': data['put_oi'],
                        'call_oi': data['call_oi'],
                        'vanna': data['call_vanna']
                    })

        # Sort by exposure strength
        support_levels = sorted(support_levels, key=lambda x: x['exposure'], reverse=True)[:top_n]
        resistance_levels = sorted(resistance_levels, key=lambda x: x['exposure'], reverse=True)[:top_n]

        return support_levels, resistance_levels

    def display_levels(self, symbol, current_price, support_levels, resistance_levels):
        """Display formatted support and resistance levels"""

        print(f"\n{'='*100}")
        print(f"VANNA-BASED 0DTE SUPPORT & RESISTANCE LEVELS FOR {symbol}")
        print(f"{'='*100}")
        print(f"Current Price: ${current_price:.2f}")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\n{'='*100}")
        print("SUPPORT LEVELS (Buy zones - Dealers hedge by buying on dips)")
        print(f"{'='*100}")
        print(f"{'Strike':<10} {'Distance':<12} {'Put OI':<12} {'Call OI':<12} {'Vanna':<12} {'Exposure':<15}")
        print(f"{'-'*100}")

        if support_levels:
            for level in support_levels:
                strike = level['strike']
                distance = ((current_price - strike) / current_price) * 100
                print(f"${strike:<9.2f} {distance:>6.2f}%     {level['put_oi']:<12,} {level['call_oi']:<12,} "
                      f"{level['vanna']:<12.6f} {level['exposure']:<15,.0f}")
        else:
            print("No significant support levels found")

        print(f"\n{'='*100}")
        print("RESISTANCE LEVELS (Sell zones - Dealers hedge by selling on rallies)")
        print(f"{'='*100}")
        print(f"{'Strike':<10} {'Distance':<12} {'Put OI':<12} {'Call OI':<12} {'Vanna':<12} {'Exposure':<15}")
        print(f"{'-'*100}")

        if resistance_levels:
            for level in resistance_levels:
                strike = level['strike']
                distance = ((strike - current_price) / current_price) * 100
                print(f"${strike:<9.2f} {distance:>+6.2f}%     {level['put_oi']:<12,} {level['call_oi']:<12,} "
                      f"{level['vanna']:<12.6f} {level['exposure']:<15,.0f}")
        else:
            print("No significant resistance levels found")

        print(f"\n{'='*100}")


def main():
    # Get API token from environment
    API_TOKEN = os.getenv('TRADIER_API_TOKEN')

    if not API_TOKEN:
        print("❌ ERROR: TRADIER_API_TOKEN not found!")
        print("\n📋 Setup .env file with your Tradier API token")
        return

    # Initialize calculator
    calc = VannaLevelsCalculator(API_TOKEN)

    # Symbols to analyze
    symbols = ['SPY', 'QQQ']

    for symbol in symbols:
        print(f"\n{'#'*100}")
        print(f"Analyzing {symbol}...")
        print(f"{'#'*100}")

        # Get current price
        current_price = calc.get_current_price(symbol)
        if current_price == 0:
            print(f"Could not get current price for {symbol}")
            continue

        # Get expirations
        expirations = calc.get_option_expirations(symbol)
        if not expirations:
            print(f"No expirations found for {symbol}")
            continue

        # Use today's expiration (0DTE)
        today = datetime.now().strftime('%Y-%m-%d')
        target_expiration = today if today in expirations else expirations[0]

        print(f"Using expiration: {target_expiration}")

        # Get option chain
        chain_data = calc.get_option_chain(symbol, target_expiration)

        if not chain_data:
            print(f"Could not get option chain for {symbol}")
            continue

        # Calculate vanna exposure at each strike
        vanna_levels = calc.calculate_vanna_exposure(chain_data, current_price)

        # Identify key support and resistance levels
        support_levels, resistance_levels = calc.identify_key_levels(
            vanna_levels, current_price, top_n=5
        )

        # Display results
        calc.display_levels(symbol, current_price, support_levels, resistance_levels)


if __name__ == "__main__":
    main()
