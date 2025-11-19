"""
GEX (Gamma Exposure) Calculator
================================
Calculates market maker hedge pressure levels for SPY

Key Levels:
- Zero GEX Level: Where dealer hedging flips direction
- Max GEX Strike: Highest positive gamma (strongest support)
- Min GEX Strike: Highest negative gamma (strongest resistance)

How it works:
- Positive GEX = Dealers long gamma = Mean reversion (price gets pushed back)
- Negative GEX = Dealers short gamma = Momentum (price moves accelerate)
"""

import numpy as np
from scipy.stats import norm
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

load_dotenv()


class GEXCalculator:
    """
    Calculate Gamma Exposure (GEX) from options chain data.

    GEX shows where market makers need to hedge, creating
    support/resistance levels from their delta hedging activity.
    """

    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv('TRADIER_API_TOKEN')
        if not self.api_token:
            raise ValueError("TRADIER_API_TOKEN required")

        self.base_url = "https://api.tradier.com/v1"
        self.headers = {
            'Authorization': f'Bearer {self.api_token}',
            'Accept': 'application/json'
        }

    def calculate_gamma(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option gamma using Black-Scholes.

        Gamma = dDelta/dS = phi(d1) / (S * sigma * sqrt(T))
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

        return gamma

    def get_options_chain(self, symbol='SPY'):
        """Fetch options chain from Tradier."""
        import requests

        # Get expirations
        exp_url = f"{self.base_url}/markets/options/expirations"
        response = requests.get(exp_url, headers=self.headers, params={'symbol': symbol})

        if response.status_code != 200:
            raise Exception(f"Failed to get expirations: {response.text}")

        data = response.json()
        expirations = data.get('expirations', {}).get('date', [])

        if not expirations:
            raise Exception("No expirations found")

        # Get the nearest expiration (0DTE or next day)
        today = datetime.now().strftime('%Y-%m-%d')
        nearest_exp = expirations[0]  # First expiration

        # Get options chain for nearest expiration
        chain_url = f"{self.base_url}/markets/options/chains"
        response = requests.get(chain_url, headers=self.headers, params={
            'symbol': symbol,
            'expiration': nearest_exp,
            'greeks': 'true'
        })

        if response.status_code != 200:
            raise Exception(f"Failed to get chain: {response.text}")

        chain_data = response.json()
        options = chain_data.get('options', {}).get('option', [])

        return options, nearest_exp

    def calculate_gex(self, symbol='SPY', use_api_greeks=True):
        """
        Calculate GEX (Gamma Exposure) for all strikes.

        Returns DataFrame with GEX per strike and key levels.

        GEX Formula:
        GEX = Gamma * Open Interest * 100 * Spot Price

        Dealer assumption:
        - Dealers are typically short calls (sell to retail)
        - Dealers are typically short puts (sell to retail)
        - Therefore: Calls contribute positive GEX, Puts contribute negative GEX
        """
        import requests

        # Get current spot price
        quote_url = f"{self.base_url}/markets/quotes"
        response = requests.get(quote_url, headers=self.headers, params={'symbols': symbol})
        spot_price = response.json()['quotes']['quote']['last']

        # Get options chain
        options, expiration = self.get_options_chain(symbol)

        # Calculate days to expiration
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        dte = max((exp_date - datetime.now()).days, 1) / 365.0

        # Risk-free rate assumption
        r = 0.05

        # Process each option
        gex_data = []

        for opt in options:
            strike = opt['strike']
            option_type = opt['option_type']
            oi = opt.get('open_interest', 0)

            if oi == 0:
                continue

            # Get gamma (from API or calculate)
            if use_api_greeks and 'greeks' in opt and opt['greeks']:
                gamma = opt['greeks'].get('gamma', 0)
                iv = opt['greeks'].get('mid_iv', 0.25)
            else:
                # Estimate IV and calculate gamma
                iv = 0.25  # Default IV assumption
                gamma = self.calculate_gamma(spot_price, strike, dte, r, iv, option_type)

            # Calculate GEX contribution
            # Calls: Dealers short = they have negative gamma = positive GEX (buying dips)
            # Puts: Dealers short = they have positive gamma = negative GEX (selling rallies)
            if option_type == 'call':
                gex = gamma * oi * 100 * spot_price
            else:  # put
                gex = -gamma * oi * 100 * spot_price

            gex_data.append({
                'strike': strike,
                'option_type': option_type,
                'open_interest': oi,
                'gamma': gamma,
                'iv': iv,
                'gex': gex
            })

        df = pd.DataFrame(gex_data)

        if df.empty:
            return None, {}

        # Aggregate GEX by strike
        gex_by_strike = df.groupby('strike')['gex'].sum().reset_index()
        gex_by_strike.columns = ['strike', 'net_gex']

        # Find key levels
        key_levels = self._find_key_levels(gex_by_strike, spot_price)
        key_levels['spot_price'] = spot_price
        key_levels['expiration'] = expiration

        return gex_by_strike, key_levels

    def _find_key_levels(self, gex_df, spot_price):
        """Find key GEX levels for trading."""

        # Sort by strike
        gex_df = gex_df.sort_values('strike')

        # Max positive GEX (strongest support - dealers buy here)
        max_gex_idx = gex_df['net_gex'].idxmax()
        max_gex_strike = gex_df.loc[max_gex_idx, 'strike']
        max_gex_value = gex_df.loc[max_gex_idx, 'net_gex']

        # Max negative GEX (strongest resistance - dealers sell here)
        min_gex_idx = gex_df['net_gex'].idxmin()
        min_gex_strike = gex_df.loc[min_gex_idx, 'strike']
        min_gex_value = gex_df.loc[min_gex_idx, 'net_gex']

        # Zero GEX level (where hedging flips)
        # Find where cumulative GEX crosses zero near current price
        nearby = gex_df[abs(gex_df['strike'] - spot_price) < spot_price * 0.05]

        zero_gex = None
        for i in range(len(nearby) - 1):
            gex1 = nearby.iloc[i]['net_gex']
            gex2 = nearby.iloc[i + 1]['net_gex']
            if gex1 * gex2 < 0:  # Sign change
                # Interpolate
                strike1 = nearby.iloc[i]['strike']
                strike2 = nearby.iloc[i + 1]['strike']
                zero_gex = strike1 + (strike2 - strike1) * abs(gex1) / (abs(gex1) + abs(gex2))
                break

        # Total GEX (market regime indicator)
        total_gex = gex_df['net_gex'].sum()

        # GEX at current price level
        current_gex = gex_df[abs(gex_df['strike'] - spot_price) ==
                            abs(gex_df['strike'] - spot_price).min()]['net_gex'].values[0]

        return {
            'max_gex_strike': max_gex_strike,
            'max_gex_value': max_gex_value,
            'min_gex_strike': min_gex_strike,
            'min_gex_value': min_gex_value,
            'zero_gex_level': zero_gex,
            'total_gex': total_gex,
            'current_gex': current_gex
        }

    def print_levels(self, symbol='SPY'):
        """Print GEX levels in a readable format."""

        print("="*60)
        print(f"GEX HEDGE PRESSURE LEVELS - {symbol}")
        print("="*60)

        gex_df, levels = self.calculate_gex(symbol)

        if gex_df is None:
            print("No GEX data available")
            return

        spot = levels['spot_price']

        print(f"\nCurrent Price: ${spot:.2f}")
        print(f"Expiration: {levels['expiration']}")
        print()

        # Market Regime
        total = levels['total_gex']
        if total > 0:
            regime = "POSITIVE GEX (Mean Reversion)"
            regime_desc = "Moves likely to be contained, fade extremes"
        else:
            regime = "NEGATIVE GEX (Momentum)"
            regime_desc = "Moves can accelerate, trend following"

        print(f"Market Regime: {regime}")
        print(f"  -> {regime_desc}")
        print()

        # Key Levels
        print("KEY HEDGE LEVELS:")
        print("-"*40)

        # Support (Max GEX)
        max_strike = levels['max_gex_strike']
        pct_to_max = ((max_strike - spot) / spot) * 100
        print(f"GEX Support:    ${max_strike:.0f} ({pct_to_max:+.1f}%)")
        print(f"  Dealers will BUY here (hedge delta)")
        print()

        # Zero GEX (Flip Level)
        if levels['zero_gex_level']:
            zero = levels['zero_gex_level']
            pct_to_zero = ((zero - spot) / spot) * 100
            print(f"Zero GEX Level: ${zero:.0f} ({pct_to_zero:+.1f}%)")
            print(f"  Hedge pressure FLIPS here")
            print()

        # Resistance (Min GEX)
        min_strike = levels['min_gex_strike']
        pct_to_min = ((min_strike - spot) / spot) * 100
        print(f"GEX Resistance: ${min_strike:.0f} ({pct_to_min:+.1f}%)")
        print(f"  Dealers will SELL here (hedge delta)")
        print()

        # Current position
        curr_gex = levels['current_gex']
        print(f"Current GEX: {curr_gex:,.0f}")
        if curr_gex > 0:
            print(f"  -> Price at SUPPORT (dealers buying)")
        else:
            print(f"  -> Price at RESISTANCE (dealers selling)")

        print()
        print("="*60)

        # Top GEX strikes table
        print("\nTOP GEX STRIKES (by absolute value):")
        print("-"*40)

        gex_df['abs_gex'] = abs(gex_df['net_gex'])
        top_strikes = gex_df.nlargest(10, 'abs_gex')[['strike', 'net_gex']]

        for _, row in top_strikes.iterrows():
            strike = row['strike']
            gex = row['net_gex']
            direction = "BUY" if gex > 0 else "SELL"
            print(f"  ${strike:.0f}: {gex:>12,.0f} ({direction})")

        return gex_df, levels


def main():
    """Show SPY hedge pressure levels."""

    calc = GEXCalculator()
    calc.print_levels('SPY')

    print("\n")
    print("HOW TO USE:")
    print("-"*40)
    print("1. Above Zero GEX = Expect mean reversion")
    print("2. Below Zero GEX = Expect momentum/trend")
    print("3. At GEX Support = Dealers BUY (bounce)")
    print("4. At GEX Resistance = Dealers SELL (reject)")
    print()
    print("The Zero GEX Level is where dealer")
    print("hedging pressure FLIPS direction!")


if __name__ == "__main__":
    main()
