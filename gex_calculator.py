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

    def calculate_vanna(self, S, K, T, r, sigma, option_type='call'):
        """
        Calculate option vanna using Black-Scholes.

        Vanna = ∂²V/∂S∂σ = ∂Delta/∂σ (delta sensitivity to IV)

        Formula: Vanna = -phi(d1) * d2 / sigma
        where d2 = d1 - sigma*sqrt(T)
        """
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        vanna = -norm.pdf(d1) * d2 / sigma

        return vanna

    def get_options_chain(self, symbol='SPY'):
        """Fetch options chain from Tradier."""
        import requests

        # Get expirations
        print(f"[GEX] Fetching expirations for {symbol}...")
        exp_url = f"{self.base_url}/markets/options/expirations"
        response = requests.get(exp_url, headers=self.headers, params={'symbol': symbol})

        print(f"[GEX] Expirations API response: {response.status_code}")
        if response.status_code != 200:
            raise Exception(f"Failed to get expirations (HTTP {response.status_code}): {response.text}")

        data = response.json()
        expirations = data.get('expirations', {}).get('date', [])
        print(f"[GEX] Found {len(expirations)} expirations: {expirations[:5] if expirations else 'None'}")

        if not expirations:
            raise Exception("No expirations found in API response")

        # Get the nearest expiration (0DTE or 1DTE prioritized)
        today = datetime.now().strftime('%Y-%m-%d')
        tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')

        # Try to get 0DTE first (same day expiration)
        if today in expirations:
            nearest_exp = today
            print(f"[GEX] Using 0DTE expiration: {nearest_exp}")
        # Try to get 1DTE (next day expiration)
        elif tomorrow in expirations:
            nearest_exp = tomorrow
            print(f"[GEX] Using 1DTE expiration: {nearest_exp}")
        # Fall back to nearest available expiration
        else:
            nearest_exp = expirations[0]
            # Calculate days to expiration
            exp_date = datetime.strptime(nearest_exp, '%Y-%m-%d')
            dte = (exp_date - datetime.now()).days
            print(f"[GEX] Using {dte}DTE expiration: {nearest_exp} (0DTE/1DTE not available)")

            # Warn if using >2DTE (less impactful for daily movement)
            if dte > 2:
                print(f"[WARNING] Using {dte}DTE options - may not capture daily dealer flows accurately")

        # Get options chain for nearest expiration
        print(f"[GEX] Fetching options chain for expiration {nearest_exp}...")
        chain_url = f"{self.base_url}/markets/options/chains"
        response = requests.get(chain_url, headers=self.headers, params={
            'symbol': symbol,
            'expiration': nearest_exp,
            'greeks': 'true'
        })

        print(f"[GEX] Options chain API response: {response.status_code}")
        if response.status_code != 200:
            raise Exception(f"Failed to get chain (HTTP {response.status_code}): {response.text}")

        chain_data = response.json()
        options_data = chain_data.get('options', {}).get('option', [])

        # Handle case where API returns single option as dict instead of list
        if isinstance(options_data, dict):
            options = [options_data]
            print(f"[GEX] Received 1 option from chain (converted dict to list)")
        elif isinstance(options_data, list):
            options = options_data
            print(f"[GEX] Received {len(options)} options from chain")
        else:
            options = []
            print(f"[GEX] Unexpected options data type: {type(options_data)}")

        if not options:
            raise Exception(f"No options data in chain for expiration {nearest_exp}")

        return options, nearest_exp

    def calculate_gex(self, symbol='SPY', use_api_greeks=True):
        """
        Calculate GEX (Gamma Exposure) and Vanna Exposure for all strikes.

        Returns DataFrame with GEX/Vanna per strike and key levels.

        GEX Formula:
        GEX = Gamma * Open Interest * 100 * Spot Price

        Vanna Exposure Formula:
        Vanna Exposure = Vanna * Open Interest * 100 * Spot Price

        Dealer assumption:
        - Dealers are typically short calls (sell to retail)
        - Dealers are typically short puts (sell to retail)
        - Therefore: Calls contribute positive GEX, Puts contribute negative GEX
        - Vanna shows delta sensitivity to IV changes
        """
        import requests

        print(f"\n[GEX] ========== Starting GEX/Vanna calculation for {symbol} ==========")

        # Get current spot price
        print(f"[GEX] Fetching current price for {symbol}...")
        quote_url = f"{self.base_url}/markets/quotes"
        response = requests.get(quote_url, headers=self.headers, params={'symbols': symbol})

        if response.status_code != 200:
            raise Exception(f"Failed to get quote (HTTP {response.status_code}): {response.text}")

        spot_price = response.json()['quotes']['quote']['last']
        print(f"[GEX] Current {symbol} price: ${spot_price:.2f}")

        # Get options chain
        options, expiration = self.get_options_chain(symbol)

        # Calculate days to expiration
        exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        dte = max((exp_date - datetime.now()).days, 1) / 365.0

        # Risk-free rate assumption
        r = 0.05

        # Process each option
        gex_data = []
        print(f"[GEX] Processing {len(options)} options (DTE={dte*365:.1f} days, r={r})...")

        for opt in options:
            strike = opt['strike']
            option_type = opt['option_type']
            oi = opt.get('open_interest', 0)

            if oi == 0:
                continue

            # Get gamma and IV (from API or calculate)
            if use_api_greeks and 'greeks' in opt and opt['greeks']:
                gamma = opt['greeks'].get('gamma', 0)
                iv = opt['greeks'].get('mid_iv', 0.25)
            else:
                # Estimate IV and calculate gamma
                iv = 0.25  # Default IV assumption
                gamma = self.calculate_gamma(spot_price, strike, dte, r, iv, option_type)

            # Calculate Vanna (always calculate, not provided by API)
            vanna = self.calculate_vanna(spot_price, strike, dte, r, iv, option_type)

            # Calculate GEX contribution
            # Calls: Dealers short = they have negative gamma = positive GEX (buying dips)
            # Puts: Dealers short = they have positive gamma = negative GEX (selling rallies)
            if option_type == 'call':
                gex = gamma * oi * 100 * spot_price
                vanna_exp = vanna * oi * 100 * spot_price
            else:  # put
                gex = -gamma * oi * 100 * spot_price
                vanna_exp = vanna * oi * 100 * spot_price  # Vanna same sign for calls/puts

            gex_data.append({
                'strike': strike,
                'option_type': option_type,
                'open_interest': oi,
                'gamma': gamma,
                'vanna': vanna,
                'iv': iv,
                'gex': gex,
                'vanna_exposure': vanna_exp
            })

        df = pd.DataFrame(gex_data)

        print(f"[GEX] Processed {len(gex_data)} options with non-zero OI")

        if df.empty:
            print("[GEX/Vanna] ERROR: No options with open interest - returning empty results")
            return None, {}

        print(f"[GEX] DataFrame created with {len(df)} rows")

        # Aggregate GEX by strike
        gex_by_strike = df.groupby('strike')['gex'].sum().reset_index()
        gex_by_strike.columns = ['strike', 'net_gex']

        # Aggregate Vanna by strike
        vanna_by_strike = df.groupby('strike')['vanna_exposure'].sum().reset_index()
        vanna_by_strike.columns = ['strike', 'net_vanna']

        # Find key GEX levels
        key_levels = self._find_key_levels(gex_by_strike, spot_price)

        # Find key Vanna levels
        vanna_levels = self._find_vanna_levels(vanna_by_strike, spot_price)
        key_levels.update(vanna_levels)

        # Add wall/hotspot context so the dashboard can surface the
        # highest-impact strikes for intraday trading
        key_levels['gamma_walls'] = self._build_gamma_walls(gex_by_strike)
        key_levels['vanna_hotspots'] = self._build_vanna_hotspots(vanna_by_strike)
        key_levels['intraday_strikes'] = self._build_intraday_watchlist(
            gex_by_strike,
            vanna_by_strike,
            spot_price,
        )

        # Preserve the full curves so the dashboard can plot a strike-level overlay
        key_levels['gex_curve'] = gex_by_strike.sort_values('strike').to_dict('records')
        key_levels['vanna_curve'] = vanna_by_strike.sort_values('strike').to_dict('records')

        # Add metadata
        key_levels['spot_price'] = spot_price
        key_levels['expiration'] = expiration
        key_levels['dte'] = (exp_date - datetime.now()).days

        print(f"[GEX/Vanna] Calculated from {len(df)} options at {len(gex_by_strike)} strikes ({key_levels['dte']}DTE)")

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

    def _find_vanna_levels(self, vanna_df, spot_price):
        """
        Find key Vanna levels for support/resistance.

        Positive Vanna = Delta increases with IV increase = Support
        Negative Vanna = Delta decreases with IV increase = Resistance
        """
        # Sort by strike
        vanna_df = vanna_df.sort_values('strike')

        # Find strikes with significant Vanna exposure (abs value)
        vanna_df['abs_vanna'] = vanna_df['net_vanna'].abs()
        vanna_df = vanna_df.sort_values('abs_vanna', ascending=False)

        # Get top Vanna strikes below current price (support)
        below_price = vanna_df[vanna_df['strike'] < spot_price]
        vanna_support_1 = below_price.iloc[0]['strike'] if len(below_price) > 0 else None
        vanna_support_1_strength = below_price.iloc[0]['net_vanna'] / 1e6 if len(below_price) > 0 else 0  # Normalize
        vanna_support_2 = below_price.iloc[1]['strike'] if len(below_price) > 1 else None
        vanna_support_2_strength = below_price.iloc[1]['net_vanna'] / 1e6 if len(below_price) > 1 else 0

        # Get top Vanna strikes above current price (resistance)
        above_price = vanna_df[vanna_df['strike'] > spot_price]
        vanna_resistance_1 = above_price.iloc[0]['strike'] if len(above_price) > 0 else None
        vanna_resistance_1_strength = above_price.iloc[0]['net_vanna'] / 1e6 if len(above_price) > 0 else 0
        vanna_resistance_2 = above_price.iloc[1]['strike'] if len(above_price) > 1 else None
        vanna_resistance_2_strength = above_price.iloc[1]['net_vanna'] / 1e6 if len(above_price) > 1 else 0

        return {
            'vanna_support_1': vanna_support_1,
            'vanna_support_1_strength': vanna_support_1_strength,
            'vanna_support_2': vanna_support_2,
            'vanna_support_2_strength': vanna_support_2_strength,
            'vanna_resistance_1': vanna_resistance_1,
            'vanna_resistance_1_strength': vanna_resistance_1_strength,
            'vanna_resistance_2': vanna_resistance_2,
            'vanna_resistance_2_strength': vanna_resistance_2_strength
        }

    def _build_gamma_walls(self, gex_df, top_n: int = 5):
        """Return the biggest positive/negative gamma strikes."""

        if gex_df.empty:
            return []

        gamma_sorted = gex_df.assign(abs_gex=gex_df['net_gex'].abs())
        gamma_sorted = gamma_sorted.sort_values('abs_gex', ascending=False)

        walls = []
        for _, row in gamma_sorted.head(top_n).iterrows():
            walls.append({
                'strike': float(row['strike']),
                'net_gex': float(row['net_gex']),
                'type': 'support' if row['net_gex'] > 0 else 'resistance'
            })

        return walls

    def _build_vanna_hotspots(self, vanna_df, top_n: int = 5):
        """Return the largest vanna exposures by strike."""

        if vanna_df.empty:
            return []

        vanna_sorted = vanna_df.assign(abs_vanna=vanna_df['net_vanna'].abs())
        vanna_sorted = vanna_sorted.sort_values('abs_vanna', ascending=False)

        hotspots = []
        for _, row in vanna_sorted.head(top_n).iterrows():
            hotspots.append({
                'strike': float(row['strike']),
                'net_vanna': float(row['net_vanna'])
            })

        return hotspots

    def _build_intraday_watchlist(self, gex_df, vanna_df, spot_price: float,
                                  window_pct: float = 0.03, top_n: int = 6):
        """
        Surface the strikes most likely to matter for day trading.

        We prioritize strikes within a tight band (default ±3%) around spot
        and rank them by the combined magnitude of gamma and vanna exposure.
        """

        if gex_df.empty and vanna_df.empty:
            return []

        merged = pd.merge(gex_df, vanna_df, on='strike', how='outer').fillna(0)
        merged['distance_pct'] = (merged['strike'] - spot_price) / spot_price * 100
        merged['impact_score'] = merged['net_gex'].abs() + merged['net_vanna'].abs()

        # Focus on nearby strikes first, then fall back to top impacts overall
        nearby = merged[merged['distance_pct'].abs() <= window_pct * 100]
        if nearby.empty:
            nearby = merged

        watchlist = nearby.sort_values('impact_score', ascending=False).head(top_n)

        strikes = []
        for _, row in watchlist.iterrows():
            strikes.append({
                'strike': float(row['strike']),
                'net_gex': float(row.get('net_gex', 0)),
                'net_vanna': float(row.get('net_vanna', 0)),
                'distance_pct': float(row['distance_pct'])
            })

        return strikes

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
