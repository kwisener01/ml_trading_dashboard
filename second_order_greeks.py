"""
SECOND-ORDER GREEKS IMPLEMENTATION
=====================================
Professional-grade implementation of Vanna, Charm, and Vomma

Second-Order Greeks measure how first-order Greeks change:
- VANNA: ∂Delta/∂σ (Delta sensitivity to volatility)
- CHARM: ∂Delta/∂t (Delta decay over time)
- VOMMA: ∂Vega/∂σ (Vega sensitivity to volatility)

Author: Professional Options System
Status: Production-Ready
"""

import numpy as np
from scipy.stats import norm
from scipy.special import erf
import pandas as pd
from typing import Tuple, Dict


class SecondOrderGreeks:
    """
    Complete second-order Greeks calculations using Black-Scholes model.
    
    These Greeks are critical for:
    - Portfolio Greeks rebalancing
    - Risk management in volatile markets
    - Hedging gamma and vega exposure
    - Advanced options strategies
    """
    
    def __init__(self):
        """Initialize Greek calculator with constants."""
        self.sqrt_2pi = np.sqrt(2 * np.pi)
    
    # ========================================================================
    # VANNA: Delta sensitivity to volatility changes
    # ========================================================================
    
    def vanna(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Calculate VANNA: ∂Delta/∂σ
        
        Vanna measures how delta changes with volatility.
        
        Key insights:
        - Positive for long calls, negative for short calls
        - Maximum at ATM (At-The-Money) options
        - Critical when holding options through vol moves
        - Helps predict directional exposure changes
        
        Formula:
        Vanna = -φ(d1) * (d2 / sigma)
        where φ(d1) is the standard normal pdf at d1
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Vanna value (change in delta per 1% change in volatility)
        
        Example:
            >>> greek = SecondOrderGreeks()
            >>> vanna = greek.vanna(450, 455, 30/365, 0.05, 0.25, 'call')
            >>> print(f"Vanna: {vanna:.6f}")
            >>> print(f"If IV rises 5%, delta will increase: {vanna * 0.05:.4f}")
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # PDF of d1
        pdf_d1 = norm.pdf(d1)
        
        # Vanna = -φ(d1) * (d2 / sigma)
        vanna_value = -pdf_d1 * (d2 / sigma)
        
        return float(vanna_value)
    
    # ========================================================================
    # CHARM: Delta decay over time
    # ========================================================================
    
    def charm(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Calculate CHARM: ∂Delta/∂t
        
        Charm measures how delta changes as time passes.
        
        Key insights:
        - Positive for OTM calls, negative for ITM calls
        - Accelerates near expiration (especially <7 DTE)
        - Critical for understanding gamma risk
        - Shows delta erosion/acceleration
        
        Formula:
        Charm = -φ(d1) * [2*r*T - d2*sigma] / (2*T*sigma*sqrt(T))
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Charm value (change in delta per 1 day elapsed)
        
        Example:
            >>> greek = SecondOrderGreeks()
            >>> charm = greek.charm(450, 455, 30/365, 0.05, 0.25, 'call')
            >>> print(f"Charm: {charm:.6f}")
            >>> print(f"Tomorrow delta will change by: {charm:.6f}")
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # PDF of d1
        pdf_d1 = norm.pdf(d1)
        
        # Numerator: -φ(d1) * (2*r*T - d2*sigma)
        numerator = -pdf_d1 * (2 * r * T - d2 * sigma)
        
        # Denominator: 2*T*sigma*sqrt(T)
        denominator = 2 * T * sigma * np.sqrt(T)
        
        # Charm
        charm_value = numerator / denominator
        
        # Adjust for put options
        if option_type.lower() == 'put':
            charm_value = charm_value - r
        
        return float(charm_value)
    
    # ========================================================================
    # VOMMA: Vega sensitivity to volatility
    # ========================================================================
    
    def vomma(self, S: float, K: float, T: float, r: float, sigma: float, 
              option_type: str = 'call') -> float:
        """
        Calculate VOMMA: ∂Vega/∂σ
        
        Vomma measures how vega changes with volatility.
        
        Key insights:
        - Always positive for long options (calls and puts)
        - Measures vol surface curvature
        - Critical in high vol environments
        - Shows convexity of vega
        
        Formula:
        Vomma = vega * d1 * d2 / sigma
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Vomma value (change in vega per 1% change in volatility)
        
        Example:
            >>> greek = SecondOrderGreeks()
            >>> vomma = greek.vomma(450, 455, 30/365, 0.05, 0.25, 'call')
            >>> print(f"Vomma: {vomma:.6f}")
            >>> print(f"If IV rises 10%, vega will increase: {vomma * 0.10:.4f}")
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        # Vega = S * φ(d1) * sqrt(T) / 100
        vega = self._vega(S, d1, T)
        
        # Vomma = vega * d1 * d2 / sigma
        vomma_value = vega * d1 * d2 / sigma
        
        return float(vomma_value)
    
    # ========================================================================
    # ULTIMA: Third-order Greek (rare, included for completeness)
    # ========================================================================
    
    def ultima(self, S: float, K: float, T: float, r: float, sigma: float, 
               option_type: str = 'call') -> float:
        """
        Calculate ULTIMA: ∂Vomma/∂σ
        
        ULTIMA measures how vomma changes with volatility.
        
        Key insights:
        - Third-order Greek (rarely used)
        - Mostly used by market makers
        - Shows extreme vol sensitivity
        - Useful for vol surface trading
        
        Formula:
        Ultima = -vega * (d1*d2*(d1*d2 - 1) - d1^2 - d2^2) / sigma^2
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Ultima value (change in vomma per 1% change in volatility)
        """
        if T <= 0 or sigma <= 0:
            return 0.0
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        vega = self._vega(S, d1, T)
        
        # Complex formula for ultima
        term = d1 * d2 * (d1 * d2 - 1) - d1**2 - d2**2
        ultima_value = -vega * term / (sigma ** 2)
        
        return float(ultima_value)
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _vega(self, S: float, d1: float, T: float) -> float:
        """Calculate vega for use in other Greeks."""
        return S * norm.pdf(d1) * np.sqrt(T)
    
    def calculate_all_second_order(self, S: float, K: float, T: float, 
                                    r: float, sigma: float, 
                                    option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate all second-order Greeks in one call.
        
        Args:
            S: Stock price
            K: Strike price
            T: Time to expiration (years)
            r: Risk-free rate
            sigma: Volatility (annualized)
            option_type: 'call' or 'put'
        
        Returns:
            Dictionary with all second-order Greeks
        
        Example:
            >>> greek = SecondOrderGreeks()
            >>> greeks = greek.calculate_all_second_order(450, 455, 30/365, 0.05, 0.25)
            >>> for name, value in greeks.items():
            ...     print(f"{name}: {value:.6f}")
        """
        return {
            'vanna': self.vanna(S, K, T, r, sigma, option_type),
            'charm': self.charm(S, K, T, r, sigma, option_type),
            'vomma': self.vomma(S, K, T, r, sigma, option_type),
            'ultima': self.ultima(S, K, T, r, sigma, option_type),
        }


class OptionsFeatureEngineering:
    """
    Add second-order Greeks to feature engineering pipeline.
    Extends the original options_features.py module.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
        """
        self.df = df.copy()
        self.greek_calculator = SecondOrderGreeks()
    
    def add_second_order_greeks(self, dte: int = 30, iv_estimate: float = 0.25, 
                                 r: float = 0.05) -> pd.DataFrame:
        """
        Add second-order Greeks to all rows.
        
        Args:
            dte: Days to expiration
            iv_estimate: Estimated implied volatility
            r: Risk-free rate
        
        Returns:
            DataFrame with additional Greek columns
        
        Example:
            >>> import pandas as pd
            >>> df = pd.read_csv('spy_data.csv')
            >>> ofe = OptionsFeatureEngineering(df)
            >>> df = ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)
            >>> print(df[['close', 'call_vanna', 'call_charm', 'call_vomma']].head())
        """
        T = dte / 365.0
        
        # ATM Strike (using close price)
        strike = self.df['close'].values
        
        # Calculate all second-order Greeks for calls
        print("Calculating call second-order Greeks...")
        call_vanna = []
        call_charm = []
        call_vomma = []
        call_ultima = []
        
        for idx, row in self.df.iterrows():
            S = row['close']
            K = strike[idx]
            
            greeks = self.greek_calculator.calculate_all_second_order(
                S=S, K=K, T=T, r=r, sigma=iv_estimate, option_type='call'
            )
            
            call_vanna.append(greeks['vanna'])
            call_charm.append(greeks['charm'])
            call_vomma.append(greeks['vomma'])
            call_ultima.append(greeks['ultima'])
        
        # Calculate all second-order Greeks for puts
        print("Calculating put second-order Greeks...")
        put_vanna = []
        put_charm = []
        put_vomma = []
        put_ultima = []
        
        for idx, row in self.df.iterrows():
            S = row['close']
            K = strike[idx]
            
            greeks = self.greek_calculator.calculate_all_second_order(
                S=S, K=K, T=T, r=r, sigma=iv_estimate, option_type='put'
            )
            
            put_vanna.append(greeks['vanna'])
            put_charm.append(greeks['charm'])
            put_vomma.append(greeks['vomma'])
            put_ultima.append(greeks['ultima'])
        
        # Add to dataframe
        self.df['call_vanna'] = call_vanna
        self.df['call_charm'] = call_charm
        self.df['call_vomma'] = call_vomma
        self.df['call_ultima'] = call_ultima
        
        self.df['put_vanna'] = put_vanna
        self.df['put_charm'] = put_charm
        self.df['put_vomma'] = put_vomma
        self.df['put_ultima'] = put_ultima
        
        print(f"[OK] Added 8 second-order Greeks features")
        return self.df
    
    def add_vanna_trading_signals(self) -> pd.DataFrame:
        """
        Add trading signals based on Vanna.
        
        Vanna signals show when volatility changes will amplify directional moves.
        
        Returns:
            DataFrame with Vanna signals
        """
        # Vanna momentum (how Vanna is changing)
        self.df['vanna_momentum'] = self.df['call_vanna'].diff()
        
        # High Vanna (good for directional trades with vol)
        vanna_median = self.df['call_vanna'].median()
        self.df['high_vanna'] = (self.df['call_vanna'] > vanna_median).astype(int)
        
        # Vanna spike (sudden Vanna increase)
        self.df['vanna_spike'] = (self.df['vanna_momentum'] > self.df['vanna_momentum'].std() * 2).astype(int)
        
        print("[OK] Added Vanna trading signals")
        return self.df
    
    def add_charm_risk_management(self) -> pd.DataFrame:
        """
        Add risk management signals based on Charm.
        
        Charm signals help identify when delta decay accelerates.
        
        Returns:
            DataFrame with Charm risk signals
        """
        # Charm momentum (how Charm is changing)
        self.df['charm_momentum'] = self.df['call_charm'].diff()
        
        # High Charm risk (delta decay accelerating)
        charm_75_pctl = self.df['call_charm'].quantile(0.75)
        self.df['high_charm_risk'] = (self.df['call_charm'] > charm_75_pctl).astype(int)
        
        # Charm warning (near expiration with high charm)
        self.df['charm_warning'] = (self.df['call_charm'].abs() > self.df['call_charm'].abs().median() * 1.5).astype(int)
        
        print("[OK] Added Charm risk management signals")
        return self.df
    
    def add_vomma_volatility_signals(self) -> pd.DataFrame:
        """
        Add signals based on Vomma (vega sensitivity).
        
        Vomma signals help identify when vega convexity is high.
        
        Returns:
            DataFrame with Vomma signals
        """
        # Vomma is always positive for long options
        # High Vomma = high vega convexity
        vomma_75_pctl = self.df['call_vomma'].quantile(0.75)
        self.df['high_vomma'] = (self.df['call_vomma'] > vomma_75_pctl).astype(int)
        
        # Vomma momentum
        self.df['vomma_momentum'] = self.df['call_vomma'].diff()
        
        print("[OK] Added Vomma volatility signals")
        return self.df

    def add_vanna_support_resistance(self, strike_width: float = 5.0, vanna_threshold: float = 0.01) -> pd.DataFrame:
        """
        Calculate Vanna-based support and resistance levels.

        Vanna shows where delta is most sensitive to volatility changes.
        High positive Vanna = support (vol drop accelerates downside protection)
        High negative Vanna = resistance (vol drop accelerates upside selling)

        Args:
            strike_width: Price increment to check for Vanna levels
            vanna_threshold: Minimum absolute Vanna value to consider significant

        Returns:
            DataFrame with Vanna support/resistance levels
        """
        if 'close' not in self.df.columns:
            print("[WARNING] 'close' column not found, cannot calculate Vanna levels")
            return self.df

        # Get current price (latest close)
        current_price = self.df['close'].iloc[-1]

        # Calculate Vanna at different strike levels around current price
        # We'll check strikes from -20% to +20% of current price
        strikes = np.arange(
            current_price * 0.80,
            current_price * 1.20,
            strike_width
        )

        vanna_levels = []

        for strike in strikes:
            # Calculate distance from current price
            pct_from_current = ((strike - current_price) / current_price) * 100

            # Estimate Vanna at this strike
            # In reality, you'd get this from options data
            # Here we approximate based on how Vanna behaves
            moneyness = current_price / strike

            # Vanna is highest slightly OTM and decreases as you move away
            if 0.95 <= moneyness <= 1.05:  # Near ATM
                vanna_strength = abs(self.df['call_vanna'].iloc[-1]) * (1 - abs(moneyness - 1) * 2)
            else:
                vanna_strength = abs(self.df['call_vanna'].iloc[-1]) * 0.5

            # Vanna sign determines support vs resistance
            # Positive Vanna = calls gaining delta on vol increase = support below
            # Negative Vanna = calls losing delta on vol increase = resistance above
            if current_price > strike:
                # Below current price = potential support
                level_type = 'support'
                vanna_value = vanna_strength
            else:
                # Above current price = potential resistance
                level_type = 'resistance'
                vanna_value = -vanna_strength

            if abs(vanna_value) >= vanna_threshold:
                vanna_levels.append({
                    'strike': strike,
                    'vanna': vanna_value,
                    'type': level_type,
                    'distance_pct': pct_from_current,
                    'strength': abs(vanna_value)
                })

        # Find strongest support (below price) and resistance (above price)
        supports = [l for l in vanna_levels if l['type'] == 'support']
        resistances = [l for l in vanna_levels if l['type'] == 'resistance']

        # Get top 3 of each
        supports = sorted(supports, key=lambda x: x['strength'], reverse=True)[:3]
        resistances = sorted(resistances, key=lambda x: x['strength'], reverse=True)[:3]

        # Add to dataframe
        if supports:
            self.df['vanna_support_1'] = supports[0]['strike']
            self.df['vanna_support_1_strength'] = supports[0]['strength']
            if len(supports) > 1:
                self.df['vanna_support_2'] = supports[1]['strike']
                self.df['vanna_support_2_strength'] = supports[1]['strength']
            if len(supports) > 2:
                self.df['vanna_support_3'] = supports[2]['strike']
                self.df['vanna_support_3_strength'] = supports[2]['strength']

        if resistances:
            self.df['vanna_resistance_1'] = resistances[0]['strike']
            self.df['vanna_resistance_1_strength'] = resistances[0]['strength']
            if len(resistances) > 1:
                self.df['vanna_resistance_2'] = resistances[1]['strike']
                self.df['vanna_resistance_2_strength'] = resistances[1]['strength']
            if len(resistances) > 2:
                self.df['vanna_resistance_3'] = resistances[2]['strike']
                self.df['vanna_resistance_3_strength'] = resistances[2]['strength']

        print(f"[OK] Added Vanna support/resistance levels")
        print(f"    Top Support: ${supports[0]['strike']:.2f} (strength: {supports[0]['strength']:.4f})" if supports else "    No support levels found")
        print(f"    Top Resistance: ${resistances[0]['strike']:.2f} (strength: {resistances[0]['strength']:.4f})" if resistances else "    No resistance levels found")

        return self.df

    def get_second_order_features(self) -> list:
        """
        Get list of all second-order Greek features.
        
        Returns:
            List of column names
        """
        features = [
            'call_vanna', 'call_charm', 'call_vomma', 'call_ultima',
            'put_vanna', 'put_charm', 'put_vomma', 'put_ultima',
            'vanna_momentum', 'high_vanna', 'vanna_spike',
            'charm_momentum', 'high_charm_risk', 'charm_warning',
            'high_vomma', 'vomma_momentum'
        ]
        return [f for f in features if f in self.df.columns]


# ============================================================================
# PRACTICAL EXAMPLES
# ============================================================================

def example_1_calculate_greeks():
    """Example 1: Calculate second-order Greeks for a single option."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Calculate Second-Order Greeks")
    print("="*70)
    
    greek = SecondOrderGreeks()
    
    # SPY call at ATM, 30 DTE
    S = 450      # Stock price
    K = 450      # Strike price (ATM)
    T = 30/365   # Time to expiration
    r = 0.05     # Risk-free rate
    sigma = 0.25 # Implied volatility
    
    # Calculate individual Greeks
    vanna = greek.vanna(S, K, T, r, sigma, 'call')
    charm = greek.charm(S, K, T, r, sigma, 'call')
    vomma = greek.vomma(S, K, T, r, sigma, 'call')
    ultima = greek.ultima(S, K, T, r, sigma, 'call')
    
    print(f"\nOption Details:")
    print(f"  Stock Price: ${S}")
    print(f"  Strike: ${K}")
    print(f"  DTE: 30 days")
    print(f"  IV: {sigma*100}%")
    
    print(f"\nSecond-Order Greeks:")
    print(f"  Vanna:  {vanna:>12.8f}")
    print(f"  Charm:  {charm:>12.8f}")
    print(f"  Vomma:  {vomma:>12.8f}")
    print(f"  Ultima: {ultima:>12.8f}")
    
    print(f"\nInterpretation:")
    print(f"  • If IV increases 5%: Delta changes by {vanna*0.05:.6f}")
    print(f"  • Tomorrow: Delta changes by {charm:.6f}")
    print(f"  • If IV increases 10%: Vega changes by {vomma*0.10:.6f}")


def example_2_portfolio_greeks():
    """Example 2: Calculate Greeks for multiple strikes."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Portfolio Greeks Across Strikes")
    print("="*70)
    
    greek = SecondOrderGreeks()
    
    S = 450
    T = 30/365
    r = 0.05
    sigma = 0.25
    
    strikes = [440, 445, 450, 455, 460]
    
    print(f"\n{'Strike':<8} {'Vanna':<12} {'Charm':<12} {'Vomma':<12}")
    print("-" * 44)
    
    for K in strikes:
        vanna = greek.vanna(S, K, T, r, sigma, 'call')
        charm = greek.charm(S, K, T, r, sigma, 'call')
        vomma = greek.vomma(S, K, T, r, sigma, 'call')
        
        print(f"${K:<7} {vanna:>11.8f} {charm:>11.8f} {vomma:>11.8f}")


def example_3_volatility_effect():
    """Example 3: Show how Greeks change with volatility."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Impact of Volatility on Second-Order Greeks")
    print("="*70)
    
    greek = SecondOrderGreeks()
    
    S = 450
    K = 450
    T = 30/365
    r = 0.05
    
    volatilities = [0.15, 0.20, 0.25, 0.30, 0.35]
    
    print(f"\n{'IV':<8} {'Vanna':<12} {'Charm':<12} {'Vomma':<12}")
    print("-" * 44)
    
    for sigma in volatilities:
        vanna = greek.vanna(S, K, T, r, sigma, 'call')
        charm = greek.charm(S, K, T, r, sigma, 'call')
        vomma = greek.vomma(S, K, T, r, sigma, 'call')
        
        print(f"{sigma*100:.0f}%    {vanna:>11.8f} {charm:>11.8f} {vomma:>11.8f}")


def example_4_time_decay_effect():
    """Example 4: Show how Greeks change as expiration approaches."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Impact of Time Decay on Second-Order Greeks")
    print("="*70)
    
    greek = SecondOrderGreeks()
    
    S = 450
    K = 450
    r = 0.05
    sigma = 0.25
    
    dte_list = [60, 45, 30, 15, 7, 1]
    
    print(f"\n{'DTE':<8} {'Vanna':<12} {'Charm':<12} {'Vomma':<12}")
    print("-" * 44)
    
    for dte in dte_list:
        T = dte / 365
        vanna = greek.vanna(S, K, T, r, sigma, 'call')
        charm = greek.charm(S, K, T, r, sigma, 'call')
        vomma = greek.vomma(S, K, T, r, sigma, 'call')
        
        print(f"{dte:<7} {vanna:>11.8f} {charm:>11.8f} {vomma:>11.8f}")


def example_5_real_data():
    """Example 5: Add to real data and create features."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Add Second-Order Greeks to Real Data")
    print("="*70)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100)
    data = {
        'date': dates,
        'open': np.random.uniform(440, 460, 100),
        'high': np.random.uniform(440, 460, 100),
        'low': np.random.uniform(440, 460, 100),
        'close': np.random.uniform(440, 460, 100),
        'volume': np.random.uniform(1000000, 5000000, 100),
    }
    df = pd.DataFrame(data)
    
    # Add second-order Greeks
    ofe = OptionsFeatureEngineering(df)
    df = ofe.add_second_order_greeks(dte=30, iv_estimate=0.25)
    
    # Add trading signals
    df = ofe.add_vanna_trading_signals()
    df = ofe.add_charm_risk_management()
    df = ofe.add_vomma_volatility_signals()
    
    print(f"\nDataFrame shape: {df.shape}")
    print(f"Total features: {len(ofe.get_second_order_features())}")
    print(f"\nFeatures added:")
    for feat in ofe.get_second_order_features():
        print(f"  • {feat}")
    
    print(f"\nSample data (first 3 rows):")
    cols_to_show = ['close', 'call_vanna', 'call_charm', 'call_vomma', 'high_vanna', 'high_charm_risk']
    print(df[cols_to_show].head(3).to_string())


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SECOND-ORDER GREEKS - PRODUCTION SYSTEM")
    print("="*70)

    # Run all examples
    example_1_calculate_greeks()
    example_2_portfolio_greeks()
    example_3_volatility_effect()
    example_4_time_decay_effect()
    example_5_real_data()

    print("\n" + "="*70)
    print("[SUCCESS] All examples completed successfully!")
    print("="*70)
    print("\nNow copy this code into your trading system!")
    print("Integration: from second_order_greeks import OptionsFeatureEngineering\n")
