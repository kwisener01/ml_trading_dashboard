import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from data_collector import TradierDataCollector
from feature_engineering import FeatureEngineering
import warnings
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

warnings.filterwarnings('ignore')

class TradingPredictor:
    """
    Real-time trading predictions using trained ML models
    """
    
    def __init__(self, api_token, model_prefix='spy_trading_model', model_timestamp=None):
        """Initialize with API token and load models"""
        self.api_token = api_token  # Store API token for GEX calculator
        self.collector = TradierDataCollector(api_token)
        self.models = {}
        self.load_models(model_prefix, model_timestamp)
    
    def load_models(self, prefix, timestamp=None):
        """Load trained models"""
        import glob
        import re

        if timestamp:
            pattern = f"{prefix}_*_{timestamp}.pkl"
        else:
            # Find most recent models
            pattern = f"{prefix}_*.pkl"

        model_files = sorted(glob.glob(pattern))

        if not model_files:
            raise FileNotFoundError(f"No models found matching: {pattern}")

        for filepath in model_files:
            # Extract model type from filename
            # Format: spy_trading_model_<TYPE>_<TIMESTAMP>.pkl
            # Example: spy_trading_model_trade_quality_20251113_230157.pkl
            # We want just <TYPE> (e.g., "trade_quality")

            basename = filepath.replace('.pkl', '')

            # Use regex to extract the model type
            # Pattern: prefix_<model_type>_<14digit_timestamp>
            import re
            pattern = f"{prefix}_(.+?)_(\\d{{8}}_\\d{{6}}|\\d{{14}})$"
            match = re.match(pattern, basename)

            if match:
                model_type = match.group(1)
            else:
                # Fallback: just remove prefix and take everything before last underscore with digits
                model_part = basename.replace(f"{prefix}_", "")
                # Remove last part if it looks like a timestamp (8 or 14 digits)
                parts = model_part.split('_')
                if parts[-1].isdigit() and len(parts[-1]) in [6, 8, 14]:
                    model_type = '_'.join(parts[:-1])
                elif len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                    # Handle format like 20251113_230157
                    model_type = '_'.join(parts[:-2])
                else:
                    model_type = model_part

            # Only keep one model of each type (last one loaded wins)
            self.models[model_type] = joblib.load(filepath)
            print(f"[OK] Loaded: {model_type} from {filepath}")
    
    def get_current_data(self, symbol, lookback_periods=200):
        """
        Get recent data for making predictions
        Need enough history to calculate indicators
        """
        print(f"\nFetching current data for {symbol}...")

        # Use daily data (models were trained on daily data with VIX)
        print("Using daily data (models trained on daily timeframe)...")
        try:
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_periods)

            daily = self.collector.get_historical_quotes(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not daily.empty:
                # Add VIX data
                print("Fetching VIX data...")
                vix_data = self.collector.get_vix_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )

                if not vix_data.empty:
                    # Rename VIX close to vix
                    vix_data = vix_data.rename(columns={'close': 'vix'})
                    # Merge VIX into daily data
                    daily = pd.merge(daily, vix_data[['date', 'vix']], on='date', how='left')
                    # Forward fill VIX for any missing dates
                    daily['vix'] = daily['vix'].fillna(method='ffill')
                    print(f"VIX data added: {daily['vix'].notna().sum()} values")
                else:
                    print("Warning: VIX data unavailable, using median as fallback")
                    daily['vix'] = 20.0  # Approximate VIX median

                print(f"Using daily data: {len(daily)} bars with VIX")
                return daily
        except Exception as e:
            print(f"Error getting daily data: {e}")

        print("No data available")
        return None
    
    def prepare_features(self, df):
        """
        Engineer features from raw data
        """
        print("Engineering features...")

        fe = FeatureEngineering(df)
        fe.add_technical_indicators()
        fe.add_support_resistance_levels()
        fe.add_market_regime_features()
        fe.add_time_features()
        # Note: Vanna levels now come from real options chain via GEX calculator (0DTE/1DTE)

        # Get feature set (without creating labels since we're predicting)
        features, full_data = fe.get_features_for_ml()

        return features, full_data

    def calculate_vanna_levels(self, df):
        """
        Calculate Vanna levels separately (not used for ML predictions)
        """
        print("Calculating Vanna support/resistance levels...")

        fe = FeatureEngineering(df.copy())
        fe.add_vanna_levels()

        return fe.data

    def calculate_options_flow_data(self, symbol, current_price, vix):
        """
        Calculate comprehensive options flow metrics:
        - Charm (time decay flows)
        - Put/Call walls
        - IV metrics
        - Combined dealer flow score
        """
        flow_data = {}

        try:
            from second_order_greeks import SecondOrderGreeks

            greeks_calc = SecondOrderGreeks()

            # Calculate ATM implied volatility (using VIX as proxy)
            flow_data['iv'] = vix / 100  # Convert VIX to decimal
            flow_data['iv_percentile'] = min(100, max(0, (vix - 10) / 30 * 100))  # Rough IV percentile

            # Calculate Charm for ATM options (1-week and 1-month)
            # Charm measures delta decay - important for understanding dealer hedging flows
            try:
                charm_1w = greeks_calc.charm(
                    S=current_price,
                    K=current_price,  # ATM
                    T=7/365,  # 1 week
                    r=0.05,
                    sigma=flow_data['iv']
                )
                charm_1m = greeks_calc.charm(
                    S=current_price,
                    K=current_price,  # ATM
                    T=30/365,  # 1 month
                    r=0.05,
                    sigma=flow_data['iv']
                )
                flow_data['charm'] = charm_1w
                flow_data['charm_1m'] = charm_1m
                flow_data['charm_pressure'] = abs(charm_1w) * 100  # Scale for visualization
            except Exception as e:
                print(f"[WARN] Charm calculation failed: {e}")
                flow_data['charm'] = 0
                flow_data['charm_1m'] = 0
                flow_data['charm_pressure'] = 0

            # Calculate Put/Call wall levels (largest OI strikes)
            # These act as magnets/barriers
            try:
                # Try to get options chain data
                url = f'{self.collector.base_url}/markets/options/chains'
                chain_data = self.collector._request_json(url, params={'symbol': symbol})

                if chain_data and 'options' in chain_data and 'option' in chain_data['options']:
                    options_data = chain_data['options']['option']
                    # Process options chain to find max OI strikes
                    # This would require more detailed implementation
                    flow_data['put_wall'] = None  # Placeholder
                    flow_data['call_wall'] = None  # Placeholder
                else:
                    flow_data['put_wall'] = None
                    flow_data['call_wall'] = None
            except Exception as e:
                print(f"[WARN] Put/Call wall calculation unavailable: {e}")
                flow_data['put_wall'] = None
                flow_data['call_wall'] = None

        except Exception as e:
            print(f"[ERROR] Options flow calculation failed: {e}")
            flow_data = {
                'iv': vix / 100,
                'iv_percentile': 50,
                'charm': 0,
                'charm_1m': 0,
                'charm_pressure': 0,
                'put_wall': None,
                'call_wall': None
            }

        return flow_data

    def estimate_time_to_target(self, current_price, resistance_levels, support_levels,
                                atr, volatility_rank, trend_strength):
        """
        Estimate time to reach bounce and rejection zones

        For day trading and options:
        - Returns hours to target
        - Probability of reaching by end of day
        - Accounts for volatility, momentum, time of day
        """
        from datetime import datetime
        import pytz

        # Filter out None values and find closest levels
        resistance_levels = [r for r in resistance_levels if r and r > current_price]
        support_levels = [s for s in support_levels if s and s < current_price]

        results = {}

        # Current time for intraday calculations
        now = datetime.now(pytz.timezone('America/New_York'))
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        hours_until_close = max(0, (market_close - now).seconds / 3600)

        # Calculate for rejection zone (above current price)
        if resistance_levels:
            rejection_target = min(resistance_levels)  # Closest resistance
            distance_points = rejection_target - current_price
            distance_pct = distance_points / current_price

            # Base time estimate using ATR (average true range)
            # ATR represents typical daily movement
            # Adjust for intraday: typically 60-70% of daily range happens in trading hours
            intraday_factor = 0.65
            moves_per_day = atr * intraday_factor if atr > 0 else current_price * 0.01
            moves_per_hour = moves_per_day / 6.5  # 6.5 trading hours

            # Base hours needed
            if moves_per_hour > 0:
                base_hours = distance_points / moves_per_hour
            else:
                base_hours = 999  # Unreachable

            # Adjust for trend strength (stronger trend = faster)
            trend_factor = 0.7 + (trend_strength / 100) * 0.6  # Range: 0.7 to 1.3
            base_hours *= trend_factor

            # Adjust for volatility (higher vol = faster moves)
            vol_factor = 2.0 - volatility_rank  # Range: 1.0 to 2.0 (high vol = faster)
            base_hours *= vol_factor

            # Store rejection zone estimates
            results['hours_to_rejection_zone'] = round(base_hours, 1)
            results['rejection_zone_target'] = round(rejection_target, 2)
            results['rejection_zone_distance_pct'] = round(distance_pct * 100, 2)

            # Probability of reaching today
            if hours_until_close > 0:
                # Sigmoid function for probability
                # P = 1 / (1 + exp(k * (hours_needed - hours_available)))
                k = 0.5  # Steepness factor
                import math
                prob = 1 / (1 + math.exp(k * (base_hours - hours_until_close)))
                results['prob_reach_rejection_today'] = round(prob * 100, 1)
            else:
                results['prob_reach_rejection_today'] = 0.0
        else:
            results['hours_to_rejection_zone'] = None
            results['rejection_zone_target'] = None
            results['rejection_zone_distance_pct'] = None
            results['prob_reach_rejection_today'] = None

        # Calculate for bounce zone (below current price)
        if support_levels:
            bounce_target = max(support_levels)  # Closest support
            distance_points = current_price - bounce_target
            distance_pct = distance_points / current_price

            # Same calculation as above
            intraday_factor = 0.65
            moves_per_day = atr * intraday_factor if atr > 0 else current_price * 0.01
            moves_per_hour = moves_per_day / 6.5

            if moves_per_hour > 0:
                base_hours = distance_points / moves_per_hour
            else:
                base_hours = 999

            # Adjust for downward trend (inverse of trend strength for downside)
            trend_factor = 0.7 + ((100 - trend_strength) / 100) * 0.6
            base_hours *= trend_factor

            vol_factor = 2.0 - volatility_rank
            base_hours *= vol_factor

            results['hours_to_bounce_zone'] = round(base_hours, 1)
            results['bounce_zone_target'] = round(bounce_target, 2)
            results['bounce_zone_distance_pct'] = round(distance_pct * 100, 2)

            if hours_until_close > 0:
                import math
                k = 0.5
                prob = 1 / (1 + math.exp(k * (base_hours - hours_until_close)))
                results['prob_reach_bounce_today'] = round(prob * 100, 1)
            else:
                results['prob_reach_bounce_today'] = 0.0
        else:
            results['hours_to_bounce_zone'] = None
            results['bounce_zone_target'] = None
            results['bounce_zone_distance_pct'] = None
            results['prob_reach_bounce_today'] = None

        # Hours until market close
        results['hours_until_close'] = round(hours_until_close, 1)

        return results

    def suggest_options_strategy(self, current_price, time_estimates, predictions):
        """
        Suggest options strikes and expirations for day trading / 0DTE

        Based on:
        - Time to reach zones
        - Probability of reaching
        - Current regime
        """
        from datetime import datetime, timedelta
        import pytz

        suggestions = []

        # Get zone targets
        rejection_target = time_estimates.get('rejection_zone_target')
        bounce_target = time_estimates.get('bounce_zone_target')
        hours_to_rejection = time_estimates.get('hours_to_rejection_zone')
        hours_to_bounce = time_estimates.get('hours_to_bounce_zone')
        prob_rejection = time_estimates.get('prob_reach_rejection_today', 0)
        prob_bounce = time_estimates.get('prob_reach_bounce_today', 0)

        # Determine expiration based on time needed
        now = datetime.now(pytz.timezone('America/New_York'))
        today = now.date()

        # CALL strategy (bullish - rejection zone)
        if rejection_target and hours_to_rejection:
            # Suggest expiration
            if hours_to_rejection <= 3:
                expiration = "0DTE (Today)"
                exp_date = today
            elif hours_to_rejection <= 6.5:
                expiration = "0DTE (Today)"
                exp_date = today
            else:
                # Need more time - suggest 1DTE or weekly
                expiration = "1-2 DTE or Weekly"
                exp_date = today + timedelta(days=2)

            # Strike selection
            # For day trading: slightly OTM for better risk/reward
            strike = round(rejection_target * 0.995 / 5) * 5  # Round to nearest $5

            # Expected profit estimate (simplified)
            distance_pct = time_estimates.get('rejection_zone_distance_pct', 0)
            # Option delta approximation: ~0.4 for slight OTM
            option_move = distance_pct * 0.4 * current_price
            expected_profit_per_contract = round(option_move * 100, 0)  # x100 multiplier

            suggestions.append({
                'direction': 'CALL',
                'strike': strike,
                'expiration': expiration,
                'exp_date': exp_date.strftime('%Y-%m-%d'),
                'target': rejection_target,
                'probability': prob_rejection,
                'hours_to_target': hours_to_rejection,
                'expected_profit': expected_profit_per_contract,
                'reason': f"Rejection zone ${rejection_target:.2f} reachable in {hours_to_rejection}h ({prob_rejection}% prob)"
            })

        # PUT strategy (bearish - bounce zone)
        if bounce_target and hours_to_bounce:
            if hours_to_bounce <= 3:
                expiration = "0DTE (Today)"
                exp_date = today
            elif hours_to_bounce <= 6.5:
                expiration = "0DTE (Today)"
                exp_date = today
            else:
                expiration = "1-2 DTE or Weekly"
                exp_date = today + timedelta(days=2)

            strike = round(bounce_target * 1.005 / 5) * 5

            distance_pct = time_estimates.get('bounce_zone_distance_pct', 0)
            option_move = distance_pct * 0.4 * current_price
            expected_profit_per_contract = round(option_move * 100, 0)

            suggestions.append({
                'direction': 'PUT',
                'strike': strike,
                'expiration': expiration,
                'exp_date': exp_date.strftime('%Y-%m-%d'),
                'target': bounce_target,
                'probability': prob_bounce,
                'hours_to_target': hours_to_bounce,
                'expected_profit': expected_profit_per_contract,
                'reason': f"Bounce zone ${bounce_target:.2f} reachable in {hours_to_bounce}h ({prob_bounce}% prob)"
            })

        # Rank by probability and return top suggestion
        if suggestions:
            suggestions = sorted(suggestions, key=lambda x: x['probability'], reverse=True)
            return {
                'primary_strategy': suggestions[0],
                'alternative_strategy': suggestions[1] if len(suggestions) > 1 else None
            }
        else:
            return {'primary_strategy': None, 'alternative_strategy': None}

    def predict(self, symbol):
        """
        Make comprehensive trading prediction for a symbol
        """
        print("\n" + "="*80)
        print(f"GENERATING PREDICTIONS FOR {symbol}")
        print("="*80)
        
        # Get current data
        df = self.get_current_data(symbol)
        
        if df is None or len(df) < 50:
            print("Insufficient data for prediction")
            return None
        
        # Prepare features for ML models
        features, full_data = self.prepare_features(df)

        # Vanna levels now come from real options chain via GEX calculator (below)
        # OLD estimated Vanna calculation removed - using real 0DTE/1DTE data instead

        # Use only the latest data point for prediction
        latest_features = features.iloc[-1:].copy()
        latest_full = full_data.iloc[-1:].copy()
        
        # Handle NaN and inf values
        latest_features = latest_features.replace([np.inf, -np.inf], np.nan)
        latest_features = latest_features.fillna(latest_features.median())
        
        # Make predictions
        predictions = {}
        current_price = latest_full['close'].iloc[0]

        print(f"\n[DEBUG] Making predictions with {len(latest_features.columns)} features")
        print(f"[DEBUG] Current price: ${current_price:.2f}")
        print(f"[DEBUG] Available models: {list(self.models.keys())}")

        # 1. Trade Quality Score
        if 'trade_quality' in self.models:
            try:
                quality_proba = self.models['trade_quality'].predict_proba(latest_features)[0, 1]
                predictions['trade_quality_score'] = quality_proba * 100
                predictions['should_trade'] = quality_proba > 0.6
                print(f"[DEBUG] Trade quality: {predictions['trade_quality_score']:.1f}%")
            except Exception as e:
                print(f"[ERROR] Trade quality prediction failed: {e}")
                predictions['trade_quality_score'] = None
                predictions['should_trade'] = None
        else:
            print("[WARN] No trade_quality model found")
            predictions['trade_quality_score'] = None
            predictions['should_trade'] = None
        
        # 2. Profit Target Probability
        if 'profit_target' in self.models:
            profit_proba = self.models['profit_target'].predict_proba(latest_features)[0, 1]
            predictions['profit_probability'] = profit_proba * 100
        else:
            predictions['profit_probability'] = None
        
        # 3. Price Level Predictions
        if 'future_high' in self.models:
            pred_high = self.models['future_high'].predict(latest_features)[0]
            predictions['predicted_high'] = pred_high
            predictions['upside_target'] = ((pred_high - current_price) / current_price) * 100
        else:
            predictions['predicted_high'] = None
            predictions['upside_target'] = None
        
        if 'future_low' in self.models:
            pred_low = self.models['future_low'].predict(latest_features)[0]
            predictions['predicted_low'] = pred_low
            predictions['downside_risk'] = ((current_price - pred_low) / current_price) * 100
        else:
            predictions['predicted_low'] = None
            predictions['downside_risk'] = None
        
        # Add context
        predictions['current_price'] = current_price
        predictions['symbol'] = symbol
        predictions['timestamp'] = datetime.now().isoformat()
        
        # Market regime indicators
        predictions['trend_strength'] = latest_full['trend_strength'].iloc[0] if 'trend_strength' in latest_full else None
        predictions['choppiness'] = latest_full['choppiness'].iloc[0] if 'choppiness' in latest_full else None
        predictions['volatility_rank'] = latest_full['volatility_rank'].iloc[0] if 'volatility_rank' in latest_full else None
        predictions['optimal_hours'] = bool(latest_full['optimal_hours'].iloc[0]) if 'optimal_hours' in latest_full else None

        # Vanna levels now calculated from real options chain below (in GEX calculator section)
        # Initialize as None - will be populated from GEX calculator if successful
        predictions['vanna_support_1'] = None
        predictions['vanna_support_1_strength'] = None
        predictions['vanna_support_2'] = None
        predictions['vanna_support_2_strength'] = None
        predictions['vanna_resistance_1'] = None
        predictions['vanna_resistance_1_strength'] = None
        predictions['vanna_resistance_2'] = None
        predictions['vanna_resistance_2_strength'] = None
        predictions['dealer_pressure_score'] = None
        predictions['dealer_pressure_factors'] = []

        # GEX (Gamma Exposure) levels for hedge pressure
        try:
            from gex_calculator import GEXCalculator
            print(f"[INFO] Attempting to calculate GEX levels for {symbol}...")
            gex_calc = GEXCalculator(self.api_token)
            gex_df, gex_levels = gex_calc.calculate_gex(symbol)

            print(f"[DEBUG] GEX calculation returned: gex_df={type(gex_df)}, gex_levels={type(gex_levels)} with {len(gex_levels) if gex_levels else 0} keys")

            if gex_levels is not None and len(gex_levels) > 0:
                print(f"[SUCCESS] GEX/Vanna levels calculated: {list(gex_levels.keys())}")

                # GEX levels
                predictions['gex_support'] = gex_levels.get('max_gex_strike')
                predictions['gex_resistance'] = gex_levels.get('min_gex_strike')
                predictions['gex_zero_level'] = gex_levels.get('zero_gex_level')
                predictions['gex_regime'] = 'positive' if gex_levels.get('total_gex', 0) > 0 else 'negative'
                predictions['gex_current'] = gex_levels.get('current_gex')
                predictions['gex_error'] = None

                # Estimate dealer hedge pressure using notional GEX/Vanna around spot
                dealer_pressure_score, dealer_pressure_factors = self._compute_dealer_pressure(
                    current_price,
                    gex_levels,
                )
                predictions['dealer_pressure_score'] = dealer_pressure_score
                predictions['dealer_pressure_factors'] = dealer_pressure_factors

                # High-impact strikes for day trading
                predictions['gamma_walls'] = gex_levels.get('gamma_walls', [])
                predictions['vanna_hotspots'] = gex_levels.get('vanna_hotspots', [])
                predictions['intraday_strikes'] = gex_levels.get('intraday_strikes', [])

                # REAL Vanna levels from options chain (0DTE/1DTE)
                predictions['vanna_support_1'] = gex_levels.get('vanna_support_1')
                predictions['vanna_support_1_strength'] = gex_levels.get('vanna_support_1_strength')
                predictions['vanna_support_2'] = gex_levels.get('vanna_support_2')
                predictions['vanna_support_2_strength'] = gex_levels.get('vanna_support_2_strength')
                predictions['vanna_resistance_1'] = gex_levels.get('vanna_resistance_1')
                predictions['vanna_resistance_1_strength'] = gex_levels.get('vanna_resistance_1_strength')
                predictions['vanna_resistance_2'] = gex_levels.get('vanna_resistance_2')
                predictions['vanna_resistance_2_strength'] = gex_levels.get('vanna_resistance_2_strength')

                print(f"  - Gamma Flip: {predictions['gex_zero_level']}")
                print(f"  - GEX Support: {predictions['gex_support']}")
                print(f"  - GEX Resistance: {predictions['gex_resistance']}")
                print(f"  - Vanna Support 1: {predictions['vanna_support_1']}")
                print(f"  - Vanna Resistance 1: {predictions['vanna_resistance_1']}")
            else:
                print(f"[WARNING] GEX calculator returned empty/invalid results: gex_levels={gex_levels}")
                predictions['gex_support'] = None
                predictions['gex_resistance'] = None
                predictions['gex_zero_level'] = None
                predictions['gex_regime'] = None
                predictions['gex_current'] = None
                predictions['gex_error'] = f"Empty results from GEX calculator (returned {len(gex_levels) if gex_levels else 0} keys)"
                predictions['gamma_walls'] = []
                predictions['vanna_hotspots'] = []
                predictions['intraday_strikes'] = []
        except Exception as e:
            import traceback
            error_msg = str(e)
            print(f"[ERROR] GEX calculation failed: {error_msg}")
            print(f"[ERROR] Traceback: {traceback.format_exc()}")
            predictions['gex_support'] = None
            predictions['gex_resistance'] = None
            predictions['gex_zero_level'] = None
            predictions['gex_regime'] = None
            predictions['gex_current'] = None
            predictions['gex_error'] = error_msg
            predictions['gamma_walls'] = []
            predictions['vanna_hotspots'] = []
            predictions['intraday_strikes'] = []

        # Options flow data (IV, Charm, Put/Call walls)
        try:
            vix = latest_full['vix'].iloc[0] if 'vix' in latest_full.columns else 20.0
            flow_data = self.calculate_options_flow_data(symbol, current_price, vix)
            predictions.update(flow_data)

            # Calculate combined dealer flow score
            # Score based on: GEX regime, Vanna strength, Charm pressure
            dealer_score = 0
            if predictions.get('gex_regime') == 'positive':
                dealer_score += 30  # Dealers support mean reversion
            elif predictions.get('gex_regime') == 'negative':
                dealer_score -= 30  # Dealers amplify momentum

            # Add vanna contribution
            vanna_s1_str = predictions.get('vanna_support_1_strength', 0) or 0
            vanna_r1_str = predictions.get('vanna_resistance_1_strength', 0) or 0
            dealer_score += (vanna_s1_str - abs(vanna_r1_str)) * 50

            # Add charm contribution (time decay flows)
            charm = flow_data.get('charm', 0)
            dealer_score += charm * 10

            predictions['dealer_flow_score'] = max(-100, min(100, dealer_score))

            # Calculate Vanna × IV for trend indication
            avg_vanna = (abs(vanna_s1_str) + abs(vanna_r1_str)) / 2 if (vanna_s1_str or vanna_r1_str) else 0
            predictions['vanna_iv_trend'] = avg_vanna * flow_data.get('iv', 0.2) * 100

        except Exception as e:
            print(f"[WARNING] Could not calculate options flow data: {e}")
            predictions['iv'] = 0.2
            predictions['iv_percentile'] = 50
            predictions['charm'] = 0
            predictions['charm_pressure'] = 0
            predictions['put_wall'] = None
            predictions['call_wall'] = None
            predictions['dealer_flow_score'] = 0
            predictions['vanna_iv_trend'] = 0

        # Calculate time-to-target estimates for zones
        try:
            time_estimates = self.estimate_time_to_target(
                current_price=current_price,
                resistance_levels=[predictions.get('vanna_resistance_1'),
                                  predictions.get('vanna_resistance_2'),
                                  predictions.get('gex_resistance')],
                support_levels=[predictions.get('vanna_support_1'),
                               predictions.get('vanna_support_2'),
                               predictions.get('gex_support')],
                atr=latest_full['atr'].iloc[0] if 'atr' in latest_full else current_price * 0.01,
                volatility_rank=predictions.get('volatility_rank', 0.5),
                trend_strength=predictions.get('trend_strength', 50)
            )
            predictions.update(time_estimates)
            print(f"[INFO] Time-to-target estimates calculated:")
            print(f"  - To Rejection Zone: {time_estimates.get('hours_to_rejection_zone')} hours")
            print(f"  - To Bounce Zone: {time_estimates.get('hours_to_bounce_zone')} hours")

            # Calculate options strategy recommendations
            try:
                options_strategy = self.suggest_options_strategy(
                    current_price=current_price,
                    time_estimates=time_estimates,
                    predictions=predictions
                )
                predictions['options_primary'] = options_strategy.get('primary_strategy')
                predictions['options_alternative'] = options_strategy.get('alternative_strategy')

                if options_strategy.get('primary_strategy'):
                    primary = options_strategy['primary_strategy']
                    print(f"[INFO] Primary options strategy:")
                    print(f"  - {primary['direction']} ${primary['strike']} {primary['expiration']}")
                    print(f"  - Expected profit: ${primary['expected_profit']}/contract")
                    print(f"  - Probability: {primary['probability']}%")
            except Exception as e:
                print(f"[WARNING] Could not calculate options strategy: {e}")
                predictions['options_primary'] = None
                predictions['options_alternative'] = None

        except Exception as e:
            print(f"[WARNING] Could not calculate time-to-target: {e}")
            predictions['hours_to_rejection_zone'] = None
            predictions['hours_to_bounce_zone'] = None
            predictions['prob_reach_rejection_today'] = None
            predictions['prob_reach_bounce_today'] = None
            predictions['options_primary'] = None
            predictions['options_alternative'] = None

        return predictions

    def _compute_dealer_pressure(self, spot_price, gex_levels):
        """
        Derive a dealer hedge pressure score from notional GEX/Vanna.

        A positive score tilts toward dealer buying support; negative means
        hedging flows lean bearish. The score is bounded to [-100, 100].
        """

        factors = []
        score = 0.0

        total_gex = gex_levels.get('total_gex') or 0
        if total_gex > 0:
            score += 10
            factors.append("Positive total GEX (mean reversion)")
        elif total_gex < 0:
            score -= 10
            factors.append("Negative total GEX (momentum)")

        current_gex = gex_levels.get('current_gex')
        max_abs_wall = max(
            abs(gex_levels.get('max_gex_value') or 0),
            abs(gex_levels.get('min_gex_value') or 0),
            1e-9,
        )

        if current_gex is not None:
            gex_bias = np.tanh(current_gex / max_abs_wall) * 60
            score += gex_bias
            direction = "buying" if current_gex > 0 else "selling"
            factors.append(f"Dealers {direction} near spot: {current_gex:,.0f} GEX notional")

        flip = gex_levels.get('zero_gex_level')
        if flip:
            if spot_price > flip:
                score += 8
                factors.append(f"Price above gamma flip (${flip:.0f})")
            else:
                score -= 8
                factors.append(f"Price below gamma flip (${flip:.0f})")

        vanna_support = gex_levels.get('vanna_support_1_strength') or 0
        vanna_resistance = abs(gex_levels.get('vanna_resistance_1_strength') or 0)
        net_vanna = vanna_support - vanna_resistance

        if net_vanna != 0:
            vanna_bias = np.tanh(net_vanna) * 25
            score += vanna_bias
            bias_label = "supportive" if net_vanna > 0 else "resistive"
            factors.append(f"Vanna {bias_label}: net {net_vanna:+.2f}M deltas/IV")

        score = max(-100, min(100, score))

        return score, factors
    
    def format_signal(self, predictions):
        """
        Format prediction into actionable trading signal
        """
        if predictions is None:
            return "No signal generated"
        
        symbol = predictions['symbol']
        price = predictions['current_price']
        
        signal = []
        signal.append(f"\n{'='*80}")
        signal.append(f"TRADING SIGNAL: {symbol} @ ${price:.2f}")
        signal.append(f"Time: {predictions['timestamp']}")
        signal.append(f"{'='*80}\n")
        
        # Trade Quality Assessment
        quality = predictions.get('trade_quality_score')
        should_trade = predictions.get('should_trade')
        
        if quality is not None:
            signal.append(f"[TARGET] TRADE QUALITY: {quality:.1f}/100")

            if should_trade:
                signal.append("[SIGNAL] TRADEABLE SETUP")
            else:
                signal.append("[AVOID] SIGNAL: LOW QUALITY SETUP")
                signal.append("\nReasons to avoid:")
                
                if predictions.get('choppiness', 0) > 50:
                    signal.append("  - Market is choppy")
                if predictions.get('trend_strength', 0) < 20:
                    signal.append("  - Weak trend")
                if not predictions.get('optimal_hours', True):
                    signal.append("  - Outside optimal trading hours")
        
        signal.append("")
        
        # Win Probability
        profit_prob = predictions.get('profit_probability')
        if profit_prob is not None:
            signal.append(f"[PROBABILITY] Win Probability: {profit_prob:.1f}%")
        
        signal.append("")
        
        # Price Targets
        signal.append("[PRICE TARGETS]")

        pred_high = predictions.get('predicted_high')
        upside = predictions.get('upside_target')
        if pred_high is not None and upside is not None:
            signal.append(f"  [UP] Upside Target: ${pred_high:.2f} (+{upside:.2f}%)")

        pred_low = predictions.get('predicted_low')
        downside = predictions.get('downside_risk')
        if pred_low is not None and downside is not None:
            signal.append(f"  [DOWN] Downside Stop:  ${pred_low:.2f} (-{downside:.2f}%)")
        
        # Risk/Reward
        if upside is not None and downside is not None and downside > 0:
            rr_ratio = upside / downside
            signal.append(f"\n[R/R] Risk/Reward Ratio: {rr_ratio:.2f}:1")
        
        signal.append("")
        
        # Market Regime
        signal.append("[MARKET CONDITIONS]")
        
        trend = predictions.get('trend_strength')
        if trend is not None:
            trend_status = "Strong" if trend > 40 else "Moderate" if trend > 20 else "Weak"
            signal.append(f"  - Trend Strength: {trend_status} ({trend:.1f})")
        
        chop = predictions.get('choppiness')
        if chop is not None:
            chop_status = "High" if chop > 50 else "Moderate" if chop > 35 else "Low"
            signal.append(f"  - Choppiness: {chop_status} ({chop:.1f})")
        
        vol = predictions.get('volatility_rank')
        if vol is not None:
            vol_status = "High" if vol > 0.7 else "Moderate" if vol > 0.3 else "Low"
            signal.append(f"  - Volatility: {vol_status} ({vol:.2f})")
        
        signal.append("")
        signal.append("="*80)
        
        return "\n".join(signal)
    
    def generate_multi_symbol_signals(self, symbols):
        """
        Generate signals for multiple symbols
        Returns ranked list of opportunities
        """
        all_predictions = []
        
        for symbol in symbols:
            try:
                pred = self.predict(symbol)
                if pred:
                    all_predictions.append(pred)
            except Exception as e:
                print(f"Error predicting {symbol}: {e}")
        
        # Rank by trade quality score (handle None values)
        all_predictions.sort(key=lambda x: x.get('trade_quality_score') or 0, reverse=True)
        
        return all_predictions
    
    def save_prediction(self, predictions, filename='prediction_log.csv'):
        """Save prediction to log file"""
        pred_df = pd.DataFrame([predictions])
        
        try:
            existing = pd.read_csv(filename)
            pred_df = pd.concat([existing, pred_df], ignore_index=True)
        except FileNotFoundError:
            pass
        
        pred_df.to_csv(filename, index=False)
        print(f"\n[OK] Prediction logged to: {filename}")


def main():
    """Example usage"""
    API_TOKEN = os.getenv('TRADIER_API_TOKEN')
    
    if not API_TOKEN:
        print("❌ ERROR: TRADIER_API_TOKEN not found!")
        print("\n📋 SETUP REQUIRED:")
        print("1. Copy .env.example to .env")
        print("2. Edit .env and add your API token")
        print("3. Run this script again")
        return
    
    print("Initializing Trading Predictor...")
    predictor = TradingPredictor(API_TOKEN)
    
    # Single symbol prediction
    symbol = 'SPY'
    predictions = predictor.predict(symbol)
    
    if predictions:
        # Display formatted signal
        print(predictor.format_signal(predictions))
        
        # Save prediction
        predictor.save_prediction(predictions)
    
    # Multi-symbol analysis
    print("\n\n" + "#"*80)
    print("SCANNING MULTIPLE SYMBOLS")
    print("#"*80)
    
    symbols = ['SPY', 'QQQ']
    all_signals = predictor.generate_multi_symbol_signals(symbols)
    
    print("\n[RANKED OPPORTUNITIES]")
    print("-" * 80)
    
    for i, pred in enumerate(all_signals, 1):
        quality = pred.get('trade_quality_score', 0)
        should_trade = pred.get('should_trade', False)
        symbol = pred['symbol']
        price = pred['current_price']
        
        status = "[TRADE]" if should_trade else "[SKIP]"
        quality_str = f"{quality:5.1f}" if quality is not None else "  N/A"
        print(f"{i}. {symbol:5} @ ${price:7.2f} | Quality: {quality_str}/100 | {status}")


if __name__ == "__main__":
    main()
