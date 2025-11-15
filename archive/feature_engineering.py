import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineering:
    """
    Create features for ML model to predict:
    - Support/Resistance levels
    - Price targets
    - When NOT to trade
    """
    
    def __init__(self, data):
        """Initialize with price data DataFrame"""
        self.data = data.copy()
        self.data = self.data.sort_values('date' if 'date' in self.data.columns else 'time')
        self.data = self.data.reset_index(drop=True)
    
    def add_technical_indicators(self):
        """Add standard technical indicators"""
        df = self.data
        
        # Moving Averages
        for period in [5, 10, 20, 50, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # RSI
        df['rsi_14'] = self._calculate_rsi(df['close'], 14)
        df['rsi_7'] = self._calculate_rsi(df['close'], 7)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ATR (Average True Range) - Volatility
        df['atr'] = self._calculate_atr(df, 14)
        
        # Volume indicators
        df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        
        # Price momentum
        for period in [1, 5, 10, 20]:
            df[f'return_{period}'] = df['close'].pct_change(period)
        
        self.data = df
        return self
    
    def add_support_resistance_levels(self, lookback=20):
        """
        Identify support and resistance levels
        These become prediction targets for the ML model
        """
        df = self.data
        
        # Rolling highs and lows
        df['resistance_1'] = df['high'].rolling(window=lookback).max()
        df['support_1'] = df['low'].rolling(window=lookback).min()
        
        # Distance from support/resistance
        df['dist_from_resistance'] = (df['resistance_1'] - df['close']) / df['close']
        df['dist_from_support'] = (df['close'] - df['support_1']) / df['close']
        
        # Pivot points (classic)
        df['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        df['r1'] = 2 * df['pivot'] - df['low']
        df['s1'] = 2 * df['pivot'] - df['high']
        df['r2'] = df['pivot'] + (df['high'] - df['low'])
        df['s2'] = df['pivot'] - (df['high'] - df['low'])
        
        self.data = df
        return self
    
    def add_market_regime_features(self):
        """
        Detect market regime (trending, choppy, volatile)
        Critical for "when NOT to trade"
        """
        df = self.data
        
        # Trend strength (ADX-like)
        df['trend_strength'] = self._calculate_trend_strength(df)
        
        # Choppiness Index
        df['choppiness'] = self._calculate_choppiness(df, 14)
        
        # Volatility regime
        df['volatility_rank'] = df['atr'].rolling(window=50).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
        )
        
        # Price range compression (tight ranges = avoid trading)
        df['range_compression'] = (df['high'] - df['low']) / df['close']
        df['avg_range'] = df['range_compression'].rolling(window=20).mean()
        df['range_vs_avg'] = df['range_compression'] / df['avg_range']
        
        # Consecutive up/down bars (momentum)
        df['consecutive_up'] = (df['close'] > df['close'].shift(1)).astype(int)
        df['consecutive_down'] = (df['close'] < df['close'].shift(1)).astype(int)
        df['momentum_streak'] = df.groupby(
            (df['consecutive_up'] != df['consecutive_up'].shift()).cumsum()
        )['consecutive_up'].transform('sum')
        
        self.data = df
        return self
    
    def add_time_features(self):
        """
        Time-based features (CRITICAL for 0DTE)
        """
        df = self.data
        
        # Determine if using date or time column
        time_col = 'date' if 'date' in df.columns else 'time'
        
        if time_col in df.columns:
            df['hour'] = pd.to_datetime(df[time_col]).dt.hour
            df['minute'] = pd.to_datetime(df[time_col]).dt.minute
            df['day_of_week'] = pd.to_datetime(df[time_col]).dt.dayofweek
            
            # Avoid first/last 30 minutes (high volatility, low quality)
            df['market_open_30'] = ((df['hour'] == 9) & (df['minute'] < 60)).astype(int)
            df['market_close_30'] = ((df['hour'] == 15) & (df['minute'] >= 30)).astype(int)
            df['avoid_time'] = df['market_open_30'] | df['market_close_30']
            
            # Best trading hours (10am-3pm typically)
            df['optimal_hours'] = ((df['hour'] >= 10) & (df['hour'] < 15)).astype(int)
            
            # Monday/Friday effects
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        self.data = df
        return self
    
    def add_option_flow_features(self, option_data=None):
        """
        Option flow indicators (if option data available)
        """
        if option_data is not None and not option_data.empty:
            df = self.data
            
            # Add call/put volume ratio
            calls = option_data[option_data['option_type'] == 'call']
            puts = option_data[option_data['option_type'] == 'put']
            
            call_volume = calls['volume'].sum()
            put_volume = puts['volume'].sum()
            
            df['call_put_ratio'] = call_volume / put_volume if put_volume > 0 else 1
            
            # Implied volatility rank
            if 'mid_iv' in option_data.columns:
                df['iv_rank'] = option_data['mid_iv'].mean()
            
            self.data = df
        
        return self
    
    def create_target_labels(self, forward_periods=12, profit_threshold=0.005, loss_threshold=-0.003):
        """
        Create labels for supervised learning:
        - profit_target_hit: Did price hit profit target?
        - stop_loss_hit: Did price hit stop loss?
        - trade_quality: Should we trade? (1=yes, 0=no)
        - future_high/low: Support/resistance predictions
        """
        df = self.data
        
        # Future price movements
        df['future_high'] = df['high'].shift(-forward_periods).rolling(window=forward_periods).max()
        df['future_low'] = df['low'].shift(-forward_periods).rolling(window=forward_periods).min()
        df['future_return'] = df['close'].shift(-forward_periods).pct_change(forward_periods)
        
        # Binary targets
        df['profit_target_hit'] = (df['future_high'] - df['close']) / df['close'] >= profit_threshold
        df['stop_loss_hit'] = (df['future_low'] - df['close']) / df['close'] <= loss_threshold
        
        # Trade quality (avoid when choppy, low volume, or extreme times)
        df['trade_quality'] = (
            (df['choppiness'] < 50) &  # Not too choppy
            (df['volume_ratio'] > 0.8) &  # Decent volume
            (~df['avoid_time'].astype(bool)) &  # Good time of day
            (df['trend_strength'] > 20) &  # Some trend present
            (df['range_vs_avg'] > 0.5)  # Not too compressed
        ).astype(int)
        
        # Expected move (for setting targets)
        df['expected_move_up'] = df['close'] + (df['atr'] * 2)
        df['expected_move_down'] = df['close'] - (df['atr'] * 2)
        
        self.data = df
        return self
    
    def get_features_for_ml(self):
        """
        Return clean feature set for ML model
        """
        # Features to use (exclude raw OHLCV and target labels)
        exclude_cols = [
            'date', 'time', 'open', 'high', 'low', 'close', 'volume',
            'future_high', 'future_low', 'future_return',
            'profit_target_hit', 'stop_loss_hit', 'trade_quality',
            'expected_move_up', 'expected_move_down'
        ]
        
        feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        
        return self.data[feature_cols], self.data
    
    # Helper methods
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return macd, signal_line, histogram
    
    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def _calculate_trend_strength(self, df, period=14):
        """Calculate trend strength (simplified ADX)"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = self._calculate_atr(df, period)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / tr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / tr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        
        return dx.rolling(window=period).mean()
    
    def _calculate_choppiness(self, df, period=14):
        """
        Calculate Choppiness Index
        High values = choppy (don't trade)
        Low values = trending (good to trade)
        """
        high = df['high']
        low = df['low']
        
        atr_sum = (high - low).rolling(window=period).sum()
        range_sum = high.rolling(window=period).max() - low.rolling(window=period).min()
        
        chop = 100 * np.log10(atr_sum / range_sum) / np.log10(period)
        
        return chop


if __name__ == "__main__":
    # Example usage
    print("Loading sample data...")
    
    # Load data collected from data_collector.py
    try:
        df = pd.read_csv('spy_training_data_intraday.csv')
        print(f"Loaded {len(df)} rows")
        
        # Rename 'time' to 'date' if needed
        if 'time' in df.columns and 'date' not in df.columns:
            df['date'] = df['time']
        
        # Feature engineering
        print("\nCreating features...")
        fe = FeatureEngineering(df)
        
        fe.add_technical_indicators()
        print("✓ Technical indicators")
        
        fe.add_support_resistance_levels()
        print("✓ Support/Resistance levels")
        
        fe.add_market_regime_features()
        print("✓ Market regime features")
        
        fe.add_time_features()
        print("✓ Time features")
        
        fe.create_target_labels()
        print("✓ Target labels")
        
        # Get final dataset
        features, full_data = fe.get_features_for_ml()
        
        print(f"\nFeature set created:")
        print(f"- Total features: {len(features.columns)}")
        print(f"- Total samples: {len(full_data)}")
        print(f"- Trade quality distribution: {full_data['trade_quality'].value_counts().to_dict()}")
        
        # Save
        full_data.to_csv('spy_ml_dataset.csv', index=False)
        print("\n✓ Saved to: spy_ml_dataset.csv")
        
    except FileNotFoundError:
        print("Run data_collector.py first to collect training data!")
