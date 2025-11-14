import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingMLModel:
    """
    Machine Learning models for:
    1. Trade Quality Classifier (when NOT to trade)
    2. Price Target Regressor (where price will go)
    3. Support/Resistance Predictor
    """
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.metrics = {}
    
    def prepare_data(self, df, target_col, feature_cols=None):
        """
        Prepare data for training with proper time series handling
        """
        # Remove NaN values
        df = df.dropna()
        
        # If no feature columns specified, use all except targets and time
        if feature_cols is None:
            exclude = [
                'date', 'time', 'open', 'high', 'low', 'close', 'volume',
                'future_high', 'future_low', 'future_return',
                'profit_target_hit', 'stop_loss_hit', 'trade_quality',
                'expected_move_up', 'expected_move_down'
            ]
            feature_cols = [col for col in df.columns if col not in exclude and col != target_col]
        
        X = df[feature_cols]
        y = df[target_col]
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        return X, y, feature_cols
    
    def train_trade_quality_classifier(self, df, test_size=0.2):
        """
        Train classifier to predict if we should take the trade
        This is the "when NOT to trade" model - MOST IMPORTANT
        """
        print("\n" + "="*80)
        print("TRAINING TRADE QUALITY CLASSIFIER")
        print("="*80)
        
        X, y, feature_cols = self.prepare_data(df, 'trade_quality')
        
        # Time series split (respect temporal order)
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Positive class: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        # Train XGBoost (usually best for tabular data)
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        self.models['trade_quality'] = model
        self.feature_importance['trade_quality'] = importance_df
        self.metrics['trade_quality'] = metrics
        
        return model, metrics
    
    def train_profit_target_classifier(self, df, test_size=0.2):
        """
        Train classifier to predict if profit target will be hit
        """
        print("\n" + "="*80)
        print("TRAINING PROFIT TARGET CLASSIFIER")
        print("="*80)
        
        X, y, feature_cols = self.prepare_data(df, 'profit_target_hit')
        
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Positive class: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
        
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1': f1_score(y_test, y_pred, zero_division=0)
        }
        
        print("\nPerformance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        self.models['profit_target'] = model
        self.metrics['profit_target'] = metrics
        
        return model, metrics
    
    def train_price_level_regressor(self, df, test_size=0.2):
        """
        Train regressor to predict future high/low levels
        """
        print("\n" + "="*80)
        print("TRAINING PRICE LEVEL REGRESSOR")
        print("="*80)
        
        # Train two models: one for future highs, one for future lows
        results = {}
        
        for target in ['future_high', 'future_low']:
            print(f"\n[Training] {target}...")
            
            X, y, feature_cols = self.prepare_data(df, target)
            
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            model = xgb.XGBRegressor(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)

            # Calculate percentage error (more meaningful for trading)
            # Get current prices from the same filtered dataset used for training
            valid_df = df.dropna(subset=feature_cols + [target])
            current_prices = valid_df.iloc[split_idx:]['close'].values
            # Ensure arrays are same length
            min_len = min(len(y_pred), len(y_test), len(current_prices))
            pct_error = np.abs((y_pred[:min_len] - y_test[:min_len]) / current_prices[:min_len]) * 100
            
            metrics = {
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred),
                'mean_pct_error': np.mean(pct_error),
                'median_pct_error': np.median(pct_error)
            }
            
            print(f"MAE: ${metrics['mae']:.2f}")
            print(f"R²: {metrics['r2']:.4f}")
            print(f"Mean % Error: {metrics['mean_pct_error']:.2f}%")
            print(f"Median % Error: {metrics['median_pct_error']:.2f}%")
            
            self.models[target] = model
            self.metrics[target] = metrics
            results[target] = (model, metrics)
        
        return results
    
    def train_all_models(self, df):
        """Train all models at once"""
        print("\n" + "#"*80)
        print("TRAINING ALL MODELS")
        print("#"*80)
        
        # 1. Trade Quality (most important - filters bad setups)
        self.train_trade_quality_classifier(df)
        
        # 2. Profit Target (predicts success probability)
        self.train_profit_target_classifier(df)
        
        # 3. Price Levels (predicts where price will go)
        self.train_price_level_regressor(df)
        
        print("\n" + "#"*80)
        print("ALL MODELS TRAINED SUCCESSFULLY")
        print("#"*80)
        
        return self.models
    
    def predict(self, X):
        """
        Make predictions with all models
        Returns comprehensive trading signal
        """
        predictions = {}
        
        # Prepare features
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # 1. Trade Quality Score (0-100)
        if 'trade_quality' in self.models:
            quality_proba = self.models['trade_quality'].predict_proba(X)[:, 1]
            predictions['trade_quality_score'] = quality_proba * 100
            predictions['should_trade'] = quality_proba > 0.6  # Threshold
        
        # 2. Profit Target Probability
        if 'profit_target' in self.models:
            profit_proba = self.models['profit_target'].predict_proba(X)[:, 1]
            predictions['profit_probability'] = profit_proba * 100
        
        # 3. Price Levels
        if 'future_high' in self.models:
            predictions['predicted_high'] = self.models['future_high'].predict(X)
        
        if 'future_low' in self.models:
            predictions['predicted_low'] = self.models['future_low'].predict(X)
        
        return predictions
    
    def save_models(self, prefix='trading_model'):
        """Save all trained models"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for name, model in self.models.items():
            filename = f"{prefix}_{name}_{timestamp}.pkl"
            joblib.dump(model, filename)
            print(f"[OK] Saved: {filename}")
        
        # Save metrics
        metrics_file = f"{prefix}_metrics_{timestamp}.json"
        with open(metrics_file, 'w') as f:
            # Convert numpy types to Python types
            metrics_serializable = {}
            for model_name, model_metrics in self.metrics.items():
                metrics_serializable[model_name] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in model_metrics.items()
                }
            json.dump(metrics_serializable, f, indent=2)
        print(f"[OK] Saved metrics: {metrics_file}")
        
        # Save feature importance
        for name, importance_df in self.feature_importance.items():
            importance_file = f"{prefix}_importance_{name}_{timestamp}.csv"
            importance_df.to_csv(importance_file, index=False)
            print(f"[OK] Saved importance: {importance_file}")
    
    def load_models(self, prefix='trading_model', timestamp=None):
        """Load previously trained models"""
        import glob
        
        if timestamp:
            pattern = f"{prefix}_*_{timestamp}.pkl"
        else:
            pattern = f"{prefix}_*.pkl"
        
        model_files = glob.glob(pattern)
        
        for filepath in model_files:
            model_name = filepath.split('_')[2]  # Extract model name
            self.models[model_name] = joblib.load(filepath)
            print(f"✓ Loaded: {filepath}")
        
        return self.models


def main():
    """Main training pipeline"""
    print("Loading ML dataset...")
    
    try:
        df = pd.read_csv('spy_ml_dataset.csv')
        print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        # Initialize model trainer
        trainer = TradingMLModel()
        
        # Train all models
        models = trainer.train_all_models(df)
        
        # Save models
        trainer.save_models('spy_trading_model')
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE!")
        print("="*80)
        print("\nNext steps:")
        print("1. Test predictions with real-time data")
        print("2. Backtest on historical data")
        print("3. Deploy to Lindy agent")
        
    except FileNotFoundError:
        print("ERROR: spy_ml_dataset.csv not found!")
        print("Run feature_engineering.py first to create the dataset.")
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
