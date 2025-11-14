"""
DOE-based Hyperparameter Optimization using Latin Hypercube Sampling

Uses Design of Experiments methodology to efficiently explore hyperparameter space
and find optimal ML model configurations.
"""

import pandas as pd
import numpy as np
from scipy.stats import qmc
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, mean_absolute_error
import xgboost as xgb
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class DOEHyperparameterOptimizer:
    """
    Design of Experiments approach to hyperparameter optimization.
    Uses Latin Hypercube Sampling for efficient parameter space exploration.
    """

    def __init__(self, n_samples=50, cv_splits=3):
        """
        Args:
            n_samples: Number of DOE samples to test (default 50)
            cv_splits: Number of cross-validation splits
        """
        self.n_samples = n_samples
        self.cv_splits = cv_splits
        self.results = []
        self.best_params = {}

    def define_parameter_space(self, model_type='classifier'):
        """
        Define hyperparameter search space for XGBoost models

        Returns:
            dict: Parameter bounds for DOE sampling
        """
        if model_type == 'classifier':
            return {
                'n_estimators': (50, 500),      # Number of trees
                'max_depth': (3, 12),            # Tree depth
                'learning_rate': (0.01, 0.3),    # Step size
                'subsample': (0.6, 1.0),         # Row sampling
                'colsample_bytree': (0.6, 1.0),  # Column sampling
                'min_child_weight': (1, 10),     # Minimum leaf weight
                'gamma': (0, 5),                 # Min loss reduction for split
                'reg_alpha': (0, 1),             # L1 regularization
                'reg_lambda': (0, 2)             # L2 regularization
            }
        else:  # regressor
            return {
                'n_estimators': (50, 500),
                'max_depth': (3, 12),
                'learning_rate': (0.01, 0.3),
                'subsample': (0.6, 1.0),
                'colsample_bytree': (0.6, 1.0),
                'min_child_weight': (1, 10),
                'gamma': (0, 5),
                'reg_alpha': (0, 1),
                'reg_lambda': (0, 2)
            }

    def generate_doe_samples(self, param_space):
        """
        Generate DOE samples using Latin Hypercube Sampling

        Args:
            param_space: Dictionary of parameter bounds

        Returns:
            list of parameter dictionaries to test
        """
        n_params = len(param_space)
        param_names = list(param_space.keys())

        # Generate Latin Hypercube samples (0 to 1)
        sampler = qmc.LatinHypercube(d=n_params, seed=42)
        samples = sampler.random(n=self.n_samples)

        # Scale samples to actual parameter ranges
        param_configs = []
        for sample in samples:
            config = {}
            for i, param_name in enumerate(param_names):
                lower, upper = param_space[param_name]
                scaled_value = lower + sample[i] * (upper - lower)

                # Integer parameters
                if param_name in ['n_estimators', 'max_depth', 'min_child_weight']:
                    config[param_name] = int(round(scaled_value))
                else:
                    config[param_name] = float(scaled_value)

            param_configs.append(config)

        return param_configs

    def evaluate_params_cv(self, params, X, y, model_type='classifier'):
        """
        Evaluate parameter configuration using time series cross-validation

        Args:
            params: Hyperparameter configuration
            X: Features
            y: Target
            model_type: 'classifier' or 'regressor'

        Returns:
            Mean cross-validation score
        """
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []

        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Create and train model
            if model_type == 'classifier':
                model = xgb.XGBClassifier(**params, random_state=42, eval_metric='logloss')
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_val)
                score = accuracy_score(y_val, y_pred)
            else:  # regressor
                model = xgb.XGBRegressor(**params, random_state=42)
                model.fit(X_train, y_train, verbose=False)
                y_pred = model.predict(X_val)
                score = -mean_absolute_error(y_val, y_pred)  # Negative MAE (higher is better)

            scores.append(score)

        return np.mean(scores)

    def optimize(self, X, y, model_type='classifier', target_name='model'):
        """
        Run DOE-based hyperparameter optimization

        Args:
            X: Feature DataFrame
            y: Target Series
            model_type: 'classifier' or 'regressor'
            target_name: Name for logging purposes

        Returns:
            Best parameters dictionary
        """
        print(f"\n{'='*80}")
        print(f"DOE HYPERPARAMETER OPTIMIZATION: {target_name}")
        print(f"{'='*80}")
        print(f"Method: Latin Hypercube Sampling")
        print(f"Samples: {self.n_samples}")
        print(f"CV Splits: {self.cv_splits}")

        # Define parameter space
        param_space = self.define_parameter_space(model_type)

        # Generate DOE samples
        param_configs = self.generate_doe_samples(param_space)
        print(f"\nGenerated {len(param_configs)} parameter configurations")

        # Evaluate each configuration
        print("\nEvaluating configurations...")
        results = []

        for i, params in enumerate(param_configs):
            score = self.evaluate_params_cv(params, X, y, model_type)
            results.append({
                'config_id': i,
                'params': params,
                'score': score
            })

            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i+1}/{len(param_configs)} configurations...")

        # Sort by score (descending)
        results.sort(key=lambda x: x['score'], reverse=True)
        self.results = results

        # Best configuration
        best = results[0]
        self.best_params[target_name] = best['params']

        print(f"\n{'='*80}")
        print(f"OPTIMIZATION COMPLETE")
        print(f"{'='*80}")
        print(f"Best Score: {best['score']:.4f}")
        print(f"\nBest Parameters:")
        for param, value in best['params'].items():
            print(f"  {param:20} = {value}")

        # Show top 5 configurations
        print(f"\nTop 5 Configurations:")
        for i, result in enumerate(results[:5]):
            print(f"\n  Rank {i+1}: Score = {result['score']:.4f}")
            print(f"    n_estimators={result['params']['n_estimators']}, "
                  f"max_depth={result['params']['max_depth']}, "
                  f"learning_rate={result['params']['learning_rate']:.3f}")

        return best['params']

    def save_results(self, filename='doe_optimization_results.json'):
        """Save optimization results to JSON file"""
        output = {
            'timestamp': datetime.now().isoformat(),
            'n_samples': self.n_samples,
            'cv_splits': self.cv_splits,
            'best_params': self.best_params,
            'all_results': [
                {
                    'config_id': r['config_id'],
                    'score': float(r['score']),
                    'params': r['params']
                }
                for r in self.results[:20]  # Save top 20
            ]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\n[OK] Saved results to: {filename}")


def optimize_all_models(df):
    """
    Optimize hyperparameters for all trading models using DOE

    Args:
        df: Training dataset with features and targets

    Returns:
        Dictionary of optimized parameters for each model
    """
    print("\n" + "#"*80)
    print("DOE HYPERPARAMETER OPTIMIZATION - ALL MODELS")
    print("#"*80)

    optimizer = DOEHyperparameterOptimizer(n_samples=50, cv_splits=3)

    # Prepare data
    exclude = [
        'date', 'time', 'open', 'high', 'low', 'close', 'volume',
        'future_high', 'future_low', 'future_return',
        'profit_target_hit', 'stop_loss_hit', 'trade_quality',
        'expected_move_up', 'expected_move_down'
    ]

    # Clean data
    df_clean = df.dropna()
    df_clean = df_clean.replace([np.inf, -np.inf], np.nan).fillna(df_clean.median())

    all_best_params = {}

    # 1. Optimize Trade Quality Classifier
    if 'trade_quality' in df_clean.columns:
        print("\n[1/4] Optimizing Trade Quality Classifier...")
        feature_cols = [col for col in df_clean.columns if col not in exclude + ['trade_quality']]
        X = df_clean[feature_cols]
        y = df_clean['trade_quality']

        best_params = optimizer.optimize(X, y, model_type='classifier', target_name='trade_quality')
        all_best_params['trade_quality'] = best_params

    # 2. Optimize Profit Target Classifier
    if 'profit_target_hit' in df_clean.columns:
        print("\n[2/4] Optimizing Profit Target Classifier...")
        feature_cols = [col for col in df_clean.columns if col not in exclude + ['profit_target_hit']]
        X = df_clean[feature_cols]
        y = df_clean['profit_target_hit']

        best_params = optimizer.optimize(X, y, model_type='classifier', target_name='profit_target')
        all_best_params['profit_target'] = best_params

    # 3. Optimize Future High Regressor
    if 'future_high' in df_clean.columns:
        print("\n[3/4] Optimizing Future High Regressor...")
        feature_cols = [col for col in df_clean.columns if col not in exclude + ['future_high']]
        X = df_clean[feature_cols]
        y = df_clean['future_high']

        best_params = optimizer.optimize(X, y, model_type='regressor', target_name='future_high')
        all_best_params['future_high'] = best_params

    # 4. Optimize Future Low Regressor
    if 'future_low' in df_clean.columns:
        print("\n[4/4] Optimizing Future Low Regressor...")
        feature_cols = [col for col in df_clean.columns if col not in exclude + ['future_low']]
        X = df_clean[feature_cols]
        y = df_clean['future_low']

        best_params = optimizer.optimize(X, y, model_type='regressor', target_name='future_low')
        all_best_params['future_low'] = best_params

    # Save all results
    optimizer.save_results('doe_optimization_results.json')

    print("\n" + "#"*80)
    print("DOE OPTIMIZATION COMPLETE - ALL MODELS")
    print("#"*80)
    print("\nOptimized parameters saved. Use these in train_models.py for best performance.")

    return all_best_params


if __name__ == "__main__":
    # Load training data
    try:
        df = pd.read_csv('spy_ml_dataset.csv')
        print(f"Loaded {len(df)} samples")

        # Run DOE optimization
        best_params = optimize_all_models(df)

        print("\n[NEXT STEP] Update train_models.py with optimized hyperparameters")

    except FileNotFoundError:
        print("ERROR: spy_ml_dataset.csv not found!")
        print("Run feature_engineering.py first to create the dataset.")
