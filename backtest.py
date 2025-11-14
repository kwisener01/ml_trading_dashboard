"""
Backtesting module for ML Trading System

Simulates trading with ML predictions on historical data to measure performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from feature_engineering import FeatureEngineering
from train_models import TradingMLModel
import warnings
warnings.filterwarnings('ignore')


class TradingBacktest:
    """
    Backtest ML trading predictions on historical data
    """

    def __init__(self, models, initial_capital=10000):
        """
        Args:
            models: Dictionary of trained ML models
            initial_capital: Starting capital for backtest
        """
        self.models = models
        self.initial_capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def run_backtest(self, df, start_date=None, end_date=None,
                     trade_quality_threshold=60, profit_target_pct=0.5,
                     stop_loss_pct=0.3, position_size_pct=0.1):
        """
        Run backtest on historical data

        Args:
            df: DataFrame with features and OHLCV data
            start_date: Start date for backtest (default: last 30 days)
            end_date: End date for backtest (default: most recent)
            trade_quality_threshold: Minimum quality score to take trade (0-100)
            profit_target_pct: Profit target as % of price (0.5 = 0.5%)
            stop_loss_pct: Stop loss as % of price (0.3 = 0.3%)
            position_size_pct: Position size as % of capital (0.1 = 10%)

        Returns:
            Dictionary with backtest results
        """
        print("\n" + "="*80)
        print("BACKTESTING ML TRADING SYSTEM")
        print("="*80)

        # Handle date column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['date'] = pd.to_datetime(df['time'])

        # Set date range
        if end_date is None:
            end_date = df['date'].max()
        if start_date is None:
            start_date = end_date - timedelta(days=30)

        # Filter to backtest period
        backtest_df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()

        # Drop rows with NaN in critical columns (needed for labels)
        critical_cols = ['close', 'high', 'low']
        backtest_df = backtest_df.dropna(subset=critical_cols)
        backtest_df = backtest_df.reset_index(drop=True)

        print(f"\nBacktest Period: {start_date.date()} to {end_date.date()}")
        print(f"Trading Days: {len(backtest_df)}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Trade Quality Threshold: {trade_quality_threshold}/100")
        print(f"Profit Target: {profit_target_pct}%")
        print(f"Stop Loss: {stop_loss_pct}%")
        print(f"Position Size: {position_size_pct*100}% of capital")

        # Prepare features
        exclude = [
            'date', 'time', 'open', 'high', 'low', 'close', 'volume',
            'future_high', 'future_low', 'future_return',
            'profit_target_hit', 'stop_loss_hit', 'trade_quality',
            'expected_move_up', 'expected_move_down'
        ]
        feature_cols = [col for col in backtest_df.columns if col not in exclude]

        # Initialize tracking
        capital = self.initial_capital
        self.trades = []
        self.equity_curve = [{'date': start_date, 'equity': capital}]

        # Simulate trading day by day
        for i in range(len(backtest_df)):
            row = backtest_df.iloc[i]
            current_date = row['date']
            current_price = row['close']

            # Get features for this day
            X = backtest_df.iloc[[i]][feature_cols]
            X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())

            # Generate predictions
            predictions = self._predict(X)

            if predictions is None:
                continue

            trade_quality = predictions.get('trade_quality_score', [0])[0]
            should_trade = trade_quality >= trade_quality_threshold

            # Only trade if quality threshold met
            if should_trade:
                profit_prob = predictions.get('profit_probability', [0])[0]
                predicted_high = predictions.get('predicted_high', [current_price])[0]
                predicted_low = predictions.get('predicted_low', [current_price])[0]

                # Calculate targets
                entry_price = current_price
                profit_target = entry_price * (1 + profit_target_pct/100)
                stop_loss = entry_price * (1 - stop_loss_pct/100)

                # Position sizing
                position_value = capital * position_size_pct
                shares = int(position_value / entry_price)

                if shares > 0:
                    # Simulate trade outcome using NEXT day's data (forward-looking)
                    if i + 1 < len(backtest_df):
                        next_row = backtest_df.iloc[i + 1]
                        next_high = next_row['high']
                        next_low = next_row['low']
                        exit_price = next_row['close']

                        # Check if profit target or stop loss hit
                        profit_hit = next_high >= profit_target
                        stop_hit = next_low <= stop_loss

                        if profit_hit:
                            exit_price = profit_target
                            result = 'WIN'
                        elif stop_hit:
                            exit_price = stop_loss
                            result = 'LOSS'
                        else:
                            # Exit at close if neither hit
                            result = 'NEUTRAL'

                        # Calculate P&L
                        pnl = (exit_price - entry_price) * shares
                        pnl_pct = ((exit_price - entry_price) / entry_price) * 100
                        capital += pnl

                        # Record trade
                        trade = {
                            'entry_date': current_date,
                            'exit_date': next_row['date'],
                            'entry_price': entry_price,
                            'exit_price': exit_price,
                            'shares': shares,
                            'pnl': pnl,
                            'pnl_pct': pnl_pct,
                            'result': result,
                            'trade_quality': trade_quality,
                            'profit_probability': profit_prob,
                            'predicted_high': predicted_high,
                            'predicted_low': predicted_low
                        }
                        self.trades.append(trade)

            # Update equity curve
            self.equity_curve.append({'date': current_date, 'equity': capital})

        # Calculate performance metrics
        results = self._calculate_metrics(start_date, end_date)

        return results

    def _predict(self, X):
        """Generate predictions from models"""
        try:
            predictions = {}

            # Trade Quality
            if 'trade_quality' in self.models:
                quality_proba = self.models['trade_quality'].predict_proba(X)[:, 1]
                predictions['trade_quality_score'] = quality_proba * 100

            # Profit Probability
            if 'profit_target' in self.models:
                profit_proba = self.models['profit_target'].predict_proba(X)[:, 1]
                predictions['profit_probability'] = profit_proba * 100

            # Price Levels
            if 'future_high' in self.models:
                predictions['predicted_high'] = self.models['future_high'].predict(X)

            if 'future_low' in self.models:
                predictions['predicted_low'] = self.models['future_low'].predict(X)

            return predictions
        except Exception as e:
            return None

    def _calculate_metrics(self, start_date, end_date):
        """Calculate backtest performance metrics"""
        if not self.trades:
            return {
                'error': 'No trades taken during backtest period',
                'total_trades': 0
            }

        trades_df = pd.DataFrame(self.trades)
        final_capital = self.equity_curve[-1]['equity']

        # Basic metrics
        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100
        num_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / num_trades * 100) if num_trades > 0 else 0

        # P&L metrics
        total_pnl = trades_df['pnl'].sum()
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() /
                           trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if losing_trades > 0 else 0

        # Risk metrics
        returns = trades_df['pnl_pct'].values
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        max_drawdown = self._calculate_max_drawdown()

        results = {
            'period': {
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d'),
                'days': (end_date - start_date).days
            },
            'capital': {
                'initial': self.initial_capital,
                'final': final_capital,
                'total_return_pct': total_return,
                'total_pnl': total_pnl
            },
            'trades': {
                'total': num_trades,
                'wins': winning_trades,
                'losses': losing_trades,
                'win_rate_pct': win_rate
            },
            'pnl': {
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_pnl_per_trade': total_pnl / num_trades if num_trades > 0 else 0
            },
            'risk': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown_pct': max_drawdown,
                'avg_trade_pct': returns.mean(),
                'std_dev_pct': returns.std()
            }
        }

        return results

    def _calculate_max_drawdown(self):
        """Calculate maximum drawdown from equity curve"""
        equity_values = [e['equity'] for e in self.equity_curve]
        peak = equity_values[0]
        max_dd = 0

        for value in equity_values:
            if value > peak:
                peak = value
            dd = ((peak - value) / peak) * 100
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def print_results(self, results):
        """Print formatted backtest results"""
        if 'error' in results:
            print(f"\n[ERROR] {results['error']}")
            return

        print("\n" + "="*80)
        print("BACKTEST RESULTS")
        print("="*80)

        print(f"\n[PERIOD]")
        print(f"  Start Date:        {results['period']['start']}")
        print(f"  End Date:          {results['period']['end']}")
        print(f"  Trading Days:      {results['period']['days']}")

        print(f"\n[PERFORMANCE]")
        print(f"  Initial Capital:   ${results['capital']['initial']:,.2f}")
        print(f"  Final Capital:     ${results['capital']['final']:,.2f}")
        print(f"  Total Return:      {results['capital']['total_return_pct']:+.2f}%")
        print(f"  Total P&L:         ${results['capital']['total_pnl']:+,.2f}")

        print(f"\n[TRADES]")
        print(f"  Total Trades:      {results['trades']['total']}")
        print(f"  Winning Trades:    {results['trades']['wins']}")
        print(f"  Losing Trades:     {results['trades']['losses']}")
        print(f"  Win Rate:          {results['trades']['win_rate_pct']:.1f}%")

        print(f"\n[P&L ANALYSIS]")
        print(f"  Avg Win:           ${results['pnl']['avg_win']:+,.2f}")
        print(f"  Avg Loss:          ${results['pnl']['avg_loss']:+,.2f}")
        print(f"  Profit Factor:     {results['pnl']['profit_factor']:.2f}")
        print(f"  Avg P&L/Trade:     ${results['pnl']['avg_pnl_per_trade']:+,.2f}")

        print(f"\n[RISK METRICS]")
        print(f"  Sharpe Ratio:      {results['risk']['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown:      {results['risk']['max_drawdown_pct']:.2f}%")
        print(f"  Avg Trade Return:  {results['risk']['avg_trade_pct']:+.2f}%")
        print(f"  Std Deviation:     {results['risk']['std_dev_pct']:.2f}%")

        # Trade quality
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            print(f"\n[TRADE QUALITY]")
            print(f"  Avg Quality Score: {trades_df['trade_quality'].mean():.1f}/100")
            print(f"  Avg Win Prob:      {trades_df['profit_probability'].mean():.1f}%")

    def save_results(self, results, filename='backtest_results.json'):
        """Save backtest results to file"""
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {filename}")

        # Save trades
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df.to_csv('backtest_trades.csv', index=False)
            print(f"[OK] Trades saved to: backtest_trades.csv")

        # Save equity curve
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.to_csv('backtest_equity_curve.csv', index=False)
        print(f"[OK] Equity curve saved to: backtest_equity_curve.csv")


def main():
    """Run backtest on last 30 days"""
    import pandas as pd
    import glob
    import joblib

    print("Loading ML dataset and models...")

    try:
        # Load data
        df = pd.read_csv('spy_ml_dataset.csv')
        print(f"[OK] Loaded {len(df)} samples")

        # Load trained models
        trainer = TradingMLModel()
        model_files = glob.glob('spy_trading_model_*.pkl')

        if not model_files:
            print("\n[ERROR] No trained models found!")
            print("Run train_models.py first to train models.")
            return

        # Load models manually
        for filepath in model_files:
            # Extract model name from filename
            parts = filepath.replace('.pkl', '').split('_')
            # Handle multi-word model names like 'future_high'
            if 'future' in filepath:
                if 'high' in filepath:
                    model_name = 'future_high'
                elif 'low' in filepath:
                    model_name = 'future_low'
            elif 'trade' in filepath and 'quality' in filepath:
                model_name = 'trade_quality'
            elif 'profit' in filepath:
                model_name = 'profit_target'
            else:
                continue

            trainer.models[model_name] = joblib.load(filepath)
            print(f"[OK] Loaded: {model_name}")

        if not trainer.models:
            print("\n[ERROR] Failed to load any models!")
            return

        # Run backtest on valid data range
        # Valid data ends on 2025-10-28, so backtest Sept-Oct 2025
        end_valid = pd.to_datetime('2025-10-28')
        start_valid = end_valid - pd.Timedelta(days=30)

        backtester = TradingBacktest(trainer.models, initial_capital=10000)
        results = backtester.run_backtest(
            df,
            start_date=start_valid,
            end_date=end_valid,
            trade_quality_threshold=0,  # Take all trades to demonstrate
            profit_target_pct=0.5,
            stop_loss_pct=0.3,
            position_size_pct=0.1
        )

        # Display results
        backtester.print_results(results)

        # Save results
        backtester.save_results(results)

        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)

    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e}")
        print("Run the full pipeline first: feature_engineering.py -> train_models.py")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
