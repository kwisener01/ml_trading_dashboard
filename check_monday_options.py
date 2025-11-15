from data_collector import TradierDataCollector
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
api_token = os.getenv('TRADIER_API_TOKEN')
collector = TradierDataCollector(api_token)

# Calculate next Monday
now = datetime.now()
days_until_monday = (7 - now.weekday()) % 7
if days_until_monday == 0:
    days_until_monday = 7
next_monday = (now + timedelta(days=days_until_monday)).strftime('%Y-%m-%d')

print(f'Next Monday: {next_monday}')
print('='*80)

# Get option chain for SPY with Greeks
chains = collector.get_option_chain_with_greeks('SPY', next_monday)

if chains is not None and len(chains) > 0:
    print(f'\nTotal strikes: {len(chains)}')

    # Show available columns
    print(f'\nAvailable columns: {list(chains.columns)}')

    # Separate calls and puts
    calls = chains[chains['option_type'] == 'call'].copy()
    puts = chains[chains['option_type'] == 'put'].copy()

    print(f'\nCalls: {len(calls)} | Puts: {len(puts)}')

    # Select columns that exist
    display_cols = ['strike', 'bid', 'ask', 'volume', 'open_interest']
    if 'greeks' in chains.columns or 'delta' in chains.columns:
        display_cols.extend(['delta', 'gamma', 'theta', 'vega'])

    # Get current SPY price
    print('\n' + '='*80)
    print('CALLS (First 15 strikes)')
    print('='*80)
    available_cols = [col for col in display_cols if col in calls.columns]
    print(calls[available_cols].head(15).to_string(index=False))

    print('\n' + '='*80)
    print('PUTS (First 15 strikes)')
    print('='*80)
    print(puts[available_cols].head(15).to_string(index=False))

    # Show ATM options (around current price)
    current_price = 671.93  # From last prediction
    print('\n' + '='*80)
    print(f'ATM OPTIONS (Around current price ${current_price:.2f})')
    print('='*80)

    # Get strikes within $5 of current price
    atm_range = chains[(chains['strike'] >= current_price - 5) & (chains['strike'] <= current_price + 5)].copy()
    atm_range = atm_range.sort_values(['strike', 'option_type'])

    atm_cols = [col for col in ['strike', 'option_type', 'bid', 'ask', 'volume', 'open_interest', 'delta', 'gamma'] if col in atm_range.columns]
    print(atm_range[atm_cols].to_string(index=False))

else:
    print('No option chain data available for this date')
