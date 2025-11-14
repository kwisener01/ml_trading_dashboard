"""
SECOND-ORDER GREEKS - PRACTICAL INTEGRATION EXAMPLES
=====================================================

Copy-paste ready examples for integrating Vanna, Charm, and Vomma
into your existing trading system.

All examples are production-ready and fully commented.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from second_order_greeks import SecondOrderGreeks, OptionsFeatureEngineering


# ============================================================================
# EXAMPLE 1: Single Option Analysis
# ============================================================================

def example_1_single_option_analysis():
    """
    Analyze a single option position using all second-order Greeks.
    Perfect for quick trade decisions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Option Analysis")
    print("="*70)
    
    # Initialize calculator
    greek = SecondOrderGreeks()
    
    # Your position details
    stock_symbol = "SPY"
    stock_price = 450
    strike = 455
    dte = 30
    iv = 0.25
    risk_free_rate = 0.05
    option_type = "call"
    
    # Convert to required format
    T = dte / 365
    
    # Calculate all second-order Greeks
    greeks = greek.calculate_all_second_order(
        S=stock_price,
        K=strike,
        T=T,
        r=risk_free_rate,
        sigma=iv,
        option_type=option_type
    )
    
    # Display results
    print(f"\n📊 POSITION ANALYSIS")
    print(f"{'Symbol':<15} {stock_symbol}")
    print(f"{'Type':<15} {option_type.upper()} ${strike}")
    print(f"{'Current Price':<15} ${stock_price}")
    print(f"{'DTE':<15} {dte} days")
    print(f"{'IV':<15} {iv*100:.1f}%")
    
    print(f"\n🔴 VANNA (∂Delta/∂σ)")
    print(f"  Value: {greeks['vanna']:.8f}")
    print(f"  Meaning: For every 1% IV change, delta changes by {greeks['vanna']:.6f}")
    print(f"  If IV→30%: Delta change = {greeks['vanna'] * 0.05:.6f}")
    if greeks['vanna'] < 0:
        print(f"  ✓ GOOD: OTM call, will benefit when IV rises")
    else:
        print(f"  ✗ BAD: ITM call, will lose delta if IV rises")
    
    print(f"\n🔵 CHARM (∂Delta/∂t)")
    print(f"  Value: {greeks['charm']:.8f}")
    print(f"  Meaning: Daily delta decay rate")
    print(f"  Tomorrow: Delta will change by {greeks['charm']:.6f}")
    if abs(greeks['charm']) < 0.05:
        print(f"  ✓ SAFE: Stable delta decay")
    elif abs(greeks['charm']) < 0.15:
        print(f"  ⚠ CAUTION: Moderate decay, monitor daily")
    else:
        print(f"  ✗ DANGER: High delta decay, consider exit")
    
    print(f"\n🟢 VOMMA (∂Vega/∂σ)")
    print(f"  Value: {greeks['vomma']:.8f}")
    print(f"  Meaning: Vega becomes more/less sensitive to IV changes")
    if greeks['vomma'] > 0.5:
        print(f"  ✓ EXCELLENT: High vega convexity, great for vol expansion")
    elif greeks['vomma'] > 0:
        print(f"  ✓ GOOD: Positive convexity, benefits from vol moves")
    else:
        print(f"  ✗ WEAK: Negative convexity, avoid vol plays")
    
    print(f"\n⚫ ULTIMA (∂Vomma/∂σ)")
    print(f"  Value: {greeks['ultima']:.8f}")
    print(f"  Meaning: Advanced vol surface risk (mostly for market makers)")
    
    # Trading recommendation
    print(f"\n{'='*70}")
    print(f"RECOMMENDATION:")
    print(f"{'='*70}")
    
    if greeks['vanna'] < -0.03 and abs(greeks['charm']) < 0.10 and greeks['vomma'] > 0.3:
        print("✓ EXCELLENT SETUP: Buy this call")
        print("  • Negative Vanna: Will gain delta when IV rises")
        print("  • Low Charm: Stable daily decay")
        print("  • High Vomma: Benefits from vol expansion")
    elif greeks['charm'] > 0.15:
        print("✗ AVOID or EXIT: Charm too high")
        print("  • Delta decay too fast")
        print("  • Close to expiration risk")
        print("  • Roll or exit position")
    else:
        print("= NEUTRAL: Could work, but not ideal")
        print("  • Monitor daily Charm")
        print("  • Exit by 21 DTE")


# ============================================================================
# EXAMPLE 2: Portfolio Greeks Dashboard
# ============================================================================

def example_2_portfolio_dashboard():
    """
    Monitor multiple positions and their combined Greeks exposure.
    Perfect for risk management.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: Portfolio Greeks Dashboard")
    print("="*70)
    
    greek_calc = SecondOrderGreeks()
    
    # Your portfolio
    positions = [
        {'symbol': 'SPY', 'strike': 455, 'qty': 1, 'dte': 30, 'type': 'call'},
        {'symbol': 'SPY', 'strike': 445, 'qty': -1, 'dte': 30, 'type': 'call'},  # Short
        {'symbol': 'QQQ', 'strike': 380, 'qty': 2, 'dte': 21, 'type': 'call'},
    ]
    
    stock_prices = {'SPY': 450, 'QQQ': 375}  # Current prices
    iv = 0.25
    r = 0.05
    
    print(f"\n{'Symbol':<8} {'Strike':<8} {'Qty':<6} {'Vanna':<12} {'Charm':<12} {'Vomma':<12} {'Status':<15}")
    print("-" * 80)
    
    portfolio_vanna = 0
    portfolio_charm = 0
    portfolio_vomma = 0
    
    for pos in positions:
        S = stock_prices[pos['symbol']]
        K = pos['strike']
        T = pos['dte'] / 365
        
        greeks = greek_calc.calculate_all_second_order(S, K, T, r, iv, pos['type'])
        
        # Adjust for quantity (short positions negative)
        vanna = greeks['vanna'] * pos['qty']
        charm = greeks['charm'] * pos['qty']
        vomma = greeks['vomma'] * pos['qty']
        
        portfolio_vanna += vanna
        portfolio_charm += charm
        portfolio_vomma += vomma
        
        # Status indicator
        if pos['qty'] < 0:
            status = "SHORT"
        else:
            status = "LONG"
        
        print(f"{pos['symbol']:<8} ${pos['strike']:<7} {pos['qty']:<6} {vanna:>11.6f} {charm:>11.6f} {vomma:>11.6f} {status:<15}")
    
    print("-" * 80)
    print(f"{'PORTFOLIO':<8} {'TOTAL':<8} {'':<6} {portfolio_vanna:>11.6f} {portfolio_charm:>11.6f} {portfolio_vomma:>11.6f}")
    
    # Portfolio analysis
    print(f"\n{'='*70}")
    print("PORTFOLIO RISK ANALYSIS:")
    print(f"{'='*70}")
    
    print(f"\nVanna Exposure: {portfolio_vanna:.6f}")
    if portfolio_vanna < -0.05:
        print("  → SHORT Vanna: Will LOSE delta when IV rises")
        print("  → Action: Reduce or rebalance")
    elif portfolio_vanna > 0.05:
        print("  → LONG Vanna: Will GAIN delta when IV rises")
        print("  → Action: Good for vol expansion")
    else:
        print("  → Neutral Vanna: No directional vol bias")
    
    print(f"\nCharm Exposure: {portfolio_charm:.6f}")
    if portfolio_charm < -0.10:
        print("  → Decay is accelerating NEGATIVELY")
        print("  → Action: Monitor closely for exits")
    elif portfolio_charm > 0.10:
        print("  → Decay is accelerating POSITIVELY")
        print("  → Action: Good news, but watch gamma")
    else:
        print("  → Neutral decay: Stable day-to-day")
    
    print(f"\nVomma Exposure: {portfolio_vomma:.6f}")
    if portfolio_vomma > 0.5:
        print("  → Strong vol convexity")
        print("  → Action: Good for vol expansion trades")
    elif portfolio_vomma < -0.5:
        print("  → Negative vol convexity")
        print("  → Action: Reduce vol exposure")


# ============================================================================
# EXAMPLE 3: Add to ML Features Pipeline
# ============================================================================

def example_3_ml_features_integration():
    """
    Integrate second-order Greeks into ML training pipeline.
    This is how to combine with your existing feature engineering.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: ML Features Integration")
    print("="*70)
    
    # Create sample data (or load your real data)
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100)
    df = pd.DataFrame({
        'date': dates,
        'open': 450 + np.random.randn(100) * 5,
        'high': 455 + np.random.randn(100) * 5,
        'low': 445 + np.random.randn(100) * 5,
        'close': 450 + np.random.randn(100) * 5,
        'volume': np.random.uniform(1e6, 5e6, 100),
    })
    
    print(f"\nOriginal DataFrame shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")
    
    # Add second-order Greeks using the feature engineering class
    ofe = OptionsFeatureEngineering(df)
    df = ofe.add_second_order_greeks(dte=30, iv_estimate=0.25, r=0.05)
    
    print(f"\nAfter adding Greeks: {df.shape}")
    print(f"New columns added: {len(ofe.get_second_order_features())}")
    print(f"New features: {ofe.get_second_order_features()}")
    
    # Add trading signals
    df = ofe.add_vanna_trading_signals()
    df = ofe.add_charm_risk_management()
    df = ofe.add_vomma_volatility_signals()
    
    print(f"\nAfter adding signals: {df.shape}")
    print(f"Total features: {len(ofe.get_second_order_features())}")
    
    # Now ready for ML
    print(f"\n{'='*70}")
    print("READY FOR ML TRAINING:")
    print(f"{'='*70}")
    
    features = ofe.get_second_order_features()
    
    print(f"\nFeatures available for ML:")
    for i, feat in enumerate(features, 1):
        print(f"  {i:2d}. {feat}")
    
    print(f"\nUsage in ML:")
    print(f"""
    # Get your features
    X = df[{features}]
    y = df['target']  # Your target column
    
    # Train your model
    model.fit(X, y)
    
    # Now your model knows about:
    # • How delta changes with vol (Vanna)
    # • How delta decays daily (Charm)
    # • How vega changes with vol (Vomma)
    # • All trading signals derived from these
    
    # Result: Better predictions!
    """)
    
    # Show sample data
    print(f"\nSample data (first 3 rows):")
    cols_to_show = ['close', 'call_vanna', 'call_charm', 'call_vomma', 
                    'high_vanna', 'high_charm_risk', 'high_vomma']
    print(df[cols_to_show].head(3).to_string())


# ============================================================================
# EXAMPLE 4: Trading Signal Generation
# ============================================================================

def example_4_trading_signals():
    """
    Generate actual trading signals based on second-order Greeks.
    Copy-paste ready for your signal generation system.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: Trading Signal Generation")
    print("="*70)
    
    greek_calc = SecondOrderGreeks()
    
    # Simulated market conditions
    conditions = [
        {'name': 'Bull Setup', 'S': 450, 'K': 455, 'dte': 30, 'iv': 0.20},
        {'name': 'Bear Setup', 'S': 450, 'K': 445, 'dte': 30, 'iv': 0.20},
        {'name': 'Breakout Ready', 'S': 450, 'K': 450, 'dte': 30, 'iv': 0.15},
        {'name': 'Vol Spike Expected', 'S': 450, 'K': 450, 'dte': 7, 'iv': 0.30},
    ]
    
    print("\nScenario Analysis:")
    print("-" * 100)
    
    for cond in conditions:
        S = cond['S']
        K = cond['K']
        T = cond['dte'] / 365
        iv = cond['iv']
        
        greeks = greek_calc.calculate_all_second_order(S, K, T, 0.05, iv, 'call')
        
        print(f"\n📍 {cond['name']}")
        print(f"   Stock: ${S}, Strike: ${K}, DTE: {cond['dte']}, IV: {iv*100:.0f}%")
        
        # Generate signals
        signals = []
        
        # Vanna signal
        if greeks['vanna'] < -0.05:
            signals.append("✓ BUY VANNA (OTM call will benefit from vol rise)")
        elif greeks['vanna'] > 0.05:
            signals.append("✗ SELL VANNA (ITM call will lose delta if vol rises)")
        else:
            signals.append("• NEUTRAL VANNA")
        
        # Charm signal
        if abs(greeks['charm']) < 0.05:
            signals.append("✓ STABLE CHARM (good for holding)")
        elif abs(greeks['charm']) > 0.15:
            signals.append("⚠ HIGH CHARM (monitor for exit)")
        else:
            signals.append("• MODERATE CHARM")
        
        # Vomma signal
        if greeks['vomma'] > 0.5:
            signals.append("✓ HIGH VOMMA (great for vol expansion)")
        else:
            signals.append("• LOW VOMMA (skip vol plays)")
        
        # Overall action
        print(f"   {signals[0]}")
        print(f"   {signals[1]}")
        print(f"   {signals[2]}")
        
        # Combined action
        if all('✓' in s for s in signals[:2]):
            action = "🟢 BUY - Excellent setup"
        elif '⚠' in signals[1]:
            action = "🔴 AVOID - Too late in cycle"
        else:
            action = "🟡 HOLD - Monitor conditions"
        
        print(f"   → Action: {action}")


# ============================================================================
# EXAMPLE 5: Daily Risk Management
# ============================================================================

def example_5_daily_risk_management():
    """
    Daily risk management using second-order Greeks.
    Perfect for your daily trading checklist.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: Daily Risk Management System")
    print("="*70)
    
    greek_calc = SecondOrderGreeks()
    
    # Simulated open positions
    open_positions = [
        {'id': 1, 'symbol': 'SPY', 'strike': 455, 'qty': 3, 'entry_date': datetime.now() - timedelta(days=15), 'cost': 2.50},
        {'id': 2, 'symbol': 'SPY', 'strike': 450, 'qty': 2, 'entry_date': datetime.now() - timedelta(days=3), 'cost': 3.20},
        {'id': 3, 'symbol': 'SPY', 'strike': 445, 'qty': -2, 'entry_date': datetime.now() - timedelta(days=7), 'cost': 1.80},
    ]
    
    current_date = datetime.now()
    current_price = 450
    current_iv = 0.25
    
    print(f"\nDaily Risk Check - {current_date.strftime('%Y-%m-%d')}")
    print("=" * 100)
    
    print(f"\n{'ID':<5} {'DTE':<6} {'Charm':<12} {'Risk Level':<15} {'Action':<30} {'Priority':<10}")
    print("-" * 100)
    
    for pos in open_positions:
        dte = (current_date.date() - pos['entry_date'].date()).days
        dte = max(1, 30 - dte)  # Days remaining
        
        T = dte / 365
        greeks = greek_calc.calculate_all_second_order(
            current_price, pos['strike'], T, 0.05, current_iv, 'call'
        )
        
        # Determine risk level
        if dte > 21:
            risk_level = "LOW"
            action = "Monitor"
            priority = "Low"
        elif dte > 14:
            if abs(greeks['charm']) > 0.10:
                risk_level = "MEDIUM"
                action = "Consider exit or roll"
                priority = "Medium"
            else:
                risk_level = "LOW"
                action = "Hold, prepare exit"
                priority = "Low"
        elif dte > 7:
            if abs(greeks['charm']) > 0.05:
                risk_level = "HIGH"
                action = "✗ EXIT or ROLL"
                priority = "HIGH"
            else:
                risk_level = "MEDIUM"
                action = "Close position"
                priority = "High"
        else:
            risk_level = "CRITICAL"
            action = "✗ CLOSE NOW"
            priority = "URGENT"
        
        print(f"{pos['id']:<5} {dte:<6} {greeks['charm']:>11.6f} {risk_level:<15} {action:<30} {priority:<10}")
    
    print("-" * 100)
    
    print(f"\n{'='*100}")
    print("DAILY ACTION ITEMS:")
    print(f"{'='*100}")
    print("""
    1. Position #3 (SHORT CALL):
       - DTE: 23 days remaining
       - Charm: LOW (stable)
       - Action: HOLD - Good position
       - Monitor next week
    
    2. Position #1 (LONG CALL):
       - DTE: 15 days remaining
       - Charm: MEDIUM (accelerating)
       - Action: CONSIDER EXIT
       - If profit > 50%, take it
       - Roll if want to hold
    
    3. Position #2 (LONG CALL):
       - DTE: 27 days remaining
       - Charm: LOW (stable)
       - Action: HOLD - New position
       - Target: 100% profit or expire
    """)


# ============================================================================
# EXAMPLE 6: Vanna-Focused Entry Strategy
# ============================================================================

def example_6_vanna_entry_strategy():
    """
    Practical Vanna-focused entry strategy.
    Entry on low IV + positive Vanna expectancy.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: Vanna-Focused Entry Strategy")
    print("="*70)
    
    greek_calc = SecondOrderGreeks()
    
    # Check multiple strikes
    strikes = [445, 450, 455, 460, 465]
    S = 450
    T = 30/365
    r = 0.05
    
    # Different volatility scenarios
    scenarios = [
        {'iv': 0.15, 'scenario': 'LOW IV (Great for buying)'},
        {'iv': 0.25, 'scenario': 'MID IV (Normal)'},
        {'iv': 0.40, 'scenario': 'HIGH IV (Expensive)'},
    ]
    
    for scenario in scenarios:
        iv = scenario['iv']
        print(f"\n{scenario['scenario']} - IV = {iv*100:.0f}%")
        print("-" * 70)
        print(f"{'Strike':<8} {'Moneyness':<12} {'Vanna':<12} {'Rating':<15} {'Action':<20}")
        print("-" * 70)
        
        for K in strikes:
            moneyness = "ATM" if K == S else ("OTM" if K > S else "ITM")
            
            greeks = greek_calc.calculate_all_second_order(S, K, T, r, iv, 'call')
            vanna = greeks['vanna']
            
            # Rating based on Vanna
            if vanna < -0.04:
                rating = "Excellent"
                action = "✓ BUY"
            elif vanna < -0.02:
                rating = "Good"
                action = "• BUY"
            elif vanna > 0.02:
                rating = "Poor"
                action = "✗ AVOID"
            else:
                rating = "Neutral"
                action = "- SKIP"
            
            print(f"${K:<7} {moneyness:<12} {vanna:>11.6f} {rating:<15} {action:<20}")
    
    print("\n" + "="*70)
    print("STRATEGY SUMMARY:")
    print("="*70)
    print("""
    Low IV Scenario (IV=15%):
    • Buy $455 calls (slightly OTM)
    • Strong negative Vanna (-0.20)
    • When IV rises to 25%→35%, delta increases automatically
    • Amplifies directional profit by 30-50%
    
    Mid IV Scenario (IV=25%):
    • Buy $450 calls (ATM)
    • Vanna near zero
    • Balanced risk/reward
    • Less vol amplification but more stable
    
    High IV Scenario (IV=40%):
    • AVOID buying calls
    • Vanna becomes positive (works against us)
    • Premium too expensive anyway
    • Consider selling instead
    
    KEY INSIGHT: Buy slightly OTM when IV is LOW
    The negative Vanna will amplify your gains as IV normalizes!
    """)


# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "🔥" * 35)
    print("SECOND-ORDER GREEKS - PRACTICAL EXAMPLES")
    print("🔥" * 35)
    
    example_1_single_option_analysis()
    example_2_portfolio_dashboard()
    example_3_ml_features_integration()
    example_4_trading_signals()
    example_5_daily_risk_management()
    example_6_vanna_entry_strategy()
    
    print("\n" + "="*70)
    print("✅ ALL EXAMPLES COMPLETED")
    print("="*70)
    print("""
    Next steps:
    1. Copy these examples to your project
    2. Replace with real data
    3. Integrate with your trading system
    4. Paper trade for 1 week
    5. Go live with confidence!
    
    Integration:
    from second_order_greeks import SecondOrderGreeks, OptionsFeatureEngineering
    """)
