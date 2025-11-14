import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from predictor import TradingPredictor
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    .trade-signal {
        font-size: 24px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .signal-good {
        background-color: #d4edda;
        color: #155724;
    }
    .signal-bad {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 ML Trading Dashboard")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    # Get token from environment or allow override
    env_token = os.getenv('TRADIER_API_TOKEN', '')

    if env_token:
        st.success("✓ API token loaded from .env")
        api_token = env_token
        # Show masked token
        masked_token = env_token[:4] + "..." + env_token[-4:] if len(env_token) > 8 else "****"
        st.caption(f"Token: {masked_token}")
    else:
        st.warning("⚠️ No .env file found")
        api_token = st.text_input("Tradier API Token", type="password",
                                   help="Create a .env file with TRADIER_API_TOKEN=your_token")

    # Trading Mode Selection
    st.markdown("### 📊 Trading Mode")
    trading_mode = st.radio(
        "Select Mode",
        ["Daily Trading", "Day Trading (Intraday)"],
        help="Daily = Swing trades, Day Trading = 5min intraday"
    )

    symbol = st.selectbox("Select Symbol", ['SPY', 'QQQ', 'IWM', 'DIA'])

    # Day trading specific settings
    if trading_mode == "Day Trading (Intraday)":
        st.markdown("### ⏰ Intraday Settings")
        interval = st.selectbox("Chart Interval", ['5min', '15min', '1min'], index=0)

        # Show market hours status
        now = datetime.now()
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        optimal_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        optimal_end = now.replace(hour=15, minute=0, second=0, microsecond=0)

        if now < market_open:
            st.warning(f"⏸️ Pre-market ({(market_open - now).seconds // 60} min to open)")
        elif now > market_close:
            st.info("🔔 Market closed")
        elif now < optimal_start:
            st.warning("⚠️ Avoid zone (first 30 min)")
        elif now > optimal_end:
            st.warning("⚠️ Closing time (last hour)")
        else:
            st.success("✅ Optimal trading hours")
    else:
        interval = 'daily'

    trade_quality_threshold = st.slider("Trade Quality Threshold", 0, 100, 60)

    if st.button("🔄 Generate Prediction", type="primary"):
        st.session_state['generate_prediction'] = True
        st.session_state['trading_mode'] = trading_mode
        st.session_state['interval'] = interval
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    
    # Check if models exist
    import glob
    model_files = glob.glob('*_trading_model_*.pkl')
    
    if model_files:
        st.success(f"✅ {len(model_files)} models loaded")
        
        # Load metrics if available
        metric_files = glob.glob('*_metrics_*.json')
        if metric_files:
            with open(metric_files[0], 'r') as f:
                metrics = json.load(f)
            
            st.metric("Trade Quality Accuracy", 
                     f"{metrics.get('trade_quality', {}).get('accuracy', 0)*100:.1f}%")
            st.metric("Profit Target Accuracy", 
                     f"{metrics.get('profit_target', {}).get('accuracy', 0)*100:.1f}%")
    else:
        st.warning("⚠️ No models found. Train models first!")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    if st.session_state.get('trading_mode') == "Day Trading (Intraday)":
        st.info("""
        **Day Trading Mode**
        - Live 5min/15min/1min charts
        - Intraday support/resistance
        - Vanna levels (0DTE options)
        - Market hours monitoring
        - Optimal trading windows
        - Same-day profit targets
        """)
    else:
        st.info("""
        This dashboard uses ML models to predict:
        - Support/Resistance levels
        - Price targets
        - **When NOT to trade** (most important)
        """)

# Main content
if not api_token:
    st.warning("⚠️ Please enter your Tradier API token in the sidebar")
    st.stop()

# Initialize predictor
try:
    if 'predictor' not in st.session_state:
        with st.spinner("Loading models..."):
            st.session_state['predictor'] = TradingPredictor(api_token)
    
    predictor = st.session_state['predictor']
except Exception as e:
    st.error(f"Error loading predictor: {e}")
    st.info("Make sure you've trained the models first by running: `python train_models.py`")
    st.stop()

# Generate prediction
if st.session_state.get('generate_prediction', False):
    with st.spinner(f"Analyzing {symbol}..."):
        try:
            predictions = predictor.predict(symbol)
            st.session_state['predictions'] = predictions
            st.session_state['generate_prediction'] = False
        except Exception as e:
            st.error(f"Error generating prediction: {e}")
            st.session_state['generate_prediction'] = False

# Display predictions
if 'predictions' in st.session_state:
    pred = st.session_state['predictions']
    
    # Header row
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"### {pred['symbol']} @ ${pred['current_price']:.2f}")
        st.caption(f"Last updated: {pred['timestamp']}")
    
    with col2:
        quality = pred.get('trade_quality_score', 0)
        # Handle None values
        if quality is None:
            st.metric("Trade Quality", "N/A", delta=None)
        else:
            delta_color = "normal" if quality >= 50 else "inverse"
            st.metric("Trade Quality", f"{quality:.1f}/100", delta=None)
    
    with col3:
        should_trade = pred.get('should_trade', False)
        if should_trade:
            st.markdown('<div class="trade-signal signal-good">✅ TRADEABLE</div>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<div class="trade-signal signal-bad">🚫 AVOID</div>', 
                       unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        profit_prob = pred.get('profit_probability')
        if profit_prob is not None:
            st.metric("Win Probability", f"{profit_prob:.1f}%")
    
    with col2:
        upside = pred.get('upside_target')
        if upside is not None:
            st.metric("Upside Target", f"+{upside:.2f}%", 
                     delta=f"${pred['predicted_high']:.2f}")
    
    with col3:
        downside = pred.get('downside_risk')
        if downside is not None:
            st.metric("Downside Risk", f"-{downside:.2f}%", 
                     delta=f"${pred['predicted_low']:.2f}", delta_color="inverse")
    
    with col4:
        if pred.get('upside_target') and pred.get('downside_risk'):
            rr_ratio = pred['upside_target'] / pred['downside_risk']
            st.metric("Risk/Reward", f"{rr_ratio:.2f}:1")
    
    st.markdown("---")
    
    # Price chart with targets
    st.subheader("📊 Price Targets Visualization")
    
    current = pred['current_price']
    pred_high = pred.get('predicted_high', current * 1.01)
    pred_low = pred.get('predicted_low', current * 0.99)
    
    fig = go.Figure()
    
    # Current price line
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[current, current],
        mode='lines',
        name='Current Price',
        line=dict(color='blue', width=3)
    ))
    
    # Predicted high
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[pred_high, pred_high],
        mode='lines',
        name='Upside Target',
        line=dict(color='green', width=2, dash='dash')
    ))
    
    # Predicted low
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[pred_low, pred_low],
        mode='lines',
        name='Downside Stop',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    # Fill area between high and low
    fig.add_trace(go.Scatter(
        x=[0, 1, 1, 0],
        y=[pred_low, pred_low, pred_high, pred_high],
        fill='toself',
        fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(width=0),
        name='Expected Range',
        showlegend=True
    ))
    
    fig.update_layout(
        title=f"{symbol} Price Targets",
        yaxis_title="Price ($)",
        xaxis=dict(showticklabels=False),
        height=400,
        hovermode='y'
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Intraday chart for day trading mode
    if st.session_state.get('trading_mode') == "Day Trading (Intraday)":
        st.markdown("---")
        st.subheader(f"📊 Intraday Chart ({st.session_state.get('interval', '5min')})")

        try:
            from data_collector import TradierDataCollector

            # Fetch intraday data
            collector = TradierDataCollector(api_token)
            now = datetime.now()

            # Use today's date for market hours
            market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
            market_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

            # Format times for API
            start_time = market_start.strftime('%Y-%m-%d %H:%M')
            end_time = market_end.strftime('%Y-%m-%d %H:%M')

            intraday_df = collector.get_intraday_quotes(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=st.session_state.get('interval', '5min')
            )

            if not intraday_df.empty and 'time' in intraday_df.columns:
                # Create candlestick chart
                fig_intraday = go.Figure(data=[go.Candlestick(
                    x=intraday_df['time'],
                    open=intraday_df['open'],
                    high=intraday_df['high'],
                    low=intraday_df['low'],
                    close=intraday_df['close'],
                    name='Price'
                )])

                # Add volume bars
                fig_intraday.add_trace(go.Bar(
                    x=intraday_df['time'],
                    y=intraday_df['volume'],
                    name='Volume',
                    yaxis='y2',
                    marker_color='rgba(100, 100, 255, 0.3)'
                ))

                # Add Vanna levels if available
                if 'vanna_support_1' in pred and pred.get('vanna_support_1'):
                    fig_intraday.add_hline(
                        y=pred['vanna_support_1'],
                        line_dash="dash",
                        line_color="green",
                        annotation_text="Vanna Support",
                        annotation_position="right"
                    )

                if 'vanna_resistance_1' in pred and pred.get('vanna_resistance_1'):
                    fig_intraday.add_hline(
                        y=pred['vanna_resistance_1'],
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Vanna Resistance",
                        annotation_position="right"
                    )

                # Add profit target and stop loss (if available)
                if pred_high is not None:
                    fig_intraday.add_hline(
                        y=pred_high,
                        line_dash="dot",
                        line_color="green",
                        annotation_text=f"Target: ${pred_high:.2f}",
                        annotation_position="left"
                    )

                if pred_low is not None:
                    fig_intraday.add_hline(
                        y=pred_low,
                        line_dash="dot",
                        line_color="red",
                        annotation_text=f"Stop: ${pred_low:.2f}",
                        annotation_position="left"
                    )

                # Layout with dual y-axis
                fig_intraday.update_layout(
                    title=f"{symbol} Intraday ({st.session_state.get('interval', '5min')} bars)",
                    yaxis=dict(title='Price ($)'),
                    yaxis2=dict(title='Volume', overlaying='y', side='right'),
                    xaxis=dict(title='Time'),
                    height=500,
                    hovermode='x unified',
                    legend=dict(x=0, y=1)
                )

                st.plotly_chart(fig_intraday, use_container_width=True)

                # Intraday statistics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Bars Today", len(intraday_df))

                with col2:
                    day_high = intraday_df['high'].max()
                    st.metric("Day High", f"${day_high:.2f}")

                with col3:
                    day_low = intraday_df['low'].min()
                    st.metric("Day Low", f"${day_low:.2f}")

                with col4:
                    day_range = ((day_high - day_low) / day_low) * 100
                    st.metric("Day Range", f"{day_range:.2f}%")

            else:
                # Check if market is closed
                now = datetime.now()
                market_open = now.replace(hour=9, minute=30)
                market_close = now.replace(hour=16, minute=0)

                if now.weekday() >= 5:  # Saturday or Sunday
                    st.info("📅 Market is closed (Weekend)")
                elif now < market_open:
                    st.info(f"⏰ Market opens at 9:30 AM ET ({(market_open - now).seconds // 60} minutes)")
                elif now > market_close:
                    st.info("🔔 Market closed for the day (opens tomorrow at 9:30 AM ET)")
                else:
                    st.warning("⚠️ No intraday data available")

        except Exception as e:
            st.warning(f"⚠️ Intraday chart unavailable: {str(e)}")
            st.caption("Note: Intraday data only available during market hours (9:30 AM - 4:00 PM ET, Mon-Fri)")
            # Show debug info
            with st.expander("Debug Info"):
                st.write(f"Error type: {type(e).__name__}")
                st.write(f"Error details: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

    # Market conditions
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Market Conditions")
        
        trend = pred.get('trend_strength')
        if trend is not None:
            trend_status = "Strong 💪" if trend > 40 else "Moderate 👍" if trend > 20 else "Weak 😐"
            st.metric("Trend Strength", trend_status, f"{trend:.1f}")
        
        chop = pred.get('choppiness')
        if chop is not None:
            chop_status = "High 🌊" if chop > 50 else "Moderate 〰️" if chop > 35 else "Low ➡️"
            st.metric("Choppiness", chop_status, f"{chop:.1f}")
        
        vol = pred.get('volatility_rank')
        if vol is not None:
            vol_status = "High 🔥" if vol > 0.7 else "Moderate ⚡" if vol > 0.3 else "Low 💤"
            st.metric("Volatility Rank", vol_status, f"{vol:.2f}")
    
    with col2:
        st.subheader("⏰ Timing Factors")
        
        optimal_hours = pred.get('optimal_hours')
        if optimal_hours is not None:
            if optimal_hours:
                st.success("✅ Optimal trading hours (10am-3pm)")
            else:
                st.warning("⚠️ Outside optimal hours")
        
        # Add reasoning for avoid signal
        if not should_trade:
            st.subheader("🚫 Reasons to Avoid")
            reasons = []
            
            if chop and chop > 50:
                reasons.append("- Market is too choppy")
            if trend and trend < 20:
                reasons.append("- Trend is too weak")
            if not optimal_hours:
                reasons.append("- Outside optimal trading hours")
            if quality and quality < 40:
                reasons.append("- Very low quality setup")
            
            for reason in reasons:
                st.markdown(reason)
    
    # Historical predictions log
    st.markdown("---")
    st.subheader("📜 Prediction History")
    
    try:
        history = pd.read_csv('prediction_log.csv')
        history['timestamp'] = pd.to_datetime(history['timestamp'])
        history = history.sort_values('timestamp', ascending=False)
        
        # Display recent predictions
        st.dataframe(
            history[['timestamp', 'symbol', 'current_price', 'trade_quality_score', 
                    'should_trade', 'profit_probability', 'upside_target', 'downside_risk']].head(20),
            use_container_width=True
        )
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_signals = len(history)
            st.metric("Total Signals", total_signals)
        
        with col2:
            tradeable = history['should_trade'].sum()
            st.metric("Tradeable Setups", tradeable, 
                     delta=f"{(tradeable/total_signals*100):.1f}%")
        
        with col3:
            avg_quality = history['trade_quality_score'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")
        
    except FileNotFoundError:
        st.info("No prediction history yet. Generate predictions to build history.")

else:
    # Welcome screen
    st.info("👈 Configure settings in the sidebar and click 'Generate Prediction' to start")
    
    # Show example output
    st.markdown("### 📊 Example Output")
    st.image("https://via.placeholder.com/800x400.png?text=Price+Target+Chart", 
             caption="Price targets with support/resistance levels")
    
    # Show day trading or daily trading features
    if st.session_state.get('trading_mode') == "Day Trading (Intraday)":
        st.markdown("### ⚡ Day Trading Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Intraday Charts**
            - Live 5min/15min/1min candlesticks
            - Real-time volume analysis
            - Vanna support/resistance overlays
            - Entry/exit signals
            """)

            st.markdown("""
            **Market Hours Monitoring**
            - Pre-market alerts
            - Avoid first 30min (9:30-10:00)
            - Optimal window (10am-3pm)
            - Closing time warnings
            """)

        with col2:
            st.markdown("""
            **Day Trading Targets**
            - Profit: 0.3% same-day
            - Stop loss: 0.2% tight
            - 0DTE Vanna levels
            - Quick scalp opportunities
            """)

            st.markdown("""
            **Intraday Risk Management**
            - Position sizing (5-15% capital)
            - Time-based exits
            - No overnight holds
            - Max 2-5 trades/day
            """)
    else:
        st.markdown("### 🎯 Key Features")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            **Trade Quality Scoring**
            - 80-100: Excellent setup
            - 60-79: Good setup
            - 40-59: Marginal
            - 0-39: Avoid
            """)

            st.markdown("""
            **Market Regime Detection**
            - Identifies trending vs choppy markets
            - Filters out low-quality conditions
            - Optimizes timing
            """)

        with col2:
            st.markdown("""
            **Price Predictions**
            - Upside targets
            - Downside stops
            - Risk/reward ratios
            - Support/resistance levels
            """)

            st.markdown("""
            **Win Probability**
            - Based on historical patterns
            - Considers current market regime
            - Adjusts for time of day
            """)

# Footer
st.markdown("---")
st.caption("⚠️ Not financial advice. Use at your own risk. Past performance doesn't guarantee future results.")
