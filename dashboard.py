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
import pytz

# Load environment variables
load_dotenv()

# Set timezone to EST
EST = pytz.timezone('US/Eastern')

# Auto-download latest models from S3 on startup
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def ensure_latest_models():
    """
    Download latest models from S3 when dashboard starts
    Works on Streamlit Cloud - no local computer needed!
    """
    try:
        from s3_storage import S3StorageManager
        import glob

        # Check if we have any models locally
        local_models = glob.glob('spy_trading_model*.pkl')

        if len(local_models) == 0:
            st.info("📥 Downloading latest models from S3...")
            storage = S3StorageManager()
            storage.download_models(decompress=True)
            st.success("✓ Models downloaded and ready!")
        else:
            # Models exist, but check if we should update (every hour)
            storage = S3StorageManager()
            storage.download_models(decompress=True)

        return True
    except Exception as e:
        st.warning(f"Could not download models from S3: {e}")
        st.info("Using local models if available")
        return False

# Download models at startup
ensure_latest_models()

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

    # Get token from Streamlit secrets (cloud) or environment (.env file)
    api_token = None

    # Try Streamlit secrets first (for cloud deployment)
    try:
        if 'TRADIER_API_TOKEN' in st.secrets:
            api_token = st.secrets['TRADIER_API_TOKEN']
            st.success("✓ API token loaded from Streamlit secrets")
            # Show masked token
            masked_token = api_token[:4] + "..." + api_token[-4:] if len(api_token) > 8 else "****"
            st.caption(f"Token: {masked_token}")
    except:
        pass

    # If not in secrets, try environment variable
    if not api_token:
        env_token = os.getenv('TRADIER_API_TOKEN', '')
        if env_token:
            api_token = env_token
            st.success("✓ API token loaded from .env")
            # Show masked token
            masked_token = env_token[:4] + "..." + env_token[-4:] if len(env_token) > 8 else "****"
            st.caption(f"Token: {masked_token}")
        else:
            st.warning("⚠️ No API token found")
            api_token = st.text_input("Tradier API Token", type="password",
                                       help="Add to Streamlit secrets or create .env file with TRADIER_API_TOKEN=your_token")

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

    # Check if prediction is None (insufficient data)
    if pred is None:
        st.warning("⚠️ Unable to generate predictions. This could be due to:")
        st.info("""
        - No API token configured (or invalid token)
        - Insufficient historical data
        - Market is closed (for intraday mode)
        - API rate limit reached

        Try again during market hours or check your API token.
        """)
    else:
        # Header row
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            # Convert timestamp to EST for display
            timestamp_utc = pd.to_datetime(pred['timestamp'])
            timestamp_est = timestamp_utc.tz_localize('UTC').tz_convert(EST)
            formatted_time = timestamp_est.strftime('%Y-%m-%d %I:%M:%S %p EST')

            st.markdown(f"### {pred['symbol']} @ ${pred['current_price']:.2f}")
            st.caption(f"Last updated: {formatted_time}")
    
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
    
        # Price chart with targets and Vanna levels
        st.subheader("📊 Price Targets with Vanna Levels")

        current = pred['current_price']
        pred_high = pred.get('predicted_high', current * 1.01)
        pred_low = pred.get('predicted_low', current * 0.99)

        # Fetch recent price data for candlesticks
        price_df = None
        try:
            from data_collector import TradierDataCollector
            collector = TradierDataCollector(api_token)

            # Get last 2 days of 5-min bars for recent context
            now = datetime.now()
            start = (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')
            end = now.strftime('%Y-%m-%d %H:%M')

            price_df = collector.get_intraday_quotes(
                symbol=symbol,
                start_time=start,
                end_time=end,
                interval='5min'
            )

            # Create candlestick chart if we have data
            if not price_df.empty and len(price_df) > 0:
                # Take last 40 bars for clean display
                price_df = price_df.tail(40).copy()

                fig = go.Figure(data=[go.Candlestick(
                    x=price_df.index if 'time' not in price_df.columns else price_df['time'],
                    open=price_df['open'],
                    high=price_df['high'],
                    low=price_df['low'],
                    close=price_df['close'],
                    name='SPY',
                    increasing_line_color='#00ff00',
                    decreasing_line_color='#ff0000',
                    increasing_fillcolor='#00ff00',
                    decreasing_fillcolor='#ff0000'
                )])
            else:
                # If no intraday data, create empty chart
                fig = go.Figure()
                st.info("Using levels view (intraday data unavailable)")

        except Exception as e:
            st.info(f"Using levels view: {str(e)}")
            fig = go.Figure()

        # Entry price (current)
        fig.add_hline(
            y=current,
            line_dash="solid",
            line_color="blue",
            line_width=3,
            annotation_text=f"Entry: ${current:.2f}",
            annotation_position="right"
        )

        # Profit Target
        if pred_high:
            upside_pct = ((pred_high - current) / current * 100)
            fig.add_hline(
                y=pred_high,
                line_dash="dot",
                line_color="green",
                line_width=2,
                annotation_text=f"🎯 Target: ${pred_high:.2f} (+{upside_pct:.1f}%)",
                annotation_position="right"
            )

        # Stop Loss
        if pred_low:
            downside_pct = ((current - pred_low) / current * 100)
            fig.add_hline(
                y=pred_low,
                line_dash="dot",
                line_color="red",
                line_width=2,
                annotation_text=f"🛑 Stop: ${pred_low:.2f} (-{downside_pct:.1f}%)",
                annotation_position="right"
            )

        # Add Vanna resistance levels (if available) - Negative Vanna = Repellent
        if 'vanna_resistance_1' in pred and pred['vanna_resistance_1']:
            strength = pred.get('vanna_resistance_1_strength')
            strength_text = f" | -{abs(strength):.2f} ↔️ Repellent" if strength else ""
            fig.add_hline(
                y=pred['vanna_resistance_1'],
                line_dash="dash",
                line_color="orange",
                line_width=1,
                annotation_text=f"Vanna R1: ${pred['vanna_resistance_1']:.2f}{strength_text}",
                annotation_position="left"
            )

        if 'vanna_resistance_2' in pred and pred['vanna_resistance_2']:
            strength = pred.get('vanna_resistance_2_strength')
            strength_text = f" | -{abs(strength):.2f} ↔️ Repellent" if strength else ""
            fig.add_hline(
                y=pred['vanna_resistance_2'],
                line_dash="dash",
                line_color="orange",
                line_width=1,
                annotation_text=f"Vanna R2: ${pred['vanna_resistance_2']:.2f}{strength_text}",
                annotation_position="left"
            )

        # Add Vanna support levels (if available) - Positive Vanna = Attractor
        if 'vanna_support_1' in pred and pred['vanna_support_1']:
            strength = pred.get('vanna_support_1_strength')
            strength_text = f" | +{abs(strength):.2f} 🧲 Attractor" if strength else ""
            fig.add_hline(
                y=pred['vanna_support_1'],
                line_dash="dash",
                line_color="purple",
                line_width=1,
                annotation_text=f"Vanna S1: ${pred['vanna_support_1']:.2f}{strength_text}",
                annotation_position="left"
            )

        if 'vanna_support_2' in pred and pred['vanna_support_2']:
            strength = pred.get('vanna_support_2_strength')
            strength_text = f" | +{abs(strength):.2f} 🧲 Attractor" if strength else ""
            fig.add_hline(
                y=pred['vanna_support_2'],
                line_dash="dash",
                line_color="purple",
                line_width=1,
                annotation_text=f"Vanna S2: ${pred['vanna_support_2']:.2f}{strength_text}",
                annotation_position="left"
            )

        # Add GEX (Gamma Exposure) hedge levels
        if 'gex_zero_level' in pred and pred['gex_zero_level']:
            fig.add_hline(
                y=pred['gex_zero_level'],
                line_dash="dashdot",
                line_color="cyan",
                line_width=2,
                annotation_text=f"GEX Flip: ${pred['gex_zero_level']:.2f} ⚡",
                annotation_position="right"
            )

        if 'gex_support' in pred and pred['gex_support']:
            fig.add_hline(
                y=pred['gex_support'],
                line_dash="dot",
                line_color="lime",
                line_width=1,
                annotation_text=f"GEX Support: ${pred['gex_support']:.0f} (Dealers BUY)",
                annotation_position="right"
            )

        if 'gex_resistance' in pred and pred['gex_resistance']:
            fig.add_hline(
                y=pred['gex_resistance'],
                line_dash="dot",
                line_color="magenta",
                line_width=1,
                annotation_text=f"GEX Resistance: ${pred['gex_resistance']:.0f} (Dealers SELL)",
                annotation_position="right"
            )

        # Fill area between profit target and stop loss
        if pred_high and pred_low:
            fig.add_hrect(
                y0=pred_low, y1=pred_high,
                fillcolor="green", opacity=0.1,
                line_width=0
            )

        # Calculate y-axis range centered on current price and all levels
        all_levels = [current, pred_high, pred_low]
        if 'vanna_resistance_1' in pred and pred['vanna_resistance_1']:
            all_levels.append(pred['vanna_resistance_1'])
        if 'vanna_resistance_2' in pred and pred['vanna_resistance_2']:
            all_levels.append(pred['vanna_resistance_2'])
        if 'vanna_support_1' in pred and pred['vanna_support_1']:
            all_levels.append(pred['vanna_support_1'])
        if 'vanna_support_2' in pred and pred['vanna_support_2']:
            all_levels.append(pred['vanna_support_2'])
        # Add GEX levels to range
        if 'gex_support' in pred and pred['gex_support']:
            all_levels.append(pred['gex_support'])
        if 'gex_resistance' in pred and pred['gex_resistance']:
            all_levels.append(pred['gex_resistance'])
        if 'gex_zero_level' in pred and pred['gex_zero_level']:
            all_levels.append(pred['gex_zero_level'])

        # Set y-axis range with some padding
        y_min = min(all_levels) * 0.998  # 0.2% below lowest level
        y_max = max(all_levels) * 1.002  # 0.2% above highest level

        fig.update_layout(
            title=f"{symbol} Trading Setup with Vanna & GEX Levels",
            yaxis_title="Price ($)",
            xaxis_title="Time",
            yaxis=dict(range=[y_min, y_max]),
            height=600,
            showlegend=True,
            hovermode='x unified',
            template='plotly_dark'
        )

        # Update xaxis to show rangeslider for navigation
        fig.update_xaxes(rangeslider_visible=False)

        st.plotly_chart(fig, use_container_width=True)

        # Chart legend
        st.markdown("""
        **Legend:** 🔵 Entry | 🎯 Target | 🛑 Stop | 🟠 Vanna R | 🟣 Vanna S | ⚡ GEX Flip | 💚 GEX Support | 💜 GEX Resistance
        """)

        # GEX Regime indicator
        if pred.get('gex_regime'):
            regime = pred['gex_regime']
            if regime == 'positive':
                st.success("📊 **GEX Regime: POSITIVE** - Mean reversion mode (fade extremes)")
            else:
                st.warning("📊 **GEX Regime: NEGATIVE** - Momentum mode (trend following)")

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
    
        # Keep this section for backwards compatibility but it's now moved below

else:
    # Welcome screen
    st.info("👈 Configure settings in the sidebar and click 'Generate Prediction' to start")

# Always show prediction history (moved outside the predictions block)
st.markdown("---")
st.subheader("📜 Prediction History")

try:
    history = pd.read_csv('prediction_log.csv')

    # Filter to only valid predictions (with quality scores)
    valid_history = history[history['trade_quality_score'].notna()].copy()

    if len(valid_history) > 0:
        # Convert timestamps to EST
        valid_history['timestamp'] = pd.to_datetime(valid_history['timestamp'], utc=True)
        valid_history['timestamp'] = valid_history['timestamp'].dt.tz_convert(EST)
        valid_history = valid_history.sort_values('timestamp', ascending=False)

        # Format for display
        display_history = valid_history.copy()
        display_history['timestamp'] = display_history['timestamp'].dt.strftime('%Y-%m-%d %I:%M:%S %p EST')

        # Display recent predictions
        st.dataframe(
            display_history[['timestamp', 'symbol', 'current_price', 'trade_quality_score',
                    'should_trade', 'profit_probability', 'upside_target', 'downside_risk']].head(20),
            use_container_width=True
        )

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            total_signals = len(valid_history)
            st.metric("Total Signals", total_signals)

        with col2:
            tradeable = valid_history['should_trade'].sum()
            st.metric("Tradeable Setups", int(tradeable),
                     delta=f"{(tradeable/total_signals*100):.1f}%")

        with col3:
            avg_quality = valid_history['trade_quality_score'].mean()
            st.metric("Avg Quality Score", f"{avg_quality:.1f}/100")
    else:
        st.info("No valid predictions yet. Generate predictions to build history.")

except FileNotFoundError:
    st.info("No prediction history yet. Generate predictions to build history.")

# Example chart (only show if no active predictions)
if 'predictions' not in st.session_state:
    st.markdown("---")
    # Show example output with actual SPY candlestick data
    st.markdown("### 📊 Example Trading Setup")

    # Try to load actual SPY data
    try:
        # Try to load from CSV if available
        if os.path.exists('spy_training_data_daily.csv'):
            spy_df = pd.read_csv('spy_training_data_daily.csv')
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df = spy_df.sort_values('date').tail(30)  # Last 30 days

            current_price = spy_df.iloc[-1]['close']

            # Calculate Vanna levels based on recent price action
            vanna_resistance_1 = spy_df['high'].tail(10).max()
            vanna_resistance_2 = vanna_resistance_1 + (vanna_resistance_1 - current_price) * 0.5
            vanna_support_1 = spy_df['low'].tail(10).min()
            vanna_support_2 = vanna_support_1 - (current_price - vanna_support_1) * 0.5

            profit_target = current_price + (current_price * 0.015)  # 1.5% target
            stop_loss = current_price - (current_price * 0.008)  # 0.8% stop
            entry_price = current_price

            # Create candlestick chart
            fig_example = go.Figure(data=[go.Candlestick(
                x=spy_df['date'],
                open=spy_df['open'],
                high=spy_df['high'],
                low=spy_df['low'],
                close=spy_df['close'],
                name='SPY'
            )])
        else:
            raise FileNotFoundError("No SPY data available")

    except Exception as e:
        # Fallback to simple example if no data available
        st.caption("(Using simulated data - click 'Generate Prediction' for real data)")
        current_price = 450.00
        vanna_resistance_1 = 455.50
        vanna_resistance_2 = 458.00
        vanna_support_1 = 447.50
        vanna_support_2 = 445.00
        profit_target = 453.50
        stop_loss = 448.00
        entry_price = current_price
        fig_example = go.Figure()

    # Current price / Entry
    fig_example.add_hline(
        y=entry_price,
        line_dash="solid",
        line_color="blue",
        line_width=3,
        annotation_text=f"Entry: ${entry_price:.2f}",
        annotation_position="right"
    )

    # Profit Target
    fig_example.add_hline(
        y=profit_target,
        line_dash="dot",
        line_color="green",
        line_width=2,
        annotation_text=f"🎯 Target: ${profit_target:.2f} (+{((profit_target-entry_price)/entry_price*100):.1f}%)",
        annotation_position="right"
    )

    # Stop Loss
    fig_example.add_hline(
        y=stop_loss,
        line_dash="dot",
        line_color="red",
        line_width=2,
        annotation_text=f"🛑 Stop: ${stop_loss:.2f} (-{((entry_price-stop_loss)/entry_price*100):.1f}%)",
        annotation_position="right"
    )

    # Vanna Resistance Levels (Negative Vanna = Repellent)
    fig_example.add_hline(
        y=vanna_resistance_1,
        line_dash="dash",
        line_color="orange",
        line_width=1,
        annotation_text=f"Vanna R1: ${vanna_resistance_1:.2f} | -0.32 ↔️ Repellent",
        annotation_position="left"
    )

    fig_example.add_hline(
        y=vanna_resistance_2,
        line_dash="dash",
        line_color="orange",
        line_width=1,
        annotation_text=f"Vanna R2: ${vanna_resistance_2:.2f} | -0.18 ↔️ Repellent",
        annotation_position="left"
    )

    # Vanna Support Levels (Positive Vanna = Attractor)
    fig_example.add_hline(
        y=vanna_support_1,
        line_dash="dash",
        line_color="purple",
        line_width=1,
        annotation_text=f"Vanna S1: ${vanna_support_1:.2f} | +0.28 🧲 Attractor",
        annotation_position="left"
    )

    fig_example.add_hline(
        y=vanna_support_2,
        line_dash="dash",
        line_color="purple",
        line_width=1,
        annotation_text=f"Vanna S2: ${vanna_support_2:.2f} | +0.41 🧲 Attractor",
        annotation_position="left"
    )

    # Add shaded region for expected trading range
    fig_example.add_hrect(
        y0=stop_loss, y1=profit_target,
        fillcolor="green", opacity=0.1,
        line_width=0
    )

    fig_example.update_layout(
        title="SPY - Example Trading Setup with Vanna Levels",
        yaxis_title="Price ($)",
        xaxis_title="Date" if os.path.exists('spy_training_data_daily.csv') else "",
        height=600,
        showlegend=False,
        xaxis_rangeslider_visible=False,  # Hide rangeslider for cleaner look
    )

    st.plotly_chart(fig_example, use_container_width=True)

    # Show current stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Entry Price", f"${entry_price:.2f}")
    with col2:
        upside = ((profit_target - entry_price) / entry_price * 100)
        st.metric("Profit Target", f"${profit_target:.2f}", f"+{upside:.1f}%")
    with col3:
        downside = ((entry_price - stop_loss) / entry_price * 100)
        st.metric("Stop Loss", f"${stop_loss:.2f}", f"-{downside:.1f}%", delta_color="inverse")
    with col4:
        rr = upside / downside if downside > 0 else 0
        st.metric("Risk/Reward", f"{rr:.2f}:1")

    # Explanation
    st.markdown("---")
    st.markdown("""
    **Chart Legend:**
    - 📊 **Candlesticks**: Actual SPY price action (last 30 days)
    - 🔵 **Blue Line**: Entry price (current market price)
    - 🎯 **Green Dotted**: Profit target (ML predicted high, +1.5%)
    - 🛑 **Red Dotted**: Stop loss (ML predicted low, -0.8%)
    - 🟠 **Orange Dashed**: Vanna resistance levels (options dealer hedging creates selling pressure)
    - 🟣 **Purple Dashed**: Vanna support levels (options dealer hedging creates buying pressure)
    - 🟢 **Green Shaded**: Expected trading range (risk/reward zone)

    **How to Use:**
    1. Enter at blue line (current price)
    2. Take profit at green dotted line (target)
    3. Stop out at red dotted line (protective stop)
    4. Watch Vanna levels for potential reversals
    """)
    
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
