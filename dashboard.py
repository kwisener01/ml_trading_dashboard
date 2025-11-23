import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import json
from predictor import TradingPredictor
import os
from dotenv import load_dotenv
import pytz
import time
import base64

# Load environment variables
load_dotenv()

# Set timezone to EST
EST = pytz.timezone('US/Eastern')

def create_download_link(file_path, link_text):
    """Create an HTML link that opens a file in a new browser tab"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:text/html;base64,{b64}" target="_blank" style="display: inline-block; padding: 0.5em 1em; background-color: #0066cc; color: white; text-decoration: none; border-radius: 5px; font-weight: bold;">{link_text}</a>'
    return href

# Helper function for VWAP calculation
def calculate_vwap(price_df):
    """Calculate Volume-Weighted Average Price"""
    if price_df is None or price_df.empty or 'volume' not in price_df.columns:
        return None
    try:
        typical_price = (price_df['high'] + price_df['low'] + price_df['close']) / 3
        vwap = (typical_price * price_df['volume']).cumsum() / price_df['volume'].cumsum()
        return vwap.iloc[-1] if len(vwap) > 0 else None
    except:
        return None

# Multi-panel chart builder
def create_options_flow_chart(pred, price_df, symbol):
    """
    Create 3-panel chart with:
    - Panel 1: Price with options flow levels
    - Panel 2: IV & Vanna indicators
    - Panel 3: Dealer flow indicators
    """
    # Create subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.175, 0.175],
        subplot_titles=(
            f"{symbol} - Options Flow Analysis",
            "IV & Vanna Panel",
            "Dealer Flow Panel"
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    current = pred['current_price']
    gex_regime = pred.get('gex_regime', 'unknown')

    # Extract all levels
    vanna_r1 = pred.get('vanna_resistance_1')
    vanna_r2 = pred.get('vanna_resistance_2')
    vanna_s1 = pred.get('vanna_support_1')
    vanna_s2 = pred.get('vanna_support_2')
    gex_support = pred.get('gex_support')
    gex_resistance = pred.get('gex_resistance')
    gex_flip = pred.get('gex_zero_level')
    put_wall = pred.get('put_wall')
    call_wall = pred.get('call_wall')

    # Calculate VWAP
    vwap = calculate_vwap(price_df)

    # ========== PANEL 1: PRICE CHART ==========

    # Determine x-axis range
    has_price_data = price_df is not None and not price_df.empty and len(price_df) > 0

    if has_price_data:
        # Use actual price data
        x_data = price_df.index if 'time' not in price_df.columns else price_df['time']
        fig.add_trace(go.Candlestick(
            x=x_data,
            open=price_df['open'],
            high=price_df['high'],
            low=price_df['low'],
            close=price_df['close'],
            name='Price',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350',
            increasing_fillcolor='#26a69a',
            decreasing_fillcolor='#ef5350',
            showlegend=False
        ), row=1, col=1)
        x_range = [x_data.iloc[0], x_data.iloc[-1]]
    else:
        # No price data - show current price as a single point
        now = datetime.now(EST)
        x_point = now
        x_range = [now - timedelta(hours=2), now + timedelta(hours=2)]

        # Add current price marker
        fig.add_trace(go.Scatter(
            x=[x_point],
            y=[current],
            mode='markers',
            marker=dict(size=15, color='#2196F3', symbol='diamond'),
            name='Current Price',
            showlegend=False
        ), row=1, col=1)

    # Add regime shading
    if gex_regime == 'positive':
        fig.add_vrect(
            x0=x_range[0], x1=x_range[1],
            fillcolor="rgba(0, 255, 0, 0.05)",
            layer="below",
            line_width=0,
            row=1, col=1
        )
    elif gex_regime == 'negative':
        fig.add_vrect(
            x0=x_range[0], x1=x_range[1],
            fillcolor="rgba(255, 0, 0, 0.05)",
            layer="below",
            line_width=0,
            row=1, col=1
        )

    # Helper to check if level is within valid range
    def level_valid(level, pct=0.05):
        return level and abs(level - current) / current <= pct

    # Add Gamma Flip (Major Pivot)
    if gex_flip and level_valid(gex_flip):
        fig.add_hline(
            y=gex_flip,
            line_dash="solid",
            line_color="#00BCD4",
            line_width=3,
            annotation_text=f"GAMMA FLIP: ${gex_flip:.2f} (Major Pivot)",
            annotation_position="right",
            annotation=dict(font=dict(size=10, color="white"), bgcolor="#00838F"),
            row=1, col=1
        )

    # Add GEX Walls (S/R Zones)
    if gex_support and level_valid(gex_support):
        fig.add_hline(
            y=gex_support,
            line_dash="dash",
            line_color="#76FF03",
            line_width=2,
            annotation_text=f"GEX SUPPORT: ${gex_support:.0f} (S/R Zone)",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#33691E"),
            row=1, col=1
        )

    if gex_resistance and level_valid(gex_resistance):
        fig.add_hline(
            y=gex_resistance,
            line_dash="dash",
            line_color="#E040FB",
            line_width=2,
            annotation_text=f"GEX RESISTANCE: ${gex_resistance:.0f} (S/R Zone)",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#6A1B9A"),
            row=1, col=1
        )

    # Add Vanna Walls (Pressure Zones)
    if vanna_s1 and level_valid(vanna_s1):
        strength = pred.get('vanna_support_1_strength', 0)
        fig.add_hline(
            y=vanna_s1,
            line_dash="dot",
            line_color="#9C27B0",
            line_width=2,
            annotation_text=f"VANNA S1: ${vanna_s1:.2f} (Pressure Zone)",
            annotation_position="right",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#7B1FA2"),
            row=1, col=1
        )

    if vanna_r1 and level_valid(vanna_r1):
        strength = pred.get('vanna_resistance_1_strength', 0)
        fig.add_hline(
            y=vanna_r1,
            line_dash="dot",
            line_color="#FF9800",
            line_width=2,
            annotation_text=f"VANNA R1: ${vanna_r1:.2f} (Pressure Zone)",
            annotation_position="right",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#FF6D00"),
            row=1, col=1
        )

    # Add Put/Call Walls (Magnets/Barriers)
    if put_wall and level_valid(put_wall):
        fig.add_hline(
            y=put_wall,
            line_dash="dashdot",
            line_color="#00E676",
            line_width=2,
            annotation_text=f"PUT WALL: ${put_wall:.2f} (Magnet)",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="black"), bgcolor="#B9F6CA"),
            row=1, col=1
        )

    if call_wall and level_valid(call_wall):
        fig.add_hline(
            y=call_wall,
            line_dash="dashdot",
            line_color="#FF5252",
            line_width=2,
            annotation_text=f"CALL WALL: ${call_wall:.2f} (Barrier)",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="black"), bgcolor="#FF8A80"),
            row=1, col=1
        )

    # Add VWAP (Dynamic Balance Line)
    if vwap and level_valid(vwap):
        fig.add_hline(
            y=vwap,
            line_dash="solid",
            line_color="#FFC107",
            line_width=2,
            annotation_text=f"VWAP: ${vwap:.2f} (Dynamic Balance)",
            annotation_position="right",
            annotation=dict(font=dict(size=9, color="black"), bgcolor="#FFF9C4"),
            row=1, col=1
        )

    # Add current price line
    fig.add_hline(
        y=current,
        line_dash="solid",
        line_color="#2196F3",
        line_width=3,
        annotation_text=f"CURRENT: ${current:.2f}",
        annotation_position="left",
        annotation=dict(font=dict(size=11, color="white"), bgcolor="#2196F3"),
        row=1, col=1
    )

    # ========== PANEL 2: IV & VANNA ==========

    # Create x-axis for panels 2 and 3
    if has_price_data:
        panel_x = price_df.index if 'time' not in price_df.columns else price_df['time']
        panel_x_single = [panel_x.iloc[-1]]
    else:
        # Use current time when no price data
        panel_x_single = [datetime.now(EST)]

    # IV (Implied Volatility)
    iv = pred.get('iv', 0.2) * 100  # Convert to percentage
    iv_percentile = pred.get('iv_percentile', 50)
    fig.add_trace(go.Scatter(
        x=panel_x_single,
        y=[iv],
        mode='markers+text',
        name='IV',
        marker=dict(size=15, color='#FF6B6B'),
        text=[f"IV: {iv:.1f}%<br>Percentile: {iv_percentile:.0f}"],
        textposition="top center",
        showlegend=True
    ), row=2, col=1)

    # Vanna (dealer bias)
    vanna_s1_str = pred.get('vanna_support_1_strength', 0) or 0
    vanna_r1_str = pred.get('vanna_resistance_1_strength', 0) or 0
    net_vanna = vanna_s1_str + vanna_r1_str  # R1 is negative
    fig.add_trace(go.Bar(
        x=panel_x_single,
        y=[net_vanna * 100],
        name='Net Vanna',
        marker_color='#4ECDC4' if net_vanna > 0 else '#FF6B6B',
        showlegend=True
    ), row=2, col=1)

    # Vanna × IV (trend indication)
    vanna_iv_trend = pred.get('vanna_iv_trend', 0)
    fig.add_trace(go.Scatter(
        x=panel_x_single,
        y=[vanna_iv_trend],
        mode='markers+text',
        name='Vanna×IV',
        marker=dict(size=12, color='#95E1D3', symbol='diamond'),
        text=[f"{vanna_iv_trend:.1f}"],
        textposition="bottom center",
        showlegend=True
    ), row=2, col=1)

    # ========== PANEL 3: DEALER FLOW ==========

    # Charm (time decay flows)
    charm = pred.get('charm', 0)
    charm_pressure = pred.get('charm_pressure', 0)
    fig.add_trace(go.Bar(
        x=panel_x_single,
        y=[charm_pressure],
        name='Charm Pressure',
        marker_color='#A8E6CF',
        showlegend=True
    ), row=3, col=1)

    # GEX Pressure (reaction strength)
    gex_pressure = 50 if gex_regime == 'positive' else -50 if gex_regime == 'negative' else 0
    fig.add_trace(go.Bar(
        x=panel_x_single,
        y=[gex_pressure],
        name='GEX Pressure',
        marker_color='#FFD93D' if gex_pressure > 0 else '#FF6B9D',
        showlegend=True
    ), row=3, col=1)

    # Dealer Flow Score
    dealer_score = pred.get('dealer_flow_score', 0)
    fig.add_trace(go.Scatter(
        x=panel_x_single,
        y=[dealer_score],
        mode='markers+text',
        name='Dealer Flow Score',
        marker=dict(size=15, color='#6BCF7F' if dealer_score > 0 else '#F76B8A',
                   symbol='star'),
        text=[f"Score: {dealer_score:.0f}"],
        textposition="top center",
        showlegend=True
    ), row=3, col=1)

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(30, 30, 30, 1)',
        margin=dict(l=50, r=120, t=80, b=50)
    )

    # Update y-axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="IV/Vanna", row=2, col=1)
    fig.update_yaxes(title_text="Dealer Flow", row=3, col=1, range=[-100, 100])

    # Update x-axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    return fig

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="ML Trading Dashboard",
    page_icon="📈",
    layout="wide"
)

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

        # Auto-refresh settings
        st.markdown("### 🔄 Auto-Refresh")
        auto_refresh = st.checkbox("Enable Auto-Refresh", value=False)

        if auto_refresh:
            # Set refresh interval based on chart interval
            refresh_intervals = {'1min': 60, '5min': 300, '15min': 900}
            refresh_seconds = refresh_intervals.get(interval, 300)
            refresh_ms = refresh_seconds * 1000

            st.caption(f"Refreshing every {refresh_seconds // 60} min")

            # JavaScript-based auto-refresh for reliability
            st.markdown(
                f"""
                <script>
                    var refreshInterval = {refresh_ms};
                    var countdown = refreshInterval / 1000;

                    function updateCountdown() {{
                        var mins = Math.floor(countdown / 60);
                        var secs = countdown % 60;
                        var display = mins + ":" + (secs < 10 ? "0" : "") + secs;
                        var elem = document.getElementById("countdown-display");
                        if (elem) elem.innerText = "Next refresh: " + display;

                        if (countdown <= 0) {{
                            window.location.reload();
                        }} else {{
                            countdown--;
                            setTimeout(updateCountdown, 1000);
                        }}
                    }}

                    // Start countdown
                    setTimeout(updateCountdown, 1000);
                </script>
                <p id="countdown-display" style="font-size: 12px; color: gray;">Starting countdown...</p>
                """,
                unsafe_allow_html=True
            )

            # Also trigger prediction generation on auto-refresh
            st.session_state['generate_prediction'] = True
            st.session_state['trading_mode'] = trading_mode
            st.session_state['interval'] = interval

        # Show market hours status (in EST)
        now = datetime.now(EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        optimal_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        optimal_end = now.replace(hour=15, minute=0, second=0, microsecond=0)

        # NY Morning Priority Session (10:00 AM - 12:00 PM EST)
        priority_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        priority_end = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Display current EST time
        st.caption(f"🕐 {now.strftime('%I:%M %p EST')}")

        # Check if in NY morning priority session
        in_priority_session = priority_start <= now <= priority_end and now.weekday() < 5

        if now < market_open:
            st.warning(f"⏸️ Pre-market ({(market_open - now).seconds // 60} min to open)")
        elif now > market_close:
            st.info("🔔 Market closed")
        elif now < optimal_start:
            st.warning("⚠️ Avoid zone (first 30 min)")
        elif now > optimal_end:
            st.warning("⚠️ Closing time (last hour)")
        elif in_priority_session:
            st.success("🌟 **NY MORNING PRIORITY** 🌟")
            st.caption("Prime liquidity window (10AM-12PM)")
        else:
            st.success("✅ Optimal trading hours")

        # NY Morning Priority Session Info Box
        st.markdown("---")
        st.markdown("### 🌟 Priority Session")

        if in_priority_session:
            # Show countdown to end of priority session
            time_remaining = (priority_end - now).seconds
            mins_remaining = time_remaining // 60
            st.info(f"""
            **ACTIVE NOW**

            ⏱️ {mins_remaining} minutes remaining

            📈 Highest liquidity & volatility
            🎯 Best setups typically occur now
            ⚡ Focus on quality entries
            """)
        elif now < priority_start and now >= market_open:
            # Show countdown to start of priority session
            time_until = (priority_start - now).seconds
            mins_until = time_until // 60
            st.warning(f"""
            **Starts in {mins_until} min**

            ⏰ Priority session: 10AM-12PM EST
            🔔 Get ready for prime trading time
            """)
        elif now > priority_end and now < market_close:
            st.caption("""
            **Priority Session Ended**

            Next session: Tomorrow 10AM-12PM EST
            """)
        else:
            st.caption("""
            **NY Morning Priority Session**

            ⏰ Daily: 10:00 AM - 12:00 PM EST
            📊 Prime trading window
            💎 Best liquidity & setups
            """)
    else:
        interval = 'daily'

    trade_quality_threshold = st.slider("Trade Quality Threshold", 0, 100, 60)

    if st.button("🔄 Generate Prediction", type="primary"):
        st.session_state['generate_prediction'] = True
        st.session_state['trading_mode'] = trading_mode
        st.session_state['interval'] = interval
        if 'last_refresh' in st.session_state:
            st.session_state['last_refresh'] = time.time()  # Reset timer on manual refresh
    
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
        - **🌟 NY Morning Priority (10AM-12PM)**
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

    # Trading Guidebook
    st.markdown("---")
    st.markdown("### 📖 Trading Guidebook")

    with st.expander("🎯 Combined Hedge Levels", expanded=False):
        st.image("Vanna_Gamma_Hedge.png", use_column_width=True)
        st.caption("""
        **Key Zones:**
        - 🟢 **BOUNCE ZONE**: Strong support, dealers BUY
        - 🔴 **REJECTION ZONE**: Strong resistance, dealers SELL
        - ⚡ **GEX Flip**: Above = mean reversion, Below = momentum
        """)

    with st.expander("📊 GEX Regime Behavior", expanded=False):
        if os.path.exists("gex_regime_diagram.png"):
            st.image("gex_regime_diagram.png", use_column_width=True)
        st.caption("""
        **Positive GEX** (Mean Reversion):
        - Fade extremes, sell rallies, buy dips
        - Tight ranges expected

        **Negative GEX** (Momentum):
        - Follow the trend, chase breaks
        - Large moves expected
        """)

    with st.expander("🧲 Vanna Level Reactions", expanded=False):
        if os.path.exists("vanna_reaction_diagram.png"):
            st.image("vanna_reaction_diagram.png", use_column_width=True)
        st.caption("""
        **Positive Vanna** (Support):
        - Acts as magnet, pulls price down
        - Strong bounce expected

        **Negative Vanna** (Resistance):
        - Repels price away
        - Rejection expected
        """)

    with st.expander("📋 Trading Playbook", expanded=False):
        st.markdown("### How to Use the 3-Panel Chart")

        # Create tabs for different scenarios
        tab1, tab2, tab3 = st.tabs(["🟢 Bullish Setup", "🔴 Bearish Setup", "⚪ Range-Bound"])

        with tab1:
            st.markdown("""
            **🟢 BULLISH SETUP:**
            - ✅ Price **above** Gamma Flip (positive GEX regime = green background)
            - ✅ Vanna support below provides bounce levels
            - ✅ Rising Vanna×IV trend (Panel 2)
            - ✅ Positive Dealer Flow Score (Panel 3)

            **Trading Strategy:**
            - Look for dips to Vanna Support or VWAP for LONG entries
            - Target: Call Wall above
            - Stop: Below GEX Support
            """)
            if os.path.exists("bullish_setup_example.html"):
                link = create_download_link("bullish_setup_example.html", "📊 Open Bullish Setup Chart in New Tab")
                st.markdown(link, unsafe_allow_html=True)
                st.caption("⬆️ Click to open the interactive chart in a new browser tab")

        with tab2:
            st.markdown("""
            **🔴 BEARISH SETUP:**
            - ✅ Price **below** Gamma Flip (negative GEX regime = red background)
            - ✅ Vanna resistance above acts as ceiling
            - ✅ Rising IV with falling Vanna×IV (Panel 2)
            - ✅ Negative Dealer Flow Score (Panel 3)

            **Trading Strategy:**
            - Look for rallies to Vanna Resistance or VWAP for SHORT entries
            - Target: Put Wall below
            - Stop: Above GEX Resistance
            """)
            if os.path.exists("bearish_setup_example.html"):
                link = create_download_link("bearish_setup_example.html", "📊 Open Bearish Setup Chart in New Tab")
                st.markdown(link, unsafe_allow_html=True)
                st.caption("⬆️ Click to open the interactive chart in a new browser tab")

        with tab3:
            st.markdown("""
            **⚪ RANGE-BOUND SETUP:**
            - ✅ Price **at** Gamma Flip with tight GEX walls
            - ✅ Put Wall below & Call Wall above define range
            - ✅ Stable IV and Vanna indicators (Panel 2)
            - ✅ Near-neutral Dealer Flow (Panel 3)

            **Trading Strategy:**
            - BUY at GEX Support / Put Wall
            - SELL at GEX Resistance / Call Wall
            - Avoid breakout trades until Dealer Flow shifts decisively
            """)
            if os.path.exists("range-bound_setup_example.html"):
                link = create_download_link("range-bound_setup_example.html", "📊 Open Range-Bound Setup Chart in New Tab")
                st.markdown(link, unsafe_allow_html=True)
                st.caption("⬆️ Click to open the interactive chart in a new browser tab")

        st.divider()
        st.caption("""
        **Quick Reference:**
        - **Gamma Flip** (cyan solid) = Major pivot point where dealer hedging changes
        - **GEX Walls** (green/red dash) = Support/Resistance zones from gamma exposure
        - **Vanna Walls** (green/red dot) = Pressure zones from vanna exposure
        - **Put/Call Walls** (purple/orange) = Price magnets from open interest
        - **VWAP** (yellow) = Dynamic balance point for the session
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
            # IV Percentile
            iv_pct = pred.get('iv_percentile', 50)
            st.metric("IV Percentile", f"{iv_pct:.0f}%")

        with col3:
            # Vanna x IV Trend
            vanna_iv = pred.get('vanna_iv_trend', 0)
            st.metric("Vanna×IV Trend", f"{vanna_iv:.1f}")

        with col4:
            # Dealer Flow Score
            dealer_score = pred.get('dealer_flow_score', 0)
            score_status = "Bullish" if dealer_score > 20 else "Bearish" if dealer_score < -20 else "Neutral"
            st.metric("Dealer Flow", score_status, f"{dealer_score:.0f}")

        # Net Hedge Pressure Indicator
        st.markdown("---")
        st.subheader("⚡ Net Hedge Pressure")

        # Calculate hedge pressure based on price proximity to levels
        current = pred['current_price']

        # Get all levels
        vanna_s1 = pred.get('vanna_support_1')
        vanna_s2 = pred.get('vanna_support_2')
        vanna_r1 = pred.get('vanna_resistance_1')
        vanna_r2 = pred.get('vanna_resistance_2')
        gex_support = pred.get('gex_support')
        gex_resistance = pred.get('gex_resistance')
        gex_flip = pred.get('gex_zero_level')
        gex_regime = pred.get('gex_regime', 'unknown')

        # Calculate pressure score (-100 to +100)
        pressure_score = 0
        pressure_factors = []

        # GEX regime factor
        if gex_regime == 'positive':
            pressure_score += 10
            pressure_factors.append("GEX+ (mean reversion)")
        elif gex_regime == 'negative':
            pressure_score -= 10
            pressure_factors.append("GEX- (momentum)")

        # Price relative to GEX flip
        if gex_flip and current:
            if current > gex_flip:
                pressure_score += 15
                pressure_factors.append(f"Above GEX Flip (${gex_flip:.0f})")
            else:
                pressure_score -= 15
                pressure_factors.append(f"Below GEX Flip (${gex_flip:.0f})")

        # Distance to support levels (closer = more bullish)
        support_levels = [l for l in [vanna_s1, vanna_s2, gex_support] if l and abs(l - current) / current <= 0.05]
        resistance_levels = [l for l in [vanna_r1, vanna_r2, gex_resistance] if l and abs(l - current) / current <= 0.05]

        if support_levels:
            nearest_support = max(support_levels)
            support_dist = (current - nearest_support) / current * 100
            if support_dist < 0.5:  # Very close to support
                pressure_score += 30
                pressure_factors.append(f"Near support (${nearest_support:.0f})")
            elif support_dist < 1.0:
                pressure_score += 20
                pressure_factors.append(f"Approaching support")

        if resistance_levels:
            nearest_resistance = min(resistance_levels)
            resistance_dist = (nearest_resistance - current) / current * 100
            if resistance_dist < 0.5:  # Very close to resistance
                pressure_score -= 30
                pressure_factors.append(f"Near resistance (${nearest_resistance:.0f})")
            elif resistance_dist < 1.0:
                pressure_score -= 20
                pressure_factors.append(f"Approaching resistance")

        # Clamp score
        pressure_score = max(-100, min(100, pressure_score))

        # Create gauge
        col1, col2 = st.columns([2, 1])

        with col1:
            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pressure_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Dealer Hedge Pressure", 'font': {'size': 16}},
                number={'suffix': "", 'font': {'size': 24}},
                gauge={
                    'axis': {'range': [-100, 100], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#2196F3"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [-100, -50], 'color': '#FF5252', 'name': 'Strong Sell'},
                        {'range': [-50, -20], 'color': '#FF8A80', 'name': 'Sell'},
                        {'range': [-20, 20], 'color': '#FFE082', 'name': 'Neutral'},
                        {'range': [20, 50], 'color': '#A5D6A7', 'name': 'Buy'},
                        {'range': [50, 100], 'color': '#00E676', 'name': 'Strong Buy'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.75,
                        'value': pressure_score
                    }
                }
            ))

            fig_gauge.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': "white"}
            )

            st.plotly_chart(fig_gauge, use_container_width=True)

        with col2:
            # Pressure interpretation
            if pressure_score >= 50:
                st.success("**STRONG BUY PRESSURE**")
                st.caption("Dealers likely buying")
            elif pressure_score >= 20:
                st.success("**BUY PRESSURE**")
                st.caption("Mild bullish bias")
            elif pressure_score <= -50:
                st.error("**STRONG SELL PRESSURE**")
                st.caption("Dealers likely selling")
            elif pressure_score <= -20:
                st.error("**SELL PRESSURE**")
                st.caption("Mild bearish bias")
            else:
                st.warning("**NEUTRAL**")
                st.caption("No clear pressure")

            # Show factors
            st.markdown("**Factors:**")
            for factor in pressure_factors[:4]:
                st.caption(f"• {factor}")

        st.markdown("---")

        # ========== MULTI-PANEL OPTIONS FLOW CHART ==========
        st.subheader("📊 Options Flow Analysis")

        # Check market status
        now = datetime.now(EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        is_market_hours = market_open <= now <= market_close and now.weekday() < 5

        if not is_market_hours:
            st.info("📊 **Market Closed** - Showing key levels only. Candlesticks will appear during market hours (9:30 AM - 4:00 PM EST, Mon-Fri)")

        # Fetch recent price data for candlesticks
        price_df = None
        try:
            from data_collector import TradierDataCollector
            collector = TradierDataCollector(api_token)

            # Get last 2 days of 5-min bars for recent context (EST)
            now = datetime.now(EST)
            start = (now - timedelta(days=2)).strftime('%Y-%m-%d %H:%M')
            end = now.strftime('%Y-%m-%d %H:%M')

            price_df = collector.get_intraday_quotes(
                symbol=symbol,
                start_time=start,
                end_time=end,
                interval='5min'
            )

            # Take last 40 bars for clean display
            if not price_df.empty and len(price_df) > 0:
                price_df = price_df.tail(40).copy()

        except Exception as e:
            st.info(f"Using levels-only view: {str(e)}")

        # Create multi-panel chart
        try:
            fig = create_options_flow_chart(pred, price_df, symbol)
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Chart creation failed: {e}")
            import traceback
            st.code(traceback.format_exc())
            st.caption("⚠️ If this persists, try refreshing the page or restarting the app")

        # Chart legend with clear descriptions
        st.markdown("""
        **📖 Chart Guide:**

        **Panel 1 - Price Levels:**
        - 🔵 **CURRENT** = Entry price
        - 🌊 **GAMMA FLIP** = Major pivot (mean reversion above, momentum below)
        - 🟢 **GEX SUPPORT/RESISTANCE** = S/R zones where dealers hedge
        - 🟣 **VANNA WALLS** = Pressure zones from IV exposure
        - 🎯 **PUT/CALL WALLS** = OI magnets and barriers
        - 🟡 **VWAP** = Dynamic balance line
        - **Background Color**: Green = Positive GEX (fade extremes), Red = Negative GEX (follow trends)

        **Panel 2 - IV & Vanna:**
        - **IV**: Implied volatility level and percentile
        - **Net Vanna**: Dealer bias from IV exposure (positive = support, negative = resistance)
        - **Vanna×IV**: Trend strength from options flow

        **Panel 3 - Dealer Flow:**
        - **Charm Pressure**: Time decay hedging flows
        - **GEX Pressure**: Reaction strength indicator
        - **Dealer Flow Score**: Combined flow metric (-100 to +100)
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

                # Fetch intraday data (EST)
                collector = TradierDataCollector(api_token)
                now = datetime.now(EST)

                # Use today's date for market hours (EST)
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

                    # Calculate proper y-axis range for intraday chart
                    intraday_low = intraday_df['low'].min()
                    intraday_high = intraday_df['high'].max()
                    intraday_range = intraday_high - intraday_low
                    padding = max(intraday_range * 0.1, intraday_low * 0.005)

                    # Layout with dual y-axis and proper scaling
                    fig_intraday.update_layout(
                        title=f"{symbol} Intraday ({st.session_state.get('interval', '5min')} bars)",
                        yaxis=dict(
                            title='Price ($)',
                            range=[intraday_low - padding, intraday_high + padding],
                            side='left'
                        ),
                        yaxis2=dict(
                            title='Volume',
                            overlaying='y',
                            side='right',
                            showgrid=False
                        ),
                        xaxis=dict(title='Time'),
                        height=500,
                        hovermode='x unified',
                        legend=dict(x=0, y=1.05, orientation='h'),
                        margin=dict(l=60, r=60, t=50, b=50)
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
                    # Check if market is closed (EST)
                    now = datetime.now(EST)
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
            - **🌟 Priority: 10AM-12PM EST**
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
