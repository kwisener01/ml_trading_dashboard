# Version: 2025-11-26-v3-gex-panels-fixed
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
def create_options_flow_chart(pred, price_df, symbol, in_charm_session=False, in_priority_session=False):
    """
    Create 3-panel chart with:
    - Panel 1: Price with options flow levels
    - Panel 2: IV & Vanna indicators
    - Panel 3: Dealer flow indicators

    Args:
        in_charm_session: If True, highlights charm in Panel 3 (end-of-day 3-4PM)
        in_priority_session: If True, highlights GEX in Panel 3 (NY morning 10AM-12PM)
    """
    # Create subplot with 3 rows
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15,  # Spacing for mobile visibility
        row_heights=[0.45, 0.275, 0.275],  # Panel 1: 45%, Panels 2&3: 27.5% each for better visibility
        subplot_titles=(
            f"<b>{symbol} - Options Flow Analysis</b>",
            "<b>Panel 2: IV & Vanna Indicators</b>",
            "<b>Panel 3: Dealer Flow Indicators</b>"
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
        """Check if level is valid and within pct% of current price"""
        return level is not None and abs(level - current) / current <= pct

    # Add Gamma Flip (Major Pivot) - always show if available, no distance restriction
    if gex_flip is not None:
        fig.add_hline(
            y=gex_flip,
            line_dash="solid",
            line_color="#00BCD4",
            line_width=3,
            annotation_text=f"GAMMA FLIP: ${gex_flip:.2f} (Major Pivot)",
            annotation_position="left",  # Changed from right to left for mobile visibility
            annotation=dict(font=dict(size=10, color="white"), bgcolor="#00838F"),
            row=1, col=1
        )

    # Add GEX Walls (S/R Zones) - always show if available, no distance restriction
    if gex_support is not None:
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

    if gex_resistance is not None:
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
            annotation_position="left",  # Changed from right to left for mobile visibility
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
            annotation_position="left",  # Changed from right to left for mobile visibility
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
            line_color="#FFD700",  # Gold color
            line_width=2,
            opacity=0.8,
            annotation_text=f"VWAP: ${vwap:.2f}",
            annotation_position="left",  # Changed from right to left for mobile visibility
            annotation=dict(font=dict(size=10, color="white"), bgcolor="#FF8F00"),  # Dark orange bg
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
    # Show REAL current values only (no simulated historical lines)

    # Get current real values
    iv_current = pred.get('iv', 0.2) * 100
    vanna_s1_str = pred.get('vanna_support_1_strength', 0) or 0
    vanna_r1_str = pred.get('vanna_resistance_1_strength', 0) or 0
    net_vanna_current = (vanna_s1_str + vanna_r1_str) * 100
    vanna_iv_current = pred.get('vanna_iv_trend', 0)

    # Use price data x-axis if available, otherwise use current time
    if has_price_data and len(price_df) > 1:
        panel_x = price_df.index if 'time' not in price_df.columns else price_df['time']
    else:
        panel_x = pd.Series([datetime.now(EST)])

    # Add invisible trace to Panel 2 to ensure it renders
    fig.add_trace(go.Scatter(
        x=panel_x,
        y=[iv_current] * len(panel_x),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),  # Invisible
        showlegend=False,
        hoverinfo='skip'
    ), row=2, col=1)

    # Add horizontal reference lines for current values
    # IV as horizontal line spanning the chart
    fig.add_hline(
        y=iv_current,
        line=dict(color='#FF6B6B', width=2, dash='solid'),
        annotation_text=f"IV: {iv_current:.1f}%",
        annotation_position="left",  # Changed from right to left for mobile visibility
        annotation=dict(font=dict(size=10, color="white"), bgcolor="#FF6B6B"),
        row=2, col=1
    )

    # Net Vanna as horizontal line
    fig.add_hline(
        y=net_vanna_current,
        line=dict(color='#4ECDC4', width=2, dash='solid'),
        annotation_text=f"Net Vanna: {net_vanna_current:.1f}",
        annotation_position="left",  # Changed from right to left for mobile visibility
        annotation=dict(font=dict(size=10, color="white"), bgcolor="#4ECDC4"),
        row=2, col=1
    )

    # Vanna×IV as horizontal line
    fig.add_hline(
        y=vanna_iv_current,
        line=dict(color='#95E1D3', width=2, dash='dot'),
        annotation_text=f"Vanna×IV: {vanna_iv_current:.1f}",
        annotation_position="left",  # Changed from right to left for mobile visibility
        annotation=dict(font=dict(size=10, color="white"), bgcolor="#95E1D3"),
        row=2, col=1
    )

    # ========== PANEL 3: DEALER FLOW ==========
    # Show REAL current values only (no simulated data)

    # Get real current values
    charm_current = pred.get('charm_pressure', 0)
    dealer_score_current = pred.get('dealer_flow_score', 0)
    is_charm_bullish = charm_current > 0

    # Add invisible trace to Panel 3 to ensure it renders
    fig.add_trace(go.Scatter(
        x=panel_x,
        y=[dealer_score_current] * len(panel_x),
        mode='lines',
        line=dict(color='rgba(0,0,0,0)', width=0),  # Invisible
        showlegend=False,
        hoverinfo='skip'
    ), row=3, col=1)

    # Charm (REAL value from Black-Scholes calculation)
    charm_color = '#4CAF50' if is_charm_bullish else '#F44336'
    charm_label = 'Charm Pressure (REAL)'
    if in_charm_session:
        charm_label += ' ⚡ (EOD ACTIVE)'
        charm_color = '#FFD700' if is_charm_bullish else '#FF6B00'

    fig.add_hline(
        y=charm_current,
        line=dict(color=charm_color, width=3 if in_charm_session else 2, dash='solid'),
        annotation_text=f"{charm_label}: {charm_current:.1f}",
        annotation_position="left",  # Changed from right to left for mobile visibility
        annotation=dict(font=dict(size=10, color="white"), bgcolor=charm_color),
        row=3, col=1
    )

    # Add charm session indicator if active
    if in_charm_session:
        fig.add_annotation(
            text="⚡ END-OF-DAY: High Charm Pressure ⚡",
            xref="x3", yref="y3",
            x=panel_x.iloc[len(panel_x)//2] if len(panel_x) > 1 else panel_x.iloc[0],
            y=max(abs(charm_current), 50) * 1.1,
            showarrow=False,
            font=dict(size=10, color="gold", family="Arial Black"),
            bgcolor="rgba(30, 30, 30, 0.8)",
            bordercolor="gold",
            borderwidth=1,
            borderpad=4,
            row=3, col=1
        )

    # GEX Pressure - ONLY show if we have real GEX data
    has_real_gex = gex_flip is not None or gex_support is not None or gex_resistance is not None
    if has_real_gex and gex_regime:
        # Calculate real GEX pressure based on actual regime
        gex_pressure_current = 50 if gex_regime == 'positive' else -50 if gex_regime == 'negative' else 0
        gex_label = 'GEX Pressure (REAL)'
        gex_width = 2
        if in_priority_session:
            gex_label += ' 🌟 (NY MORNING)'
            gex_width = 4

        fig.add_hline(
            y=gex_pressure_current,
            line=dict(color='#4CAF50' if gex_pressure_current > 0 else '#F44336', width=gex_width, dash='solid'),
            annotation_text=f"{gex_label}: {gex_pressure_current:.0f}",
            annotation_position="left",
            annotation=dict(font=dict(size=10, color="white"), bgcolor='#4CAF50' if gex_pressure_current > 0 else '#F44336'),
            row=3, col=1
        )

        # Add priority session indicator for GEX if active
        if in_priority_session:
            fig.add_annotation(
                text="🌟 NY MORNING: High GEX Impact 🌟",
                xref="x3", yref="y3",
                x=panel_x.iloc[len(panel_x)//2] if len(panel_x) > 1 else panel_x.iloc[0],
                y=-80,
                showarrow=False,
                font=dict(size=10, color="lime", family="Arial Black"),
                bgcolor="rgba(30, 30, 30, 0.8)",
                bordercolor="lime",
                borderwidth=1,
                borderpad=4,
                row=3, col=1
            )
    else:
        # Show "No GEX Data" message in Panel 3
        fig.add_annotation(
            text="⚠️ No Real GEX Data Available",
            xref="x3", yref="y3",
            x=panel_x.iloc[len(panel_x)//2] if len(panel_x) > 1 else panel_x.iloc[0],
            y=0,
            showarrow=False,
            font=dict(size=12, color="orange"),
            bgcolor="rgba(30, 30, 30, 0.8)",
            bordercolor="orange",
            borderwidth=1,
            borderpad=4,
            row=3, col=1
        )

    # Dealer Flow Score (REAL value - combination of all flows)
    fig.add_hline(
        y=dealer_score_current,
        line=dict(color='#FFC107', width=3, dash='solid'),
        annotation_text=f"Dealer Flow: {dealer_score_current:.0f}",
        annotation_position="right",
        annotation=dict(font=dict(size=10, color="white"), bgcolor='#FFC107'),
        row=3, col=1
    )

    # Add zero reference line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, opacity=0.5, row=3, col=1)

    # Update layout
    fig.update_layout(
        height=1800,  # Significantly increased height for mobile visibility
        autosize=True,  # Enable responsive sizing
        showlegend=True,
        legend=dict(
            orientation="h",  # Horizontal legend for mobile compatibility
            yanchor="bottom",
            y=-0.08,  # Position below chart (adjusted for taller chart)
            xanchor="center",
            x=0.5,
            bgcolor="rgba(30, 30, 30, 0.9)",
            bordercolor="rgba(255, 255, 255, 0.3)",
            borderwidth=1,
            font=dict(size=10)
        ),
        hovermode='x unified',
        template='plotly_dark',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(30, 30, 30, 1)',
        margin=dict(l=60, r=40, t=100, b=100),  # Reduced margins for mobile, bottom for legend
        # Mobile-friendly defaults
        dragmode='pan',  # Better for touch devices
        modebar=dict(
            orientation='v',
            bgcolor='rgba(30, 30, 30, 0.8)'
        )
    )

    # Update y-axes labels with better styling
    fig.update_yaxes(
        title_text="<b>Price ($)</b>",
        title_font=dict(size=14),
        row=1, col=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=False
    )
    fig.update_yaxes(
        title_text="<b>IV / Vanna</b>",
        title_font=dict(size=14),
        row=2, col=1,
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.5)',
        zerolinewidth=1
    )
    fig.update_yaxes(
        title_text="<b>Dealer Flow Score</b>",
        title_font=dict(size=14),
        row=3, col=1,
        range=[-100, 100],
        gridcolor='rgba(128, 128, 128, 0.2)',
        zeroline=True,
        zerolinecolor='rgba(128, 128, 128, 0.5)',
        zerolinewidth=2
    )

    # Update x-axes
    fig.update_xaxes(title_text="<b>Time</b>", title_font=dict(size=14), row=3, col=1)
    fig.update_xaxes(rangeslider_visible=False)

    # Style subplot titles to be more prominent
    fig.update_annotations(font=dict(size=16, color='white'))

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

# Add viewport meta tag for mobile responsiveness
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=5.0, user-scalable=yes">
""", unsafe_allow_html=True)

# Custom CSS with mobile responsiveness
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

    /* Global mobile improvements */
    * {
        -webkit-tap-highlight-color: rgba(0,0,0,0);
        -webkit-touch-callout: none;
    }

    html {
        scroll-behavior: smooth;
        -webkit-text-size-adjust: 100%;
    }

    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        /* Reduce title size on mobile */
        h1 {
            font-size: 1.5rem !important;
        }
        h2 {
            font-size: 1.3rem !important;
        }
        h3 {
            font-size: 1.1rem !important;
        }

        /* Make trade signals more compact */
        .trade-signal {
            font-size: 18px !important;
            padding: 10px !important;
        }

        /* Adjust metric boxes for mobile */
        .metric-box {
            padding: 12px !important;
            margin: 5px 0 !important;
        }

        /* Make buttons full width on mobile */
        .stButton button {
            width: 100% !important;
        }

        /* Reduce padding in main container */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            max-width: 100% !important;
        }

        /* Improve touch targets */
        button, a, input, select {
            min-height: 44px !important;
        }

        /* Reduce sidebar width on mobile */
        [data-testid="stSidebar"] {
            min-width: 250px !important;
        }

        /* Make charts responsive */
        .js-plotly-plot {
            width: 100% !important;
        }

        /* Adjust plotly chart margins for mobile */
        .js-plotly-plot .plotly .main-svg {
            max-width: 100% !important;
        }

        /* Improve plotly modebar on mobile */
        .modebar {
            top: 0 !important;
            right: 0 !important;
        }

        /* Adjust font sizes in sidebar */
        .sidebar .markdown-text-container {
            font-size: 0.9rem !important;
        }

        /* Make expanders more touch-friendly */
        .streamlit-expanderHeader {
            font-size: 1rem !important;
            padding: 12px !important;
        }
    }

    /* Extra small devices (phones in portrait, less than 576px) */
    @media (max-width: 576px) {
        h1 {
            font-size: 1.3rem !important;
        }
        h2 {
            font-size: 1.1rem !important;
        }
        h3 {
            font-size: 1rem !important;
        }

        .trade-signal {
            font-size: 16px !important;
            padding: 8px !important;
        }

        /* Even more compact metrics */
        [data-testid="metric-container"] {
            padding: 5px !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.2rem !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 0.8rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🤖 ML Trading Dashboard")
st.caption("🟢 Version: 2025-11-26-v3 (GEX + Panels Fixed)")
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
        index=1,  # Default to Day Trading (Intraday)
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

            st.caption(f"Refreshing every {refresh_seconds // 60} min")

            # Initialize last refresh time if not exists
            if 'last_refresh_time' not in st.session_state:
                st.session_state['last_refresh_time'] = time.time()
                st.session_state['generate_prediction'] = True
                st.session_state['trading_mode'] = trading_mode
                st.session_state['interval'] = interval

            # Check if it's time to refresh
            current_time = time.time()
            time_since_refresh = current_time - st.session_state['last_refresh_time']

            if time_since_refresh >= refresh_seconds:
                # Time to refresh!
                st.session_state['last_refresh_time'] = current_time
                st.session_state['generate_prediction'] = True
                st.session_state['trading_mode'] = trading_mode
                st.session_state['interval'] = interval
                st.rerun()
            else:
                # Show countdown
                time_remaining = int(refresh_seconds - time_since_refresh)
                mins_remaining = time_remaining // 60
                secs_remaining = time_remaining % 60
                st.info(f"🔄 Next refresh in: {mins_remaining}:{secs_remaining:02d}")

                # Use st.empty() to trigger a rerun after a short delay
                time.sleep(1)
                st.rerun()

        # Show market hours status (in EST)
        now = datetime.now(EST)
        market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
        optimal_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        optimal_end = now.replace(hour=15, minute=0, second=0, microsecond=0)

        # NY Morning Priority Session (10:00 AM - 12:00 PM EST)
        priority_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        priority_end = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # End-of-Day Charm Session (3:00 PM - 4:00 PM EST)
        charm_start = now.replace(hour=15, minute=0, second=0, microsecond=0)
        charm_end = now.replace(hour=16, minute=0, second=0, microsecond=0)

        # Display current EST time
        st.caption(f"🕐 {now.strftime('%I:%M %p EST')}")

        # Check if in special trading sessions
        in_priority_session = priority_start <= now <= priority_end and now.weekday() < 5
        in_charm_session = charm_start <= now <= charm_end and now.weekday() < 5

        if now < market_open:
            st.warning(f"⏸️ Pre-market ({(market_open - now).seconds // 60} min to open)")
        elif now > market_close:
            st.info("🔔 Market closed")
        elif now < optimal_start:
            st.warning("⚠️ Avoid zone (first 30 min)")
        elif in_charm_session:
            st.warning("⏰ **END-OF-DAY CHARM** ⏰")
            st.caption("High charm pressure (3-4PM)")
        elif in_priority_session:
            st.success("🌟 **NY MORNING PRIORITY** 🌟")
            st.caption("Prime liquidity window (10AM-12PM)")
        else:
            st.success("✅ Optimal trading hours")

        # Session Info Boxes
        st.markdown("---")

        # Priority session or Charm session - show whichever is active/relevant
        if in_charm_session:
            st.markdown("### ⏰ End-of-Day Charm")
            time_remaining = (charm_end - now).seconds
            mins_remaining = time_remaining // 60
            st.warning(f"""
            **ACTIVE NOW**

            ⏱️ {mins_remaining} minutes to close

            ⚡ High charm pressure (time decay)
            📉 Dealers hedge delta decay
            🎯 Watch for pin to strikes or breakouts
            ⚠️ Elevated volatility possible
            """)
        elif in_priority_session:
            st.markdown("### 🌟 NY Morning Priority")
            time_remaining = (priority_end - now).seconds
            mins_remaining = time_remaining // 60
            st.info(f"""
            **ACTIVE NOW**

            ⏱️ {mins_remaining} minutes remaining

            📈 Highest liquidity & volatility
            🎯 Best setups typically occur now
            ⚡ Focus on quality entries
            """)
        elif now < priority_start and now >= market_open and not in_charm_session:
            st.markdown("### 🌟 Next Session")
            time_until = (priority_start - now).seconds
            mins_until = time_until // 60
            st.caption(f"""
            **Priority starts in {mins_until} min**

            ⏰ 10AM-12PM EST - Prime trading time
            """)
        elif now >= priority_end and now < charm_start:
            st.markdown("### ⏰ Next Session")
            time_until = (charm_start - now).seconds
            mins_until = time_until // 60
            st.caption(f"""
            **Charm session in {mins_until} min**

            ⏰ 3PM-4PM EST - End-of-day hedging
            """)
        else:
            st.markdown("### 📅 Trading Sessions")
            st.caption("""
            **🌟 NY Morning Priority**
            10:00 AM - 12:00 PM EST
            Prime liquidity & setups

            **⏰ End-of-Day Charm**
            3:00 PM - 4:00 PM EST
            Time decay hedging flows
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

    with st.expander("🎯 How to Trade (Decision Guide)", expanded=True):
        if os.path.exists("trading_decision_guide.png"):
            st.image("trading_decision_guide.png", use_container_width=True)
        st.caption("""
        **Follow these 3 steps:**
        1. Check background color (GEX regime)
        2. Check price vs Gamma Flip
        3. Check Dealer Flow score

        **Then execute the matching setup!**
        """)

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
            if os.path.exists("bullish_setup_simple.png"):
                st.image("bullish_setup_simple.png", use_container_width=True)

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
            if os.path.exists("bearish_setup_simple.png"):
                st.image("bearish_setup_simple.png", use_container_width=True)

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
            if os.path.exists("range-bound_setup_simple.png"):
                st.image("range-bound_setup_simple.png", use_container_width=True)

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
        # Refresh button at top
        col_refresh1, col_refresh2 = st.columns([6, 1])
        with col_refresh2:
            if st.button("🔄 Refresh", type="secondary", key="refresh_prediction"):
                st.session_state['generate_prediction'] = True
                st.rerun()

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
        if gex_flip is not None and current is not None:
            if current > gex_flip:
                pressure_score += 15
                pressure_factors.append(f"Above GEX Flip (${gex_flip:.0f})")
            else:
                pressure_score -= 15
                pressure_factors.append(f"Below GEX Flip (${gex_flip:.0f})")

        # Distance to support levels (closer = more bullish)
        support_levels = [l for l in [vanna_s1, vanna_s2, gex_support] if l is not None and abs(l - current) / current <= 0.05]
        resistance_levels = [l for l in [vanna_r1, vanna_r2, gex_resistance] if l is not None and abs(l - current) / current <= 0.05]

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

        # Check if in end-of-day charm session (3-4 PM EST)
        charm_start = now.replace(hour=15, minute=0, second=0, microsecond=0)
        charm_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        in_charm_session = charm_start <= now <= charm_end and now.weekday() < 5

        # Check if in NY morning priority session (10 AM - 12 PM EST)
        priority_start = now.replace(hour=10, minute=0, second=0, microsecond=0)
        priority_end = now.replace(hour=12, minute=0, second=0, microsecond=0)
        in_priority_session = priority_start <= now <= priority_end and now.weekday() < 5

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
            fig = create_options_flow_chart(pred, price_df, symbol,
                                           in_charm_session=in_charm_session,
                                           in_priority_session=in_priority_session)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)

                # GEX Level Status Display
                gex_flip = pred.get('gex_zero_level')
                gex_support = pred.get('gex_support')
                gex_resistance = pred.get('gex_resistance')
                gex_error = pred.get('gex_error')

                if gex_flip is None and gex_support is None and gex_resistance is None:
                    if gex_error:
                        st.warning(f"⚠️ **GEX/Gamma levels not available** - Error: {gex_error}")
                        with st.expander("🔍 GEX Troubleshooting"):
                            st.write("**Why GEX might fail:**")
                            st.write("- Options chain data not available from API")
                            st.write("- Market closed (options data only available during market hours)")
                            st.write("- API rate limit or connection issue")
                            st.write("- Symbol not supported for options")
                            st.write("")
                            st.write("**Note:** Vanna levels are calculated mathematically and don't require live options data, which is why they still work.")
                    else:
                        st.warning("⚠️ **GEX/Gamma levels not available** - This could be due to insufficient options data or market being closed. Vanna levels should still be visible.")

                        # Debug info
                        with st.expander("🔧 Debug Info"):
                            st.write(f"**gex_error value:** `{repr(gex_error)}`")
                            st.write(f"**All GEX-related values:**")
                            st.write(f"- gex_flip: {gex_flip}")
                            st.write(f"- gex_support: {gex_support}")
                            st.write(f"- gex_resistance: {gex_resistance}")
                            st.write(f"- gex_regime: {pred.get('gex_regime')}")
                            st.write(f"- gex_current: {pred.get('gex_current')}")
                            st.write("")
                            st.write("**This debug info will help diagnose the issue.**")
                            st.write("The GEX calculator is likely returning empty results without raising an exception.")
                else:
                    # Format GEX values with proper None handling
                    flip_str = f"${gex_flip:.2f}" if gex_flip is not None else "N/A"
                    support_str = f"${gex_support:.0f}" if gex_support is not None else "N/A"
                    resistance_str = f"${gex_resistance:.0f}" if gex_resistance is not None else "N/A"
                    st.success(f"✅ **GEX Levels Active:** Gamma Flip: {flip_str} | Support: {support_str} | Resistance: {resistance_str}")
            else:
                st.error("❌ Chart function returned None")
                st.caption("Debug info:")
                st.write(f"- price_df empty: {price_df is None or price_df.empty if price_df is not None else 'price_df is None'}")
                st.write(f"- symbol: {symbol}")
                st.write(f"- pred keys: {list(pred.keys())[:10]}")
        except Exception as e:
            st.error(f"❌ Chart creation failed: {e}")
            st.caption("Debug info:")
            st.write(f"- Error type: {type(e).__name__}")
            st.write(f"- price_df: {'None' if price_df is None else f'{len(price_df)} rows' if hasattr(price_df, '__len__') else 'exists'}")
            import traceback
            with st.expander("Full Traceback"):
                st.code(traceback.format_exc())

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

                    # Add GEX regime background shading
                    gex_regime = pred.get('gex_regime', 'unknown')
                    if gex_regime == 'positive':
                        # Green background for positive GEX (mean reversion)
                        fig_intraday.add_vrect(
                            x0=intraday_df['time'].iloc[0],
                            x1=intraday_df['time'].iloc[-1],
                            fillcolor="rgba(0, 255, 0, 0.05)",
                            layer="below",
                            line_width=0
                        )
                    elif gex_regime == 'negative':
                        # Red background for negative GEX (momentum)
                        fig_intraday.add_vrect(
                            x0=intraday_df['time'].iloc[0],
                            x1=intraday_df['time'].iloc[-1],
                            fillcolor="rgba(255, 0, 0, 0.05)",
                            layer="below",
                            line_width=0
                        )

                    # Add volume bars
                    fig_intraday.add_trace(go.Bar(
                        x=intraday_df['time'],
                        y=intraday_df['volume'],
                        name='Volume',
                        yaxis='y2',
                        marker_color='rgba(100, 100, 255, 0.3)'
                    ))

                    # Add key levels for intraday trading

                    # Gamma Flip (MOST IMPORTANT - Major pivot)
                    gex_flip = pred.get('gex_zero_level')
                    if gex_flip is not None:
                        fig_intraday.add_hline(
                            y=gex_flip,
                            line_dash="solid",
                            line_color="#00BCD4",
                            line_width=3,
                            annotation_text=f"GAMMA FLIP: ${gex_flip:.2f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=10, color="white"), bgcolor="#00838F")
                        )

                    # GEX Support (Strong floor)
                    gex_support = pred.get('gex_support')
                    if gex_support is not None:
                        fig_intraday.add_hline(
                            y=gex_support,
                            line_dash="dash",
                            line_color="#76FF03",
                            line_width=2,
                            annotation_text=f"GEX Support: ${gex_support:.0f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=9, color="white"), bgcolor="#33691E")
                        )

                    # GEX Resistance (Strong ceiling)
                    gex_resistance = pred.get('gex_resistance')
                    if gex_resistance is not None:
                        fig_intraday.add_hline(
                            y=gex_resistance,
                            line_dash="dash",
                            line_color="#FF1744",
                            line_width=2,
                            annotation_text=f"GEX Resistance: ${gex_resistance:.0f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=9, color="white"), bgcolor="#B71C1C")
                        )

                    # Vanna Support (Bounce zone)
                    vanna_support = pred.get('vanna_support_1')
                    if vanna_support is not None:
                        fig_intraday.add_hline(
                            y=vanna_support,
                            line_dash="dot",
                            line_color="#00E676",
                            line_width=2,
                            annotation_text=f"Vanna Support: ${vanna_support:.2f}",
                            annotation_position="right"
                        )

                    # Vanna Resistance (Rejection zone)
                    vanna_resistance = pred.get('vanna_resistance_1')
                    if vanna_resistance is not None:
                        fig_intraday.add_hline(
                            y=vanna_resistance,
                            line_dash="dot",
                            line_color="#FF5252",
                            line_width=2,
                            annotation_text=f"Vanna Resistance: ${vanna_resistance:.2f}",
                            annotation_position="right"
                        )

                    # Current Price
                    current_price = pred.get('current_price')
                    if current_price is not None:
                        fig_intraday.add_hline(
                            y=current_price,
                            line_dash="solid",
                            line_color="#2196F3",
                            line_width=2,
                            annotation_text=f"Current: ${current_price:.2f}",
                            annotation_position="left",
                            annotation=dict(font=dict(size=10, color="white"), bgcolor="#2196F3")
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
                        autosize=True,  # Enable responsive sizing
                        hovermode='x unified',
                        legend=dict(x=0, y=1.05, orientation='h'),
                        margin=dict(l=60, r=60, t=50, b=50),
                        # Mobile-friendly defaults
                        dragmode='pan',  # Better for touch devices
                        modebar=dict(
                            orientation='v',
                            bgcolor='rgba(30, 30, 30, 0.8)'
                        )
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

        # Show level status (always visible after prediction)
        st.markdown("---")
        st.markdown("### 🔍 Level Check")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**GEX Levels:**")
            gex_flip_val = pred.get('gex_zero_level')
            gex_sup_val = pred.get('gex_support')
            gex_res_val = pred.get('gex_resistance')

            if gex_flip_val is not None:
                st.success(f"✅ Gamma Flip: ${gex_flip_val:.2f}")
            else:
                st.error(f"❌ Gamma Flip: None")

            if gex_sup_val is not None:
                st.success(f"✅ GEX Support: ${gex_sup_val:.2f}")
            else:
                st.error(f"❌ GEX Support: None")

            if gex_res_val is not None:
                st.success(f"✅ GEX Resistance: ${gex_res_val:.2f}")
            else:
                st.error(f"❌ GEX Resistance: None")

        with col2:
            st.write("**Vanna Levels:**")
            vanna_sup = pred.get('vanna_support_1')
            vanna_res = pred.get('vanna_resistance_1')

            if vanna_sup is not None:
                st.success(f"✅ Vanna Support: ${vanna_sup:.2f}")
            else:
                st.error(f"❌ Vanna Support: None")

            if vanna_res is not None:
                st.success(f"✅ Vanna Resistance: ${vanna_res:.2f}")
            else:
                st.error(f"❌ Vanna Resistance: None")

            st.write(f"**GEX Regime:** {pred.get('gex_regime', 'unknown')}")

        # Add debug info for diagnosing GEX issues
        st.markdown("---")
        st.markdown("### 🔧 GEX Debug Info")

        gex_error_val = pred.get('gex_error')
        st.write(f"**Error captured:** `{repr(gex_error_val)}`")
        st.write(f"**GEX Regime:** `{repr(pred.get('gex_regime'))}`")
        st.write(f"**GEX Current:** `{repr(pred.get('gex_current'))}`")

        if gex_error_val:
            st.error(f"Error message: {gex_error_val}")
        elif pred.get('gex_regime') is None:
            st.warning("GEX regime is None - calculator returned empty results without error")
        else:
            st.info(f"GEX regime detected: {pred.get('gex_regime')}, but no strike levels")

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
