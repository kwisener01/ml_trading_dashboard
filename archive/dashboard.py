import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from predictor import TradingPredictor
import os

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
    
    api_token = st.text_input("Tradier API Token", type="password", 
                               value=os.environ.get('TRADIER_API_TOKEN', ''))
    
    symbol = st.selectbox("Select Symbol", ['SPY', 'QQQ', 'IWM', 'DIA'])
    
    trade_quality_threshold = st.slider("Trade Quality Threshold", 0, 100, 60)
    
    if st.button("🔄 Generate Prediction", type="primary"):
        st.session_state['generate_prediction'] = True
    
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
        delta_color = "normal" if quality >= trade_quality_threshold else "inverse"
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
            if quality < 40:
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
