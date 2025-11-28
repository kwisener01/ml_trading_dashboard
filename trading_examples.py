"""
Trading Examples - How to Use the 3-Panel Options Flow Chart

This script generates annotated example charts showing how to interpret
the three panels for different trading scenarios.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import pandas as pd


def create_example_chart(scenario_name, scenario_data):
    """Create a 3-panel example chart with annotations"""

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.65, 0.175, 0.175],
        subplot_titles=(
            f"{scenario_name} - Panel 1: Price & Options Flow Levels",
            "Panel 2: IV & Vanna Indicators",
            "Panel 3: Dealer Flow Indicators"
        ),
        specs=[[{"secondary_y": False}],
               [{"secondary_y": False}],
               [{"secondary_y": False}]]
    )

    # Generate example price data
    times = pd.date_range(start=datetime.now() - timedelta(hours=2),
                          periods=50, freq='5min')

    # Create price movement based on scenario
    base_price = scenario_data['current_price']
    price_trend = scenario_data['price_trend']
    prices = base_price + np.cumsum(np.random.randn(50) * 0.5 + price_trend)

    # Create OHLC data
    ohlc_data = pd.DataFrame({
        'time': times,
        'open': prices + np.random.randn(50) * 0.2,
        'high': prices + abs(np.random.randn(50)) * 0.3,
        'low': prices - abs(np.random.randn(50)) * 0.3,
        'close': prices
    })

    # ========== PANEL 1: PRICE CHART ==========

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=ohlc_data['time'],
        open=ohlc_data['open'],
        high=ohlc_data['high'],
        low=ohlc_data['low'],
        close=ohlc_data['close'],
        name='Price',
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        showlegend=False
    ), row=1, col=1)

    # Add regime shading
    regime_color = "rgba(0, 255, 0, 0.05)" if scenario_data['gex_regime'] == 'positive' else "rgba(255, 0, 0, 0.05)"
    fig.add_vrect(
        x0=times[0], x1=times[-1],
        fillcolor=regime_color,
        layer="below",
        line_width=0,
        row=1, col=1
    )

    # Add Gamma Flip
    fig.add_hline(
        y=scenario_data['gamma_flip'],
        line_dash="solid",
        line_color="#00BCD4",
        line_width=3,
        annotation_text=f"GAMMA FLIP: ${scenario_data['gamma_flip']:.2f}",
        annotation_position="right",
        annotation=dict(font=dict(size=10, color="white"), bgcolor="#00838F"),
        row=1, col=1
    )

    # Add GEX levels
    if 'gex_support' in scenario_data:
        fig.add_hline(
            y=scenario_data['gex_support'],
            line_dash="dash",
            line_color="#76FF03",
            line_width=2,
            annotation_text=f"GEX SUPPORT: ${scenario_data['gex_support']:.0f}",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#33691E"),
            row=1, col=1
        )

    if 'gex_resistance' in scenario_data:
        fig.add_hline(
            y=scenario_data['gex_resistance'],
            line_dash="dash",
            line_color="#FF1744",
            line_width=2,
            annotation_text=f"GEX RESISTANCE: ${scenario_data['gex_resistance']:.0f}",
            annotation_position="left",
            annotation=dict(font=dict(size=9, color="white"), bgcolor="#B71C1C"),
            row=1, col=1
        )

    # Add Vanna levels
    if 'vanna_support' in scenario_data:
        fig.add_hline(
            y=scenario_data['vanna_support'],
            line_dash="dot",
            line_color="#00E676",
            line_width=2,
            annotation_text=f"VANNA SUPPORT: ${scenario_data['vanna_support']:.0f}",
            annotation_position="right",
            row=1, col=1
        )

    if 'vanna_resistance' in scenario_data:
        fig.add_hline(
            y=scenario_data['vanna_resistance'],
            line_dash="dot",
            line_color="#FF5252",
            line_width=2,
            annotation_text=f"VANNA RESISTANCE: ${scenario_data['vanna_resistance']:.0f}",
            annotation_position="right",
            row=1, col=1
        )

    # Add Put/Call walls
    if 'put_wall' in scenario_data:
        fig.add_hline(
            y=scenario_data['put_wall'],
            line_dash="dashdot",
            line_color="#9C27B0",
            line_width=1.5,
            annotation_text=f"PUT WALL: ${scenario_data['put_wall']:.0f}",
            annotation_position="left",
            row=1, col=1
        )

    if 'call_wall' in scenario_data:
        fig.add_hline(
            y=scenario_data['call_wall'],
            line_dash="dashdot",
            line_color="#FF9800",
            line_width=1.5,
            annotation_text=f"CALL WALL: ${scenario_data['call_wall']:.0f}",
            annotation_position="left",
            row=1, col=1
        )

    # Add VWAP
    vwap_price = scenario_data['current_price']
    fig.add_hline(
        y=vwap_price,
        line_dash="solid",
        line_color="yellow",
        line_width=1,
        opacity=0.6,
        annotation_text=f"VWAP: ${vwap_price:.2f}",
        annotation_position="left",
        row=1, col=1
    )

    # ========== PANEL 2: IV & VANNA ==========

    # IV line
    iv_values = np.linspace(scenario_data['iv_start'], scenario_data['iv_end'], 50)
    fig.add_trace(go.Scatter(
        x=times,
        y=iv_values,
        mode='lines',
        name='IV',
        line=dict(color='#2196F3', width=2),
        showlegend=True
    ), row=2, col=1)

    # Vanna line
    vanna_values = np.linspace(scenario_data['vanna_start'], scenario_data['vanna_end'], 50)
    fig.add_trace(go.Scatter(
        x=times,
        y=vanna_values,
        mode='lines',
        name='Vanna',
        line=dict(color='#FF9800', width=2),
        showlegend=True
    ), row=2, col=1)

    # Vanna×IV line
    vanna_iv = iv_values * vanna_values * 100
    fig.add_trace(go.Scatter(
        x=times,
        y=vanna_iv,
        mode='lines',
        name='Vanna×IV',
        line=dict(color='#9C27B0', width=2, dash='dot'),
        showlegend=True
    ), row=2, col=1)

    # ========== PANEL 3: DEALER FLOW ==========

    # Charm pressure
    charm_values = np.linspace(scenario_data['charm_start'], scenario_data['charm_end'], 50)
    fig.add_trace(go.Scatter(
        x=times,
        y=charm_values,
        mode='lines',
        name='Charm',
        line=dict(color='#E91E63', width=2),
        fill='tozeroy',
        fillcolor='rgba(233, 30, 99, 0.2)',
        showlegend=True
    ), row=3, col=1)

    # GEX pressure
    gex_pressure = np.linspace(scenario_data['gex_pressure_start'],
                               scenario_data['gex_pressure_end'], 50)
    fig.add_trace(go.Scatter(
        x=times,
        y=gex_pressure,
        mode='lines',
        name='GEX Pressure',
        line=dict(color='#4CAF50', width=2),
        showlegend=True
    ), row=3, col=1)

    # Dealer flow score
    dealer_flow = np.linspace(scenario_data['dealer_flow_start'],
                             scenario_data['dealer_flow_end'], 50)
    fig.add_trace(go.Scatter(
        x=times,
        y=dealer_flow,
        mode='lines',
        name='Dealer Flow Score',
        line=dict(color='#FFC107', width=3),
        showlegend=True
    ), row=3, col=1)

    # Add zero line for Panel 3
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1, row=3, col=1)

    # Update layout
    fig.update_layout(
        height=900,
        title=dict(
            text=f"<b>{scenario_name}</b><br>{scenario_data['description']}",
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='x unified',
        template='plotly_dark',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Value", row=2, col=1)
    fig.update_yaxes(title_text="Score", row=3, col=1)

    return fig


def generate_bullish_example():
    """Generate a bullish trading setup example"""
    return {
        'current_price': 450,
        'gamma_flip': 445,
        'gex_support': 442,
        'vanna_support': 447,
        'call_wall': 455,
        'put_wall': 440,
        'gex_regime': 'positive',
        'price_trend': 0.3,  # Upward trend
        'iv_start': 0.25,
        'iv_end': 0.22,  # IV declining
        'vanna_start': 0.5,
        'vanna_end': 0.7,  # Vanna increasing
        'charm_start': 10,
        'charm_end': 15,  # Charm pressure increasing
        'gex_pressure_start': 20,
        'gex_pressure_end': 35,  # Positive GEX pressure
        'dealer_flow_start': 25,
        'dealer_flow_end': 45,  # Bullish dealer flow
        'description': (
            "🟢 BULLISH SETUP: Price above Gamma Flip in positive GEX regime (green background). "
            "Vanna support below provides bounce levels. Rising Vanna×IV and positive dealer flow "
            "indicate dealers are supportive. Look for dips to Vanna support or VWAP for entries. "
            "Call wall above is potential target."
        )
    }


def generate_bearish_example():
    """Generate a bearish trading setup example"""
    return {
        'current_price': 440,
        'gamma_flip': 445,
        'gex_resistance': 448,
        'vanna_resistance': 443,
        'call_wall': 455,
        'put_wall': 435,
        'gex_regime': 'negative',
        'price_trend': -0.3,  # Downward trend
        'iv_start': 0.22,
        'iv_end': 0.28,  # IV rising
        'vanna_start': 0.7,
        'vanna_end': 0.4,  # Vanna declining
        'charm_start': -10,
        'charm_end': -18,  # Negative charm pressure
        'gex_pressure_start': -15,
        'gex_pressure_end': -30,  # Negative GEX pressure
        'dealer_flow_start': -20,
        'dealer_flow_end': -40,  # Bearish dealer flow
        'description': (
            "🔴 BEARISH SETUP: Price below Gamma Flip in negative GEX regime (red background). "
            "Vanna resistance above acts as ceiling. Rising IV with falling Vanna×IV and negative "
            "dealer flow indicate dealers are selling. Look for rallies to Vanna resistance or VWAP "
            "for short entries. Put wall below is potential target."
        )
    }


def generate_neutral_example():
    """Generate a range-bound/neutral trading setup example"""
    return {
        'current_price': 445,
        'gamma_flip': 445,
        'gex_support': 441,
        'gex_resistance': 449,
        'vanna_support': 443,
        'vanna_resistance': 447,
        'call_wall': 452,
        'put_wall': 438,
        'gex_regime': 'positive',
        'price_trend': 0.0,  # Sideways
        'iv_start': 0.24,
        'iv_end': 0.24,  # Stable IV
        'vanna_start': 0.6,
        'vanna_end': 0.6,  # Stable Vanna
        'charm_start': 5,
        'charm_end': 5,  # Stable charm
        'gex_pressure_start': 10,
        'gex_pressure_end': 10,  # Stable GEX
        'dealer_flow_start': 0,
        'dealer_flow_end': 5,  # Near-neutral dealer flow
        'description': (
            "⚪ RANGE-BOUND SETUP: Price at Gamma Flip with tight GEX walls creating range. "
            "Put wall below and Call wall above define the boundaries. Stable IV and dealer flow "
            "near neutral suggest choppy, mean-reverting action. Trade the range: buy GEX support/"
            "Put wall, sell GEX resistance/Call wall. Avoid breakout trades until dealer flow shifts."
        )
    }


if __name__ == "__main__":
    """Generate and save all example charts"""

    print("Generating Trading Example Charts...")
    print("=" * 60)

    # Generate examples
    examples = [
        ("Bullish Setup Example", generate_bullish_example()),
        ("Bearish Setup Example", generate_bearish_example()),
        ("Range-Bound Setup Example", generate_neutral_example())
    ]

    # Create and save each chart
    for scenario_name, scenario_data in examples:
        print(f"\n📊 Creating: {scenario_name}")
        fig = create_example_chart(scenario_name, scenario_data)

        # Save as HTML
        filename = f"{scenario_name.lower().replace(' ', '_')}.html"
        fig.write_html(filename)
        print(f"   ✅ Saved to: {filename}")

        # Save as PNG (high resolution for readability)
        png_filename = f"{scenario_name.lower().replace(' ', '_')}.png"
        try:
            fig.write_image(png_filename, width=1400, height=1000, scale=2)
            print(f"   ✅ Saved PNG to: {png_filename}")
        except Exception as e:
            print(f"   ⚠️  Could not save PNG (install kaleido: pip install kaleido): {e}")

        # Also show in browser (optional - comment out if not needed)
        # fig.show()

    print("\n" + "=" * 60)
    print("✅ All example charts generated successfully!")
    print("\nTo view the charts:")
    print("  1. Open the .html files in your browser")
    print("  2. Or run: python trading_examples.py")
    print("\nKey Takeaways:")
    print("  🟢 Bullish: Price > Gamma Flip, positive GEX, rising Vanna×IV, +Dealer Flow")
    print("  🔴 Bearish: Price < Gamma Flip, negative GEX, falling Vanna×IV, -Dealer Flow")
    print("  ⚪ Neutral: Price ≈ Gamma Flip, tight range, stable indicators, neutral flow")
