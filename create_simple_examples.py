"""
Create simple, easy-to-read example diagrams for trading setups
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_bullish_example():
    """Create a simple bullish setup diagram"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.patch.set_facecolor('#1e1e1e')

    # Panel 1: Price Chart
    ax1.set_facecolor('#0a4d0a')  # Green background for positive GEX
    ax1.set_title('🟢 BULLISH SETUP - Panel 1: Price & Options Flow',
                  fontsize=20, fontweight='bold', color='white', pad=20)

    # Simulated price action (trending up above gamma flip)
    x = np.linspace(0, 100, 100)
    price = 445 + np.cumsum(np.random.randn(100) * 0.3 + 0.2)
    ax1.plot(x, price, color='#26a69a', linewidth=3, label='Price (Uptrend)')

    # Key levels
    ax1.axhline(y=455, color='#FF9800', linewidth=3, linestyle='--',
                label='CALL WALL: $455 (Target)')
    ax1.axhline(y=445, color='#00BCD4', linewidth=4, linestyle='-',
                label='GAMMA FLIP: $445 (Major Pivot)')
    ax1.axhline(y=447, color='#00E676', linewidth=2, linestyle=':',
                label='VANNA SUPPORT: $447 (Entry Zone)')
    ax1.axhline(y=442, color='#76FF03', linewidth=2, linestyle='--',
                label='GEX SUPPORT: $442 (Strong Floor)')
    ax1.axhline(y=440, color='#9C27B0', linewidth=2, linestyle='-.',
                label='PUT WALL: $440 (Safety Net)')

    # Add annotations
    ax1.annotate('✅ PRICE ABOVE GAMMA FLIP\n(Positive GEX Regime)',
                xy=(50, 460), fontsize=14, color='white',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
                ha='center', fontweight='bold')

    ax1.annotate('🎯 ENTRY:\nBuy dips to\nVanna Support',
                xy=(70, 447), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='#004d00', alpha=0.9),
                ha='center')

    ax1.set_ylabel('Price ($)', fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, facecolor='#2e2e2e', edgecolor='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white', labelsize=11)

    # Panel 2: IV & Vanna
    ax2.set_facecolor('#2e2e2e')
    ax2.set_title('Panel 2: IV & Vanna Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    iv = np.linspace(0.25, 0.22, 100)  # Declining IV (bullish)
    vanna = np.linspace(0.5, 0.7, 100)  # Rising Vanna (bullish)
    vanna_iv = iv * vanna * 100

    ax2.plot(x, iv * 100, color='#2196F3', linewidth=3, label='IV (Declining ↓)')
    ax2.plot(x, vanna * 100, color='#FF9800', linewidth=3, label='Vanna (Rising ↑)')
    ax2.plot(x, vanna_iv, color='#9C27B0', linewidth=3, linestyle='--',
             label='Vanna×IV (Rising ↑)')

    ax2.annotate('✅ BULLISH SIGNAL:\nRising Vanna×IV',
                xy=(50, 18), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
                ha='center', fontweight='bold')

    ax2.set_ylabel('Value', fontsize=12, color='white', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white', labelsize=10)

    # Panel 3: Dealer Flow
    ax3.set_facecolor('#2e2e2e')
    ax3.set_title('Panel 3: Dealer Flow Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    charm = np.linspace(10, 15, 100)
    gex_pressure = np.linspace(20, 35, 100)
    dealer_flow = np.linspace(25, 45, 100)

    ax3.fill_between(x, 0, charm, color='#E91E63', alpha=0.3, label='Charm Pressure')
    ax3.plot(x, gex_pressure, color='#4CAF50', linewidth=3, label='GEX Pressure (Positive)')
    ax3.plot(x, dealer_flow, color='#FFC107', linewidth=4, label='Dealer Flow Score (Bullish)')
    ax3.axhline(y=0, color='gray', linewidth=1, linestyle='--')

    ax3.annotate('✅ BULLISH SIGNAL:\nDealer Flow > +20',
                xy=(50, 38), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.8),
                ha='center', fontweight='bold')

    ax3.set_xlabel('Time', fontsize=12, color='white', fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, color='white', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    plt.savefig('bullish_setup_simple.png', dpi=150, facecolor='#1e1e1e',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: bullish_setup_simple.png")


def create_bearish_example():
    """Create a simple bearish setup diagram"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.patch.set_facecolor('#1e1e1e')

    # Panel 1: Price Chart
    ax1.set_facecolor('#4d0a0a')  # Red background for negative GEX
    ax1.set_title('🔴 BEARISH SETUP - Panel 1: Price & Options Flow',
                  fontsize=20, fontweight='bold', color='white', pad=20)

    # Simulated price action (trending down below gamma flip)
    x = np.linspace(0, 100, 100)
    price = 440 + np.cumsum(np.random.randn(100) * 0.3 - 0.2)
    ax1.plot(x, price, color='#ef5350', linewidth=3, label='Price (Downtrend)')

    # Key levels
    ax1.axhline(y=455, color='#FF9800', linewidth=2, linestyle='-.',
                label='CALL WALL: $455 (Ceiling)')
    ax1.axhline(y=448, color='#FF1744', linewidth=2, linestyle='--',
                label='GEX RESISTANCE: $448 (Strong Ceiling)')
    ax1.axhline(y=445, color='#00BCD4', linewidth=4, linestyle='-',
                label='GAMMA FLIP: $445 (Major Pivot)')
    ax1.axhline(y=443, color='#FF5252', linewidth=2, linestyle=':',
                label='VANNA RESISTANCE: $443 (Entry Zone)')
    ax1.axhline(y=435, color='#9C27B0', linewidth=3, linestyle='--',
                label='PUT WALL: $435 (Target)')

    # Add annotations
    ax1.annotate('❌ PRICE BELOW GAMMA FLIP\n(Negative GEX Regime)',
                xy=(50, 430), fontsize=14, color='white',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8),
                ha='center', fontweight='bold')

    ax1.annotate('🎯 ENTRY:\nSell rallies to\nVanna Resistance',
                xy=(70, 443), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='#4d0000', alpha=0.9),
                ha='center')

    ax1.set_ylabel('Price ($)', fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=11, facecolor='#2e2e2e', edgecolor='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white', labelsize=11)

    # Panel 2: IV & Vanna
    ax2.set_facecolor('#2e2e2e')
    ax2.set_title('Panel 2: IV & Vanna Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    iv = np.linspace(0.22, 0.28, 100)  # Rising IV (bearish)
    vanna = np.linspace(0.7, 0.4, 100)  # Falling Vanna (bearish)
    vanna_iv = iv * vanna * 100

    ax2.plot(x, iv * 100, color='#2196F3', linewidth=3, label='IV (Rising ↑)')
    ax2.plot(x, vanna * 100, color='#FF9800', linewidth=3, label='Vanna (Falling ↓)')
    ax2.plot(x, vanna_iv, color='#9C27B0', linewidth=3, linestyle='--',
             label='Vanna×IV (Falling ↓)')

    ax2.annotate('❌ BEARISH SIGNAL:\nFalling Vanna×IV',
                xy=(50, 15), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8),
                ha='center', fontweight='bold')

    ax2.set_ylabel('Value', fontsize=12, color='white', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white', labelsize=10)

    # Panel 3: Dealer Flow
    ax3.set_facecolor('#2e2e2e')
    ax3.set_title('Panel 3: Dealer Flow Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    charm = np.linspace(-10, -18, 100)
    gex_pressure = np.linspace(-15, -30, 100)
    dealer_flow = np.linspace(-20, -40, 100)

    ax3.fill_between(x, 0, charm, color='#E91E63', alpha=0.3, label='Charm Pressure')
    ax3.plot(x, gex_pressure, color='#F44336', linewidth=3, label='GEX Pressure (Negative)')
    ax3.plot(x, dealer_flow, color='#FFC107', linewidth=4, label='Dealer Flow Score (Bearish)')
    ax3.axhline(y=0, color='gray', linewidth=1, linestyle='--')

    ax3.annotate('❌ BEARISH SIGNAL:\nDealer Flow < -20',
                xy=(50, -32), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.8),
                ha='center', fontweight='bold')

    ax3.set_xlabel('Time', fontsize=12, color='white', fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, color='white', fontweight='bold')
    ax3.legend(loc='lower left', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    plt.savefig('bearish_setup_simple.png', dpi=150, facecolor='#1e1e1e',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: bearish_setup_simple.png")


def create_neutral_example():
    """Create a simple range-bound setup diagram"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                         gridspec_kw={'height_ratios': [3, 1, 1]})
    fig.patch.set_facecolor('#1e1e1e')

    # Panel 1: Price Chart
    ax1.set_facecolor('#0a4d0a')  # Slight green for positive but range-bound
    ax1.set_title('⚪ RANGE-BOUND SETUP - Panel 1: Price & Options Flow',
                  fontsize=20, fontweight='bold', color='white', pad=20)

    # Simulated price action (choppy around gamma flip)
    x = np.linspace(0, 100, 100)
    price = 445 + np.sin(x/10) * 2 + np.random.randn(100) * 0.5
    ax1.plot(x, price, color='#FFD700', linewidth=3, label='Price (Range-Bound)')

    # Key levels - tight range
    ax1.axhline(y=452, color='#FF9800', linewidth=2, linestyle='-.',
                label='CALL WALL: $452 (Range Top)')
    ax1.axhline(y=449, color='#FF1744', linewidth=2, linestyle='--',
                label='GEX RESISTANCE: $449 (Sell Zone)')
    ax1.axhline(y=445, color='#00BCD4', linewidth=4, linestyle='-',
                label='GAMMA FLIP: $445 (Pivot)')
    ax1.axhline(y=441, color='#76FF03', linewidth=2, linestyle='--',
                label='GEX SUPPORT: $441 (Buy Zone)')
    ax1.axhline(y=438, color='#9C27B0', linewidth=2, linestyle='-.',
                label='PUT WALL: $438 (Range Bottom)')

    # Add annotations
    ax1.annotate('⚪ PRICE AT GAMMA FLIP\n(Tight Range)',
                xy=(50, 453), fontsize=14, color='white',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.8),
                ha='center', fontweight='bold')

    ax1.annotate('🎯 BUY HERE\nGEX Support',
                xy=(20, 441), fontsize=11, color='white',
                bbox=dict(boxstyle='round', facecolor='green', alpha=0.9),
                ha='center')

    ax1.annotate('🎯 SELL HERE\nGEX Resistance',
                xy=(80, 449), fontsize=11, color='white',
                bbox=dict(boxstyle='round', facecolor='darkred', alpha=0.9),
                ha='center')

    ax1.set_ylabel('Price ($)', fontsize=14, color='white', fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11, facecolor='#2e2e2e', edgecolor='white')
    ax1.grid(True, alpha=0.2, color='white')
    ax1.tick_params(colors='white', labelsize=11)

    # Panel 2: IV & Vanna
    ax2.set_facecolor('#2e2e2e')
    ax2.set_title('Panel 2: IV & Vanna Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    iv = np.ones(100) * 0.24  # Stable IV
    vanna = np.ones(100) * 0.6  # Stable Vanna
    vanna_iv = iv * vanna * 100

    ax2.plot(x, iv * 100, color='#2196F3', linewidth=3, label='IV (Stable →)')
    ax2.plot(x, vanna * 100, color='#FF9800', linewidth=3, label='Vanna (Stable →)')
    ax2.plot(x, vanna_iv, color='#9C27B0', linewidth=3, linestyle='--',
             label='Vanna×IV (Stable →)')

    ax2.annotate('⚪ NEUTRAL SIGNAL:\nStable Indicators',
                xy=(50, 16), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.8),
                ha='center', fontweight='bold')

    ax2.set_ylabel('Value', fontsize=12, color='white', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax2.grid(True, alpha=0.2, color='white')
    ax2.tick_params(colors='white', labelsize=10)

    # Panel 3: Dealer Flow
    ax3.set_facecolor('#2e2e2e')
    ax3.set_title('Panel 3: Dealer Flow Indicators', fontsize=16,
                  fontweight='bold', color='white', pad=10)

    charm = np.ones(100) * 5
    gex_pressure = np.ones(100) * 10
    dealer_flow = np.linspace(0, 5, 100)

    ax3.fill_between(x, 0, charm, color='#E91E63', alpha=0.3, label='Charm Pressure')
    ax3.plot(x, gex_pressure, color='#4CAF50', linewidth=3, label='GEX Pressure (Stable)')
    ax3.plot(x, dealer_flow, color='#FFC107', linewidth=4, label='Dealer Flow Score (Neutral)')
    ax3.axhline(y=0, color='gray', linewidth=1, linestyle='--')
    ax3.axhline(y=20, color='green', linewidth=1, linestyle=':', alpha=0.5)
    ax3.axhline(y=-20, color='red', linewidth=1, linestyle=':', alpha=0.5)

    ax3.annotate('⚪ NEUTRAL:\n-20 < Flow < +20',
                xy=(50, 12), fontsize=12, color='white',
                bbox=dict(boxstyle='round', facecolor='gray', alpha=0.8),
                ha='center', fontweight='bold')

    ax3.set_xlabel('Time', fontsize=12, color='white', fontweight='bold')
    ax3.set_ylabel('Score', fontsize=12, color='white', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10, facecolor='#2e2e2e', edgecolor='white')
    ax3.grid(True, alpha=0.2, color='white')
    ax3.tick_params(colors='white', labelsize=10)

    plt.tight_layout()
    plt.savefig('range-bound_setup_simple.png', dpi=150, facecolor='#1e1e1e',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: range-bound_setup_simple.png")


if __name__ == "__main__":
    print("Creating simple, readable trading example diagrams...")
    print("=" * 60)

    create_bullish_example()
    create_bearish_example()
    create_neutral_example()

    print("=" * 60)
    print("✅ All simple diagrams created successfully!")
    print("\nFiles created:")
    print("  - bullish_setup_simple.png")
    print("  - bearish_setup_simple.png")
    print("  - range-bound_setup_simple.png")
