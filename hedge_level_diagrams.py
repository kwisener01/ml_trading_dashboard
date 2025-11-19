"""
Hedge Level Reaction Diagrams
==============================
Visual diagrams showing how price reacts to Vanna and GEX levels

Run this to generate educational charts showing:
1. GEX regime behavior (positive vs negative)
2. Vanna level reactions (attractor vs repellent)
3. Combined hedge pressure scenarios
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def create_gex_regime_diagram():
    """
    Create diagram showing GEX positive vs negative regime behavior.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('GEX REGIME BEHAVIOR - How Dealers Hedge Affects Price', fontsize=16, fontweight='bold')

    # Positive GEX - Mean Reversion
    ax1 = axes[0]
    ax1.set_title('POSITIVE GEX\n(Dealers Long Gamma)', fontsize=12, fontweight='bold', color='green')

    # Price action
    x = np.linspace(0, 10, 100)
    # Price tries to move up but gets pushed back
    y1 = 50 + 5 * np.sin(x) * np.exp(-x/5)

    ax1.plot(x, y1, 'b-', linewidth=2, label='Price Action')
    ax1.axhline(y=50, color='cyan', linestyle='--', linewidth=2, label='GEX Flip Level')
    ax1.axhline(y=55, color='red', linestyle=':', linewidth=1.5, label='Resistance')
    ax1.axhline(y=45, color='green', linestyle=':', linewidth=1.5, label='Support')

    # Add arrows showing dealer hedging
    ax1.annotate('', xy=(3, 53), xytext=(3, 55),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax1.text(3.2, 54, 'Dealers SELL\n(push down)', fontsize=8, color='red')

    ax1.annotate('', xy=(6, 47), xytext=(6, 45),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.text(6.2, 46, 'Dealers BUY\n(push up)', fontsize=8, color='green')

    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_ylim(40, 60)
    ax1.legend(loc='upper right', fontsize=8)

    # Add explanation box
    textstr = 'MEAN REVERSION MODE\n' + '-'*25 + '\n' + \
              'Price bounces between levels\n' + \
              'Fade extremes\n' + \
              'Sell rallies, buy dips\n' + \
              'Tight ranges expected'
    ax1.text(0.02, 0.02, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Negative GEX - Momentum
    ax2 = axes[1]
    ax2.set_title('NEGATIVE GEX\n(Dealers Short Gamma)', fontsize=12, fontweight='bold', color='red')

    # Price action - trending
    y2 = 50 + x * 1.5 + 2 * np.sin(x * 2)

    ax2.plot(x, y2, 'b-', linewidth=2, label='Price Action')
    ax2.axhline(y=50, color='cyan', linestyle='--', linewidth=2, label='GEX Flip Level')
    ax2.axhline(y=65, color='red', linestyle=':', linewidth=1.5, label='Resistance')
    ax2.axhline(y=45, color='green', linestyle=':', linewidth=1.5, label='Support')

    # Add arrows showing dealer hedging accelerating moves
    ax2.annotate('', xy=(4, 58), xytext=(4, 54),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(4.2, 56, 'Dealers BUY\n(chase up)', fontsize=8, color='green')

    ax2.annotate('', xy=(7, 63), xytext=(7, 59),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax2.text(7.2, 61, 'More buying\n(momentum)', fontsize=8, color='green')

    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Price', fontsize=10)
    ax2.set_ylim(40, 70)
    ax2.legend(loc='upper left', fontsize=8)

    # Add explanation box
    textstr = 'MOMENTUM MODE\n' + '-'*25 + '\n' + \
              'Moves accelerate\n' + \
              'Follow the trend\n' + \
              'Breakouts continue\n' + \
              'Large ranges expected'
    ax2.text(0.02, 0.02, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig('gex_regime_diagram.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: gex_regime_diagram.png")
    plt.show()


def create_vanna_reaction_diagram():
    """
    Create diagram showing how price reacts to Vanna levels.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('VANNA LEVEL REACTIONS - Volatility-Driven Hedging', fontsize=16, fontweight='bold')

    # Positive Vanna - Attractor (Support)
    ax1 = axes[0]
    ax1.set_title('POSITIVE VANNA (Support)\nAttractor - Price Gets Pulled', fontsize=12, fontweight='bold', color='purple')

    x = np.linspace(0, 10, 100)
    # Price approaches and bounces off support
    y1 = 50 - 8 * np.exp(-x/3) * np.sin(x * 1.5)
    y1 = np.maximum(y1, 45)  # Floor at support

    ax1.plot(x, y1, 'b-', linewidth=2, label='Price Action')
    ax1.axhline(y=45, color='purple', linestyle='--', linewidth=3, label='Vanna Support (Attractor)')
    ax1.fill_between([0, 10], 43, 47, color='purple', alpha=0.2)

    # Add magnet arrows
    for i in range(3):
        start_x = 2 + i * 2
        ax1.annotate('', xy=(start_x, 45), xytext=(start_x, 48),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.5))

    ax1.text(5, 44, 'Magnet Effect\nPulls price down', fontsize=9, ha='center', color='purple')

    ax1.set_xlabel('Time', fontsize=10)
    ax1.set_ylabel('Price', fontsize=10)
    ax1.set_ylim(40, 55)
    ax1.legend(loc='upper right', fontsize=8)

    # Explanation
    textstr = 'When IV Rises:\n' + '-'*20 + '\n' + \
              'Delta increases\n' + \
              'Dealers buy shares\n' + \
              'Price pulled to level\n' + \
              'Strong bounce expected'
    ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.8))

    # Negative Vanna - Repellent (Resistance)
    ax2 = axes[1]
    ax2.set_title('NEGATIVE VANNA (Resistance)\nRepellent - Price Gets Pushed', fontsize=12, fontweight='bold', color='orange')

    # Price approaches resistance and gets rejected
    y2 = 50 + 8 * np.exp(-x/3) * np.sin(x * 1.5)
    y2 = np.minimum(y2, 55)  # Ceiling at resistance

    ax2.plot(x, y2, 'b-', linewidth=2, label='Price Action')
    ax2.axhline(y=55, color='orange', linestyle='--', linewidth=3, label='Vanna Resistance (Repellent)')
    ax2.fill_between([0, 10], 53, 57, color='orange', alpha=0.2)

    # Add push arrows
    for i in range(3):
        start_x = 2 + i * 2
        ax2.annotate('', xy=(start_x, 55), xytext=(start_x, 52),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1.5))

    ax2.text(5, 56, 'Repel Effect\nPushes price down', fontsize=9, ha='center', color='darkorange')

    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Price', fontsize=10)
    ax2.set_ylim(45, 60)
    ax2.legend(loc='lower right', fontsize=8)

    # Explanation
    textstr = 'When IV Rises:\n' + '-'*20 + '\n' + \
              'Delta decreases\n' + \
              'Dealers sell shares\n' + \
              'Price pushed away\n' + \
              'Rejection expected'
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('vanna_reaction_diagram.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: vanna_reaction_diagram.png")
    plt.show()


def create_combined_levels_diagram():
    """
    Create diagram showing combined Vanna and GEX levels on one chart.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.suptitle('COMBINED HEDGE LEVELS - Trading Setup Example', fontsize=16, fontweight='bold')

    # Create example price movement
    x = np.linspace(0, 20, 200)
    y = 590 + 15 * np.sin(x/3) + 5 * np.sin(x) + x * 0.5

    ax.plot(x, y, 'b-', linewidth=2, label='SPY Price')

    # Current price
    current = 595
    ax.axhline(y=current, color='blue', linestyle='-', linewidth=2.5)
    ax.text(0.5, current + 1, f'Current: ${current}', fontsize=10, fontweight='bold', color='blue')

    # GEX Levels
    gex_flip = 600
    gex_support = 580
    gex_resistance = 610

    ax.axhline(y=gex_flip, color='cyan', linestyle='-.', linewidth=2)
    ax.text(15, gex_flip + 1, f'GEX Flip: ${gex_flip}', fontsize=9, color='cyan', fontweight='bold')

    ax.axhline(y=gex_support, color='lime', linestyle=':', linewidth=2)
    ax.text(0.5, gex_support - 2, f'GEX Support: ${gex_support}\nDealers BUY', fontsize=8, color='green')

    ax.axhline(y=gex_resistance, color='magenta', linestyle=':', linewidth=2)
    ax.text(15, gex_resistance + 1, f'GEX Resistance: ${gex_resistance}\nDealers SELL', fontsize=8, color='magenta')

    # Vanna Levels
    vanna_support = 585
    vanna_resistance = 605

    ax.axhline(y=vanna_support, color='purple', linestyle='--', linewidth=1.5)
    ax.text(0.5, vanna_support + 1, f'Vanna S1: ${vanna_support}', fontsize=8, color='purple')

    ax.axhline(y=vanna_resistance, color='orange', linestyle='--', linewidth=1.5)
    ax.text(0.5, vanna_resistance + 1, f'Vanna R1: ${vanna_resistance}', fontsize=8, color='darkorange')

    # Trading zones
    ax.fill_between([0, 20], gex_support, vanna_support, color='green', alpha=0.1, label='Strong Support Zone')
    ax.fill_between([0, 20], vanna_resistance, gex_resistance, color='red', alpha=0.1, label='Strong Resistance Zone')

    # Add reaction annotations
    ax.annotate('BOUNCE\nZONE', xy=(10, 582), fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    ax.annotate('REJECTION\nZONE', xy=(10, 608), fontsize=11, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # Regime indicator
    ax.annotate('Below GEX Flip = Momentum Mode\nAbove GEX Flip = Mean Reversion',
               xy=(17, 600), fontsize=8, ha='center',
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('Price ($)', fontsize=10)
    ax.set_xlim(0, 20)
    ax.set_ylim(575, 615)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('combined_levels_diagram.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: combined_levels_diagram.png")
    plt.show()


def create_trading_playbook():
    """
    Create a visual trading playbook based on level interactions.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('HEDGE LEVEL TRADING PLAYBOOK', fontsize=16, fontweight='bold')

    scenarios = [
        {
            'title': 'Scenario 1: Price at GEX Support',
            'action': 'LONG',
            'reason': 'Dealers BUY to hedge\nStrong bounce likely',
            'color': 'green',
            'price_move': 'up'
        },
        {
            'title': 'Scenario 2: Price at GEX Resistance',
            'action': 'SHORT/WAIT',
            'reason': 'Dealers SELL to hedge\nRejection likely',
            'color': 'red',
            'price_move': 'down'
        },
        {
            'title': 'Scenario 3: Break Above GEX Flip',
            'action': 'FADE RALLIES',
            'reason': 'Entering mean reversion\nSell spikes',
            'color': 'orange',
            'price_move': 'range'
        },
        {
            'title': 'Scenario 4: Break Below GEX Flip',
            'action': 'FOLLOW TREND',
            'reason': 'Entering momentum\nChase breaks',
            'color': 'blue',
            'price_move': 'trend'
        }
    ]

    for idx, (ax, scenario) in enumerate(zip(axes.flatten(), scenarios)):
        ax.set_title(scenario['title'], fontsize=11, fontweight='bold')

        # Simple price illustration
        x = np.linspace(0, 10, 100)
        if scenario['price_move'] == 'up':
            y = 50 + x * 0.5 + 2 * np.sin(x)
            ax.plot(x, y, 'g-', linewidth=2)
            ax.annotate('', xy=(8, 56), xytext=(2, 51),
                       arrowprops=dict(arrowstyle='->', color='green', lw=3))
        elif scenario['price_move'] == 'down':
            y = 55 - x * 0.5 - 2 * np.sin(x)
            ax.plot(x, y, 'r-', linewidth=2)
            ax.annotate('', xy=(8, 49), xytext=(2, 54),
                       arrowprops=dict(arrowstyle='->', color='red', lw=3))
        elif scenario['price_move'] == 'range':
            y = 50 + 3 * np.sin(x * 1.5)
            ax.plot(x, y, 'orange', linewidth=2)
        else:  # trend
            y = 50 + x * 0.8 + 1 * np.sin(x * 2)
            ax.plot(x, y, 'b-', linewidth=2)
            ax.annotate('', xy=(8, 58), xytext=(2, 51),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=3))

        # Action box
        ax.text(0.5, 0.15, f'ACTION: {scenario["action"]}', transform=ax.transAxes,
               fontsize=12, fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor=scenario['color'], alpha=0.3))

        # Reason
        ax.text(0.5, 0.85, scenario['reason'], transform=ax.transAxes,
               fontsize=9, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlim(0, 10)
        ax.set_ylim(45, 60)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.savefig('trading_playbook_diagram.png', dpi=150, bbox_inches='tight')
    print("[OK] Saved: trading_playbook_diagram.png")
    plt.show()


def main():
    """Generate all hedge level diagrams."""

    print("="*60)
    print("GENERATING HEDGE LEVEL DIAGRAMS")
    print("="*60)
    print()

    print("[1/4] Creating GEX regime diagram...")
    create_gex_regime_diagram()
    print()

    print("[2/4] Creating Vanna reaction diagram...")
    create_vanna_reaction_diagram()
    print()

    print("[3/4] Creating combined levels diagram...")
    create_combined_levels_diagram()
    print()

    print("[4/4] Creating trading playbook...")
    create_trading_playbook()
    print()

    print("="*60)
    print("ALL DIAGRAMS GENERATED!")
    print("="*60)
    print()
    print("Files created:")
    print("  - gex_regime_diagram.png")
    print("  - vanna_reaction_diagram.png")
    print("  - combined_levels_diagram.png")
    print("  - trading_playbook_diagram.png")
    print()
    print("Use these to understand how price reacts to hedge levels!")


if __name__ == "__main__":
    main()
