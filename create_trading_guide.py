"""
Create a simple trading decision flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_trading_flowchart():
    """Create a clear trading decision guide"""
    fig, ax = plt.subplots(figsize=(10, 12))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Title
    ax.text(5, 13.5, 'TRADING DECISION GUIDE',
            ha='center', va='top', fontsize=18, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#2e2e2e', edgecolor='white', linewidth=2))

    # Step 1: Check GEX Regime
    y = 12.5
    ax.add_patch(FancyBboxPatch((1, y-0.6), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2196F3', edgecolor='white', linewidth=2))
    ax.text(5, y-0.2, 'STEP 1: Check Background Color (Panel 1)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Green vs Red
    y = 11.3
    # Green box
    ax.add_patch(FancyBboxPatch((0.5, y-0.8), 4, 1, boxstyle="round,pad=0.1",
                                facecolor='#0a4d0a', edgecolor='#00ff00', linewidth=2))
    ax.text(2.5, y-0.5, '✅ GREEN BACKGROUND', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(2.5, y-0.8, 'Positive GEX', ha='center', va='center',
            fontsize=9, color='#90EE90')

    # Red box
    ax.add_patch(FancyBboxPatch((5.5, y-0.8), 4, 1, boxstyle="round,pad=0.1",
                                facecolor='#4d0a0a', edgecolor='#ff0000', linewidth=2))
    ax.text(7.5, y-0.5, '❌ RED BACKGROUND', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(7.5, y-0.8, 'Negative GEX', ha='center', va='center',
            fontsize=9, color='#FFB6C1')

    # Step 2: Check Price vs Gamma Flip
    y = 9.8
    ax.add_patch(FancyBboxPatch((1, y-0.6), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#FF9800', edgecolor='white', linewidth=2))
    ax.text(5, y-0.2, 'STEP 2: Check Price vs GAMMA FLIP (Cyan Line)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Above vs Below
    y = 8.6
    # Above
    ax.add_patch(FancyBboxPatch((0.5, y-0.6), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2e2e2e', edgecolor='#00ff00', linewidth=2))
    ax.text(2.5, y-0.2, 'Price ABOVE Gamma Flip', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#90EE90')

    # Below
    ax.add_patch(FancyBboxPatch((5.5, y-0.6), 4, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2e2e2e', edgecolor='#ff0000', linewidth=2))
    ax.text(7.5, y-0.2, 'Price BELOW Gamma Flip', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#FFB6C1')

    # Step 3: Check Dealer Flow
    y = 7.4
    ax.add_patch(FancyBboxPatch((1, y-0.6), 8, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#9C27B0', edgecolor='white', linewidth=2))
    ax.text(5, y-0.2, 'STEP 3: Check DEALER FLOW (Panel 3)',
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

    # Dealer Flow conditions
    y = 6.2
    # Bullish
    ax.add_patch(FancyBboxPatch((0.2, y-0.6), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2e2e2e', edgecolor='#00ff00', linewidth=2))
    ax.text(1.7, y-0.2, 'Score > +20\n(Bullish)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#90EE90')

    # Neutral
    ax.add_patch(FancyBboxPatch((3.5, y-0.6), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2e2e2e', edgecolor='yellow', linewidth=2))
    ax.text(5, y-0.2, 'Score -20 to +20\n(Neutral)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='yellow')

    # Bearish
    ax.add_patch(FancyBboxPatch((6.8, y-0.6), 3, 0.8, boxstyle="round,pad=0.1",
                                facecolor='#2e2e2e', edgecolor='#ff0000', linewidth=2))
    ax.text(8.3, y-0.2, 'Score < -20\n(Bearish)', ha='center', va='center',
            fontsize=9, fontweight='bold', color='#FFB6C1')

    # DECISION MATRIX
    y = 5.0
    ax.text(5, y, '═══ TRADING DECISIONS ═══',
            ha='center', va='center', fontsize=14, fontweight='bold', color='gold')

    # LONG Setups
    y = 4.2
    ax.add_patch(FancyBboxPatch((0.2, y-1.2), 4.5, 1.4, boxstyle="round,pad=0.2",
                                facecolor='#0a6b0a', edgecolor='#00ff00', linewidth=3))
    ax.text(2.45, y+0.1, '🟢 GO LONG', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='green', edgecolor='white'))
    ax.text(2.45, y-0.3, '✅ Green background (Pos GEX)', ha='left', va='center',
            fontsize=8, color='white', transform=ax.transData)
    ax.text(2.45, y-0.5, '✅ Above Gamma Flip', ha='left', va='center',
            fontsize=8, color='white')
    ax.text(2.45, y-0.7, '✅ Dealer Flow > +20', ha='left', va='center',
            fontsize=8, color='white')
    ax.text(2.45, y-0.95, 'Entry: Dips to Vanna Support/VWAP', ha='center', va='center',
            fontsize=8, color='yellow', fontweight='bold')

    # SHORT Setups
    ax.add_patch(FancyBboxPatch((5.3, y-1.2), 4.5, 1.4, boxstyle="round,pad=0.2",
                                facecolor='#6b0a0a', edgecolor='#ff0000', linewidth=3))
    ax.text(7.55, y+0.1, '🔴 GO SHORT', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='darkred', edgecolor='white'))
    ax.text(7.55, y-0.3, '✅ Red background (Neg GEX)', ha='left', va='center',
            fontsize=8, color='white')
    ax.text(7.55, y-0.5, '✅ Below Gamma Flip', ha='left', va='center',
            fontsize=8, color='white')
    ax.text(7.55, y-0.7, '✅ Dealer Flow < -20', ha='left', va='center',
            fontsize=8, color='white')
    ax.text(7.55, y-0.95, 'Entry: Rallies to Vanna Res/VWAP', ha='center', va='center',
            fontsize=8, color='yellow', fontweight='bold')

    # WAIT / Range Trade
    y = 2.2
    ax.add_patch(FancyBboxPatch((1, y-1.0), 8, 1.2, boxstyle="round,pad=0.2",
                                facecolor='#4d4d00', edgecolor='yellow', linewidth=3))
    ax.text(5, y+0.05, '⚪ WAIT / RANGE TRADE', ha='center', va='center',
            fontsize=13, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='gray', edgecolor='white'))
    ax.text(5, y-0.35, '⚠️ Mixed signals: Price near Gamma Flip OR Dealer Flow neutral (-20 to +20)',
            ha='center', va='center', fontsize=9, color='white')
    ax.text(5, y-0.65, 'Trade the range: Buy GEX Support, Sell GEX Resistance',
            ha='center', va='center', fontsize=9, color='yellow', fontweight='bold')

    # Quick Tips
    y = 0.7
    ax.text(5, y, '💡 PRO TIPS', ha='center', va='center',
            fontsize=12, fontweight='bold', color='gold')

    tips = [
        "• Use VWAP & Vanna levels as entry/exit points",
        "• Watch Put/Call walls as targets (magnets/barriers)",
        "• During NY Priority (10AM-12PM): Best setups",
        "• During Charm Session (3-4PM): Watch for pins to strikes"
    ]

    y = 0.2
    for tip in tips:
        ax.text(5, y, tip, ha='center', va='center',
                fontsize=8, color='white')
        y -= 0.25

    plt.tight_layout()
    plt.savefig('trading_decision_guide.png', dpi=150, facecolor='#1e1e1e',
                edgecolor='none', bbox_inches='tight')
    plt.close()
    print("✅ Created: trading_decision_guide.png")


if __name__ == "__main__":
    print("Creating Trading Decision Guide...")
    print("=" * 60)
    create_trading_flowchart()
    print("=" * 60)
    print("✅ Trading decision guide created successfully!")
