# Injury Impact Analysis Notebook
# Run this to generate data-driven injury impact factors

# Cell 1: Setup
import sys
from pathlib import Path

# Add project root to path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qepc.sports.nba.injury_impact_calculator import (
    generate_injury_overrides,
    merge_with_live_injuries,
    analyze_player_impact
)

print("✅ Injury Impact Calculator loaded")

# ================================================
# Cell 2: Generate Impact Reference
# ================================================

# This will analyze ALL players in your PlayerStatistics.csv
# and calculate their impact factors based on usage + on/off splits

impact_reference = generate_injury_overrides(verbose=True)

# Preview top 20 highest-impact players
print("\n" + "="*60)
print("TOP 20 HIGHEST IMPACT PLAYERS (When Absent)")
print("="*60)
print(impact_reference[['PlayerName', 'Team', 'Impact', 'Usage_Rate', 'ORtg_Delta']].head(20))

# ================================================
# Cell 3: Merge with Live Injuries
# ================================================

# This merges your data-driven impacts with live injury reports
# from the nbainjuries API (or other sources)

updated_injuries = merge_with_live_injuries(impact_reference)

print("\nUpdated Injury Overrides Preview:")
print(updated_injuries.head(10))

# ================================================
# Cell 4: Spot Check Individual Players
# ================================================

# Quick lookup for specific players
players_to_check = [
    ("Tyrese Haliburton", "Indiana Pacers"),
    ("Jayson Tatum", "Boston Celtics"),
    ("Kevin Durant", "Phoenix Suns"),
    ("LeBron James", "Los Angeles Lakers"),
]

print("\n" + "="*60)
print("INDIVIDUAL PLAYER ANALYSIS")
print("="*60)

for player, team in players_to_check:
    result = analyze_player_impact(player, team)
    if "error" in result:
        print(f"\n❌ {player} ({team}): {result['error']}")
    else:
        print(f"\n📊 {player} ({team})")
        print(f"   Games Played: {result['Games_Played']}")
        print(f"   Avg Minutes: {result['Avg_Minutes']:.1f}")
        print(f"   Avg Points: {result['Avg_Points']:.1f}")
        print(f"   Usage Rate: {result['Usage_Rate']:.1%}")
        print(f"   ORtg Delta: {result['ORtg_Delta']:+.1f}")
        print(f"   🎯 Impact Factor: {result['Impact_Factor']}")

# ================================================
# Cell 5: Compare to Manual Overrides
# ================================================

# Load your manual injury overrides
manual_path = project_root / "data" / "Injury_Overrides.csv"

if manual_path.exists():
    import pandas as pd
    manual = pd.read_csv(manual_path)
    
    # Merge to compare
    comparison = manual.merge(
        impact_reference[['PlayerName', 'Team', 'Impact']],
        on=['PlayerName', 'Team'],
        how='inner',
        suffixes=('_manual', '_data')
    )
    
    comparison['Delta'] = comparison['Impact_data'] - comparison['Impact_manual']
    
    print("\n" + "="*60)
    print("MANUAL VS DATA-DRIVEN COMPARISON")
    print("="*60)
    print(comparison[['PlayerName', 'Team', 'Impact_manual', 'Impact_data', 'Delta']].head(15))
    
    print(f"\nAverage absolute difference: {comparison['Delta'].abs().mean():.3f}")
else:
    print("\n⚠️ No manual injury overrides found for comparison")