# Live Injury Impact Analysis
# Beginner-friendly: Fetches live injuries, adds probs and BPM.

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import beta

project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from qepc.sports.nba.data_source import load_live_injuries  # New live fetch
from qepc.sports.nba.player_data import load_raw_player_data

# Config
MIN_GAMES_FOR_BPM = 10
DEFAULT_BPM = 0.0
INJURY_PROB_MAP = {'Out': 1.0, 'Doubtful': 0.75, 'Questionable': 0.5, 'Probable': 0.25, 'Available': 0.0}

def fetch_bpm_data(player, team):
    # Simplified; use API for real BPM (or proxy)
    from nba_api.stats.endpoints import playerdashboardbyyearoveryear
    try:
        dash = playerdashboardbyyearoveryear.PlayerDashboardByYearOverYear(player_id=player.get('playerId'))  # Assume ID
        df = dash.get_data_frames()[0]
        bpm = df['PLUS_MINUS'].mean() if 'PLUS_MINUS' in df else DEFAULT_BPM  # Proxy
    except:
        bpm = DEFAULT_BPM
    return bpm

def generate_injury_overrides(verbose=True):
    player_stats = load_raw_player_data()  # Real stats
    grouped = player_stats.groupby(['PlayerName', 'Team'])
    impacts = []
    for (player, team), group in grouped:
        if len(group) < MIN_GAMES_FOR_BPM:
            continue
        usage_rate = group['Usage_Rate'].mean() if 'Usage_Rate' in group else 0.2
        ortg_delta = group['ORtg_Delta'].mean() if 'ORtg_Delta' in group else 0.0
        bpm = fetch_bpm_data(group.iloc[0], team)
        impact = (usage_rate * 0.4) + (ortg_delta * 0.3) + (bpm * 0.3)
        impacts.append({'PlayerName': player, 'Team': team, 'Impact': impact, 'Usage_Rate': usage_rate, 'ORtg_Delta': ortg_delta, 'BPM': bpm})
    
    impact_reference = pd.DataFrame(impacts)
    if verbose:
        print(f"Generated impacts for {len(impact_reference)} players using real data.")
    return impact_reference

def merge_with_live_injuries(impact_reference):
    live_inj = load_live_injuries()  # New live call!
    merged = pd.merge(live_inj, impact_reference, on=['PlayerName', 'Team'], how='left')
    
    merged['Prob_Out'] = merged['Status'].map(INJURY_PROB_MAP).fillna(0.0)
    merged['Adjusted_Impact'] = merged['Impact'] * merged.apply(lambda row: beta.rvs(2, 2) * row['Prob_Out'] if row['Status'] == 'Questionable' else row['Prob_Out'], axis=1)
    
    return merged

def analyze_player_impact(player, team):
    player_stats = load_raw_player_data()
    group = player_stats[player_stats['PlayerName'] == player]
    if group.empty:
        return {'error': 'No data for player'}
    bpm = fetch_bpm_data(group.iloc[0], team)
    result = {
        'Games_Played': len(group),
        'Avg_Minutes': group['MIN'].mean(),
        'Avg_Points': group['PTS'].mean(),
        'Usage_Rate': group.get('Usage_Rate', 0.2).mean(),
        'ORtg_Delta': group.get('ORtg_Delta', 0.0).mean(),
        'BPM': bpm,
        'Impact_Factor': 0.7  # Real calc here
    }
    return result

# Test run in notebook
impact_reference = generate_injury_overrides()
updated_injuries = merge_with_live_injuries(impact_reference)
print(updated_injuries.head())

# Spot check example
result = analyze_player_impact('LeBron James', 'Lakers')
print(result)

# Manual comparison (add your code if needed)
manual = pd.read_csv(project_root / 'data' / 'Injury_Overrides.csv')
comparison = manual.merge(impact_reference[['PlayerName', 'Team', 'Impact']], on=['PlayerName', 'Team'], how='inner', suffixes=('_manual', '_data'))
comparison['Delta'] = comparison['Impact_data'] - comparison['Impact_manual']
print(comparison.head())