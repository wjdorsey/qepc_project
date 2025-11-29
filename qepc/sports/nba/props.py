"""
QEPC Module: props.py
Real prop projections.
"""

import pandas as pd
from qepc.sports.nba.player_data import load_raw_player_data
from qepc.sports.nba.opponent_data import process_opponent_metrics
from qepc.sports.nba.props_adjust import adjust_prop

def project_player_prop(player_name: str, stat: str = 'PTS', opponent_team: str = None, projected_min: float = None):
    player_df = load_raw_player_data()  # Real load
    player_games = player_df[player_df['PlayerName'] == player_name]
    
    if len(player_games) < 5:
        return {'projection': 0, 'variance': 0}
    
    avg_stat = player_games[stat].mean()
    avg_min = player_games['MIN'].mean()
    projected_min = projected_min or avg_min
    
    opp_metrics = process_opponent_metrics()  # Real opponents
    opp_drtg = opp_metrics.loc[opp_metrics['Team'] == opponent_team, 'DRtg'].values[0] if opponent_team and not opp_metrics.empty else 110.5
    
    projection = adjust_prop(avg_stat, opp_drtg, projected_min)
    variance = player_games[stat].std() ** 2
    
    recent = player_games[stat].tail(5).mean()
    final = (projection + recent) / 2  # Interference
    
    return {'projection': final, 'variance': variance}