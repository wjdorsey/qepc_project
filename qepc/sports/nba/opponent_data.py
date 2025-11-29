"""
QEPC Module: opponent_data.py
Real opponent DRtg from API.
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashopponentstats

HISTORY_DAYS = 730
HALF_LIFE_DAYS = 60

def process_opponent_metrics():
    opp_stats = leaguedashopponentstats.LeagueDashOpponentStats(season='2025-26')
    df = opp_stats.get_data_frames()[0]
    
    # Time decay (mock dates—add real)
    df['gameDate'] = pd.to_datetime('today')  # Replace with real
    df['DaysAgo'] = (pd.to_datetime('today') - df['gameDate']).dt.days
    df['Weight'] = 0.5 ** (df['DaysAgo'] / HALF_LIFE_DAYS)
    
    df_group = df.groupby('TEAM_NAME').agg({'OPP_DEF_RATING': 'weighted_mean'}, weights=df['Weight'])
    df_group.reset_index(inplace=True)
    df_group.rename(columns={'TEAM_NAME': 'Team', 'OPP_DEF_RATING': 'DRtg'}, inplace=True)
    
    return df_group