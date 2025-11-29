"""
QEPC NBA Team Strengths v2
Real from API, with SOS.
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguedashopponentstats

RECENCY_HALF_LIFE_DAYS = 45

def calculate_advanced_strengths(verbose: bool = False) -> pd.DataFrame:
    stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26')
    df = stats.get_data_frames()[0]
    
    # Real SOS: Avg opponent ORtg
    opp_stats = leaguedashopponentstats.LeagueDashOpponentStats(season='2025-26')
    opp_df = opp_stats.get_data_frames()[0]
    df = pd.merge(df, opp_df[['TEAM_NAME', 'OPP_OFF_RATING']], on='TEAM_NAME', how='left')
    df['SOS'] = df['OPP_OFF_RATING'] / df['OFF_RATING'].mean()  # Real SOS
    
    df.rename(columns={'TEAM_NAME': 'Team', 'OFF_RATING': 'ORtg', 'DEF_RATING': 'DRtg', 'PACE': 'Pace'}, inplace=True)
    if verbose:
        print(f"Loaded real {len(df)} teams with SOS.")
    return df[['Team', 'ORtg', 'DRtg', 'Pace', 'SOS', 'Volatility']]