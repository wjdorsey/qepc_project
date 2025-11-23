"""
QEPC Module: lambda_engine.py
Core modeling component for calculating Poisson rates (lambda).
"""
import pandas as pd
from typing import Dict

# --- Configuration ---
LEAGUE_AVG_POINTS_PER_GAME = 110.0 

def compute_lambda(
    schedule_df: pd.DataFrame, 
    team_stats_df: pd.DataFrame 
) -> pd.DataFrame:
    """
    Applies Team Strength Ratings (TSR) to the schedule to compute lambda.
    """
    if team_stats_df.empty: 
        print("[QEPC Lambda] ERROR: Cannot compute lambda without team strengths.")
        return schedule_df

    # 1. Calculate League Averages
    la_ortg = team_stats_df['ORtg'].mean()

    strengths = {}
    for index, row in team_stats_df.iterrows():
        team = row['Team']
        
        # Offensive Strength: Higher ORtg = Better Offense
        offensive_strength = row['ORtg'] / la_ortg
        
        # Defensive Strength: Higher DRtg = Worse Defense = Opponent Scores MORE
        # MATH FIX: High DRtg should increase opponent score, so we divide BY league avg
        defensive_strength = row['DRtg'] / la_ortg 

        strengths[team] = {
            'OS': offensive_strength,
            'DS': defensive_strength
        }
    
    # --- Calculation Loop ---
    df = schedule_df.copy()
    df['lambda_home'] = 0.0
    df['lambda_away'] = 0.0

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']

        if home_team in strengths and away_team in strengths:
            
            home_os = strengths[home_team]['OS']
            home_ds = strengths[home_team]['DS']
            away_os = strengths[away_team]['OS']
            away_ds = strengths[away_team]['DS']

            # Home Score = LeagueAvg * Home Offense * Away Defense (Multiplier)
            lambda_home = LEAGUE_AVG_POINTS_PER_GAME * home_os * away_ds 
            
            # Away Score = LeagueAvg * Away Offense * Home Defense (Multiplier)
            lambda_away = LEAGUE_AVG_POINTS_PER_GAME * away_os * home_ds

            df.loc[index, 'lambda_home'] = lambda_home
            df.loc[index, 'lambda_away'] = lambda_away

    print(f"[QEPC Lambda] Computed lambda values for {len(df)} games.")
    return df