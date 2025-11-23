"""
QEPC Module: lambda_engine.py
Core modeling component for calculating Poisson rates (lambda).
Includes HCA, Pace, and Volatility passing.
"""
import pandas as pd
from typing import Dict

# --- Configuration ---
LEAGUE_AVG_POINTS_PER_GAME = 110.0 
HCA_MULTIPLIER = 1.028 

def compute_lambda(
    schedule_df: pd.DataFrame, 
    team_stats_df: pd.DataFrame 
) -> pd.DataFrame:
    """
    Applies Team Strength Ratings (TSR), Pace, and Volatility.
    """
    if team_stats_df.empty: 
        print("[QEPC Lambda] ERROR: Cannot compute lambda without team strengths.")
        return schedule_df

    # 1. Calculate League Averages
    la_ortg = team_stats_df['ORtg'].mean()
    la_pace = team_stats_df['Pace'].mean()

    strengths = {}
    for index, row in team_stats_df.iterrows():
        team = row['Team']
        
        offensive_strength = row['ORtg'] / la_ortg
        defensive_strength = row['DRtg'] / la_ortg 
        pace_rating = row['Pace'] / la_pace
        
        # CHAOS FACTOR: Extract Volatility (Standard Deviation of Points)
        # If missing, assume standard NBA deviation (~12.0)
        volatility = row.get('Volatility', 12.0)

        strengths[team] = {
            'OS': offensive_strength,
            'DS': defensive_strength,
            'Pace': pace_rating,
            'Vol': volatility
        }
    
    # --- Calculation Loop ---
    df = schedule_df.copy()
    df['lambda_home'] = 0.0
    df['lambda_away'] = 0.0
    # Create columns to store volatility for the simulator
    df['vol_home'] = 0.0
    df['vol_away'] = 0.0

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']

        if home_team in strengths and away_team in strengths:
            
            home = strengths[home_team]
            away = strengths[away_team]

            # Pace Entanglement
            game_pace_multiplier = home['Pace'] * away['Pace']

            # Lambda Calculation
            lambda_home = (LEAGUE_AVG_POINTS_PER_GAME * home['OS'] * away['DS'] * game_pace_multiplier) * HCA_MULTIPLIER
            lambda_away = (LEAGUE_AVG_POINTS_PER_GAME * away['OS'] * home['DS'] * game_pace_multiplier)

            # Store values
            df.loc[index, 'lambda_home'] = lambda_home
            df.loc[index, 'lambda_away'] = lambda_away
            # Store Volatility for the Simulator
            df.loc[index, 'vol_home'] = home['Vol']
            df.loc[index, 'vol_away'] = away['Vol']

    print(f"[QEPC Lambda] Computed lambda & volatility for {len(df)} games.")
    return df
