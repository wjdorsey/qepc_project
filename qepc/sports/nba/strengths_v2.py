"""
QEPC Module: strengths_v2.py
Calculates robust team strength ratings (ORtg/DRtg) from raw player game data
and integrates true DRtg from opponent data.
"""
import pandas as pd
import numpy as np
from typing import Dict
from qepc.sports.nba.player_data import load_raw_player_data
from qepc.sports.nba.opponent_data import load_and_process_opponent_data 
from qepc.utils.data_cleaning import standardize_team_name 

# --- Configuration ---
FTA_MULTIPLIER = 0.44 
MIN_GAMES_PLAYED = 10 # Filter threshold to remove statistical noise

def calculate_advanced_strengths() -> pd.DataFrame:
    """
    Loads raw data, calculates ORtg (Offensive Rating) internally, 
    gets DRtg (Defensive Rating) from the opponent processor, and merges them.
    """
    print("[QEPC Strength V2] Starting Advanced Strength Calculation...") 

    # --- PART 1: CALCULATE ORTG (Requires Player Data) ---
    try:
        # Note: We assume 'gameId' is available from player_data module
        raw_df = load_raw_player_data(file_name="PlayerStatistics.csv")
    except Exception as e:
        print(f"[QEPC Strength V2] ERROR loading player data: {e}")
        return pd.DataFrame()
    
    if raw_df.empty:
        print("[QEPC Strength V2] ERROR: Player data is empty. Cannot calculate ORtg.")
        return pd.DataFrame()

    raw_df['Team'] = raw_df['Team'].apply(standardize_team_name)

    # 1. Aggregate statistics per team (Now including gameId count for normalization)
    team_stats_offense = raw_df.groupby('Team').agg(
        TotalPoints=('PTS', 'sum'),
        TotalFGA=('FGA', 'sum'),
        TotalOR=('REB', 'sum'), 
        TotalTO=('TOV', 'sum'),
        TotalFTA=('FTA', 'sum'),
        GamesPlayed=('gameId', 'nunique') # <--- CRITICAL: Count unique games played
    ).reset_index()
    
    if team_stats_offense.empty:
        print("[QEPC Strength V2] ERROR: Aggregation resulted in empty table.")
        return pd.DataFrame()

    # 2. Calculate Possessions
    team_stats_offense['Possessions'] = (
        team_stats_offense['TotalFGA'] 
        - team_stats_offense['TotalOR'] 
        + team_stats_offense['TotalTO'] 
        + (FTA_MULTIPLIER * team_stats_offense['TotalFTA'])
    )
    
    # CRITICAL FIX 1: Filter out teams with insufficient games played
    team_stats_offense = team_stats_offense[
        team_stats_offense['GamesPlayed'] >= MIN_GAMES_PLAYED
    ].copy()
    
    if team_stats_offense.empty:
        print("[QEPC Strength V2] CRITICAL ERROR: All teams filtered out by game count threshold.")
        return pd.DataFrame()
        
    # 3. Normalize Totals to Per-Game (CRITICAL STEP 2)
    team_stats_offense['PointsPerGame'] = team_stats_offense['TotalPoints'] / team_stats_offense['GamesPlayed']
    team_stats_offense['PossessionsPerGame'] = team_stats_offense['Possessions'] / team_stats_offense['GamesPlayed']
    
    # 4. Calculate ORtg (Now uses Averages)
    team_stats_offense['ORtg'] = (team_stats_offense['PointsPerGame'] / team_stats_offense['PossessionsPerGame']) * 100
    
    df_ortg = team_stats_offense[['Team', 'ORtg']].copy()

    # --- PART 2 & 3: CALCULATE DRTG AND MERGE ---
    df_drtg = load_and_process_opponent_data()

    if df_drtg.empty:
        print("[QEPC Strength V2] WARNING: DRtg calculation failed. Using placeholder DRtg.")
        LA_ORtg = df_ortg['ORtg'].mean()
        df_ortg['DRtg'] = LA_ORtg
        return df_ortg[['Team', 'ORtg', 'DRtg']]

    final_strengths = pd.merge(
        df_ortg, 
        df_drtg, 
        on='Team', 
        how='inner'
    )
    
    print(f"[QEPC Strength V2] Calculated FINAL strengths for {len(final_strengths)} teams.")
    return final_strengths[['Team', 'ORtg', 'DRtg']]