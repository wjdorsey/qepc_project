"""
QEPC Module: opponent_data.py
Processes raw team statistics to calculate Defensive (Opponent) metrics.
"""
import pandas as pd
import numpy as np
from qepc.autoload import paths
from qepc.utils.data_cleaning import standardize_team_name # <--- NEW IMPORT

# --- Configuration ---
RAW_TEAM_STATS_FILE = "TeamStatistics.csv"
FTA_MULTIPLIER = 0.44 

def load_and_process_opponent_data() -> pd.DataFrame:
    """
    Loads raw team statistics and flips the perspective to create defensive metrics.
    Calculates Opponent Possessions and Points Allowed.
    """
    print("[QEPC Opponent Processor] Loading raw team data for DRtg calculation...") 

    raw_path = paths.get_data_dir() / "raw" / RAW_TEAM_STATS_FILE
    
    if not raw_path.exists():
        print(f"[QEPC Opponent Processor] ERROR: Raw file not found at {raw_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(raw_path)
        
        # 1. Select and rename columns to standardize names
        df = df.rename(columns={
            'teamName': 'TeamName',
            'opponentTeamName': 'OpponentName',
            'teamScore': 'PointsScored',
            'opponentScore': 'PointsAllowed',
            'reboundsOffensive': 'OR', 
            'fieldGoalsAttempted': 'FGA', 
            'turnovers': 'TO', 
            'freeThrowsAttempted': 'FTA', 
            'gameId': 'GameId' 
        })
        
        # 2. Calculate Team Possessions (Offensive Possessions)
        df['TeamPossessions'] = df['FGA'] - df['OR'] + df['TO'] + (FTA_MULTIPLIER * df['FTA'])
        
        # 3. Create the OPPONENT View
        opponent_view = pd.DataFrame({
            'Team': df['OpponentName'],
            'PointsAllowed': df['PointsScored'],  
            'PossessionsAllowed': df['TeamPossessions'],
            'GameId': df['GameId']
        })
        
        # 4. Aggregate to get Total Defensive Metrics (Per Team)
        df_defensive_metrics = opponent_view.groupby('Team').agg(
            TotalPointsAllowed=('PointsAllowed', 'sum'),
            TotalPossessionsAllowed=('PossessionsAllowed', 'sum'),
            GamesPlayed=('GameId', 'nunique') 
        ).reset_index()

        # 5. Normalize Totals to Per-Game
        df_defensive_metrics['PointsAllowedPerGame'] = df_defensive_metrics['TotalPointsAllowed'] / df_defensive_metrics['GamesPlayed']
        df_defensive_metrics['PossessionsAllowedPerGame'] = df_defensive_metrics['TotalPossessionsAllowed'] / df_defensive_metrics['GamesPlayed']
        
        # 6. Calculate DRtg
        df_defensive_metrics['DRtg'] = (df_defensive_metrics['PointsAllowedPerGame'] / df_defensive_metrics['PossessionsAllowedPerGame']) * 100
        
        # CRITICAL FIX: Standardize Team Names to ensure merge match
        df_defensive_metrics['Team'] = df_defensive_metrics['Team'].apply(standardize_team_name)
        
        print(f"[QEPC Opponent Processor] Calculated DRtg metrics for {len(df_defensive_metrics)} teams.")
        return df_defensive_metrics[['Team', 'DRtg']]

    except Exception as e:
        print(f"❌ [QEPC Opponent Processor] CRITICAL FAILURE during processing: {e}")
        return pd.DataFrame()