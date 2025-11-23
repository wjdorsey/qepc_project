"""
QEPC Module: player_data.py
Handles memory-safe loading of large, raw player data files.
"""
import pandas as pd
from typing import Optional, List
from qepc.autoload import paths

# --- Configuration ---
# CRITICAL FIX: 'gameId' is included for aggregation
PLAYER_COLS_TO_LOAD = [
    'firstName', 'lastName', 'playerteamName', 'gameDate', 'gameId', 
    'points', 'reboundsTotal', 'assists', 'numMinutes', 
    'fieldGoalsAttempted', 'fieldGoalsMade', 
    'threePointersAttempted', 'threePointersMade', 
    'freeThrowsAttempted', 'freeThrowsMade',
    'turnovers', 'plusMinusPoints'
]

# Renaming map for internal QEPC consistency
INTERNAL_COL_MAP = {
    'playerteamName': 'Team',
    'reboundsTotal': 'REB', 
    'numMinutes': 'MIN',    
    'points': 'PTS',
    'assists': 'AST',
    'turnovers': 'TOV',
    'plusMinusPoints': 'PM',
    'fieldGoalsAttempted': 'FGA',
    'fieldGoalsMade': 'FGM',
    'threePointersAttempted': '3PA',
    'threePointersMade': '3PM',
    'freeThrowsAttempted': 'FTA',
    'freeThrowsMade': 'FTM',
}

def load_raw_player_data(
    file_name: str = "PlayerStatistics.csv", 
    usecols: Optional[list] = None
) -> pd.DataFrame:
    
    raw_path = paths.get_data_dir() / "raw" / file_name
    
    if not raw_path.exists():
        print(f"[QEPC PlayerData] ERROR: Raw file not found at {raw_path}")
        return pd.DataFrame()

    try:
        if usecols is None:
            usecols = PLAYER_COLS_TO_LOAD
            
        df = pd.read_csv(raw_path, usecols=usecols, dtype={col: 'object' for col in usecols})
        
        # --- POST-LOAD CLEANING AND STANDARDIZATION ---
        
        if 'firstName' in df.columns and 'lastName' in df.columns:
            df['PlayerName'] = df['firstName'].astype(str) + ' ' + df['lastName'].astype(str)
            df.drop(columns=['firstName', 'lastName'], inplace=True)
            
        df.rename(columns=INTERNAL_COL_MAP, inplace=True)
        
        numeric_cols = ['PTS', 'AST', 'MIN', 'REB', 'TOV', 'PM', 'FGA', 'FGM', '3PA', '3PM', 'FTA', 'FTM']
        for col in numeric_cols:
             if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'gameDate' in df.columns:
             df['gameDate'] = pd.to_datetime(df['gameDate'], utc=True, errors='coerce') 
        
        # 5. FINAL FIX: Reorder columns for usability 
        cols = df.columns.tolist()
        
        first_cols = ['PlayerName', 'Team', 'gameDate', 'gameId'] # gameId is included here
        core_stats_order = ['PTS', 'AST', 'REB', '3PM', '3PA', 'FGM', 'FGA', 'FTM', 'FTA', 'MIN', 'TOV', 'PM']
        
        # Ensure gameId is included in the final order
        all_fixed_cols = first_cols + core_stats_order
        other_cols = [col for col in cols if col not in all_fixed_cols]
        new_cols = first_cols + core_stats_order + other_cols
        
        df = df[new_cols] 

        # ---------------------------------------------
        
        print(f"[QEPC PlayerData] Successfully loaded {len(df)} rows from {file_name}.")
        return df
        
    except Exception as e:
        print(f"[QEPC PlayerData] CRITICAL FAILURE: Could not safely load {file_name}.")
        print(f"Error: {e}")
        return pd.DataFrame()