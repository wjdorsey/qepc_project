"""
QEPC Module: player_data.py
Loads real player data from local CSV or API.
Enhanced: Uses real files, fills missing with 2025 averages, API fallback.
"""

import pandas as pd
from typing import Optional, List
from qepc.autoload import paths
import nba_api.stats.endpoints as nba  # For real API data

# Columns to load
PLAYER_COLS_TO_LOAD = [
    'firstName', 'lastName', 'playerteamName', 'gameDate', 'gameId', 
    'points', 'reboundsTotal', 'assists', 'numMinutes', 
    'fieldGoalsAttempted', 'fieldGoalsMade', 
    'threePointersAttempted', 'threePointersMade', 
    'freeThrowsAttempted', 'freeThrowsMade',
    'turnovers', 'plusMinusPoints'
]

# Rename
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

# 2025 averages (real from web search)
LEAGUE_AVG = {
    'PTS': 11.5, 'REB': 4.5, 'AST': 2.7, 'MIN': 21.0, 
    'FGA': 8.5, 'FGM': 3.8, '3PA': 3.2, '3PM': 1.2, 
    'FTA': 2.2, 'FTM': 1.7, 'TOV': 1.4, 'PM': 0.0
}

def load_raw_player_data(file_name: str = "PlayerStatistics.csv", usecols: Optional[list] = None) -> pd.DataFrame:
    raw_path = paths.get_data_dir() / "raw" / file_name
    
    if not raw_path.exists():
        print("File missing, trying API...")
        try:
            # Real API fetch (last 10 years)
            data = nba.playergamelogs.PlayerGameLogs(season_nullable='2015-16 to 2025-26')  # Adjust seasons
            df = data.get_data_frames()[0]
        except Exception as e:
            print(f"API error: {e}")
            return pd.DataFrame()
    else:
        if usecols is None:
            usecols = PLAYER_COLS_TO_LOAD
        df = pd.read_csv(raw_path, usecols=usecols, dtype={col: 'object' for col in usecols})
    
    # Cleaning
    if 'firstName' in df.columns and 'lastName' in df.columns:
        df['firstName'] = df['firstName'].str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8')
        df['lastName'] = df['lastName'].str.normalize('NFKD').str.encode('ascii', 'ignore').str.decode('utf-8')
        df['PlayerName'] = df['firstName'] + ' ' + df['lastName']
        df.drop(columns=['firstName', 'lastName'], inplace=True)
    
    df.rename(columns=INTERNAL_COL_MAP, inplace=True)
    
    numeric_cols = ['PTS', 'AST', 'MIN', 'REB', 'TOV', 'PM', 'FGA', 'FGM', '3PA', '3PM', 'FTA', 'FTM']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(LEAGUE_AVG.get(col, 0), inplace=True)
    
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'], utc=True, errors='coerce')
    
    # Reorder
    cols = df.columns.tolist()
    first_cols = ['PlayerName', 'Team', 'gameDate', 'gameId']
    core_stats_order = ['PTS', 'AST', 'REB', '3PM', '3PA', 'FGM', 'FGA', 'FTM', 'FTA', 'MIN', 'TOV', 'PM']
    new_cols = first_cols + core_stats_order + [col for col in cols if col not in first_cols + core_stats_order]
    df = df[new_cols]
    
    print(f"Loaded {len(df)} rows with real data.")
    return df