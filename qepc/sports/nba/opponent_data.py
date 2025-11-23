"""
QEPC Module: opponent_data.py
Processes raw team statistics to calculate Defensive (Opponent) metrics.
Includes Time Decay and Date Filtering for higher accuracy.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qepc.autoload import paths
from qepc.utils.data_cleaning import standardize_team_name 

# --- Configuration ---
RAW_TEAM_STATS_FILE = "TeamStatistics.csv"
FTA_MULTIPLIER = 0.44 

# ACCURACY SETTINGS
DAYS_HISTORY_WINDOW = 730  # Look back 2 years (approx 2 seasons)
DECAY_LAMBDA = 0.002       # Decay rate (higher = more recency bias)

def apply_time_decay(df: pd.DataFrame, date_col: str = 'gameDate') -> pd.DataFrame:
    """
    Applies exponential time decay weights to the data.
    Recent games get Weight ~ 1.0, older games get Weight -> 0.0
    """
    # Ensure date is datetime
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
    
    # 1. Filter by Date Window
    cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=DAYS_HISTORY_WINDOW)
    df = df[df[date_col] >= cutoff_date].copy()
    
    if df.empty:
        return df
        
    # 2. Calculate Days Ago
    current_time = pd.Timestamp.now(tz='UTC')
    df['DaysAgo'] = (current_time - df[date_col]).dt.days
    
    # 3. Calculate Weight: e^(-lambda * days_ago)
    df['Weight'] = np.exp(-DECAY_LAMBDA * df['DaysAgo'])
    
    return df

def load_and_process_opponent_data() -> pd.DataFrame:
    """
    Loads raw team statistics, applies time decay, and calculates Weighted DRtg.
    """
    print("[QEPC Opponent Processor] Loading raw team data for Weighted DRtg...") 

    raw_path = paths.get_data_dir() / "raw" / RAW_TEAM_STATS_FILE
    
    if not raw_path.exists():
        print(f"[QEPC Opponent Processor] ERROR: Raw file not found at {raw_path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(raw_path)
        
        # 1. Standardization
        df = df.rename(columns={
            'teamName': 'TeamName',
            'opponentTeamName': 'OpponentName',
            'teamScore': 'PointsScored',
            'opponentScore': 'PointsAllowed',
            'reboundsOffensive': 'OR', 
            'fieldGoalsAttempted': 'FGA', 
            'turnovers': 'TO', 
            'freeThrowsAttempted': 'FTA', 
            'gameId': 'GameId',
            'gameDate': 'gameDate'
        })
        
        # 2. Apply Accuracy Filters (Date & Decay)
        df = apply_time_decay(df, date_col='gameDate')
        
        if df.empty:
            print("[QEPC Opponent Processor] WARNING: No data found within history window.")
            return pd.DataFrame()

        # 3. Calculate Team Possessions (Per Game)
        df['TeamPossessions'] = df['FGA'] - df['OR'] + df['TO'] + (FTA_MULTIPLIER * df['FTA'])
        
        # 4. Create Opponent View (Flipping perspective)
        # We want to know what the Opponent (Team) ALLOWED.
        # 'Team' below is the Opponent Name in the original row.
        opponent_view = pd.DataFrame({
            'Team': df['OpponentName'],
            'PointsAllowed': df['PointsScored'],  
            'PossessionsAllowed': df['TeamPossessions'],
            'Weight': df['Weight'] # Carry over the recency weight
        })
        
        # 5. Calculate Weighted Averages
        # Weighted Avg = Sum(Value * Weight) / Sum(Weight)
        
        opponent_view['Wt_PointsAllowed'] = opponent_view['PointsAllowed'] * opponent_view['Weight']
        opponent_view['Wt_PossessionsAllowed'] = opponent_view['PossessionsAllowed'] * opponent_view['Weight']
        
        df_metrics = opponent_view.groupby('Team').agg(
            Sum_Wt_Points=('Wt_PointsAllowed', 'sum'),
            Sum_Wt_Possessions=('Wt_PossessionsAllowed', 'sum'),
            Sum_Weights=('Weight', 'sum'),
            GamesPlayed=('PointsAllowed', 'count')
        ).reset_index()

        # Filter small samples
        df_metrics = df_metrics[df_metrics['GamesPlayed'] >= 5].copy()

        # 6. Calculate Final Weighted DRtg
        # Normalized per 100 possessions
        # Formula: (SumWeightedPoints / SumWeights) / (SumWeightedPoss / SumWeights) * 100
        # Simplifies to: SumWeightedPoints / SumWeightedPoss * 100
        df_metrics['DRtg'] = (df_metrics['Sum_Wt_Points'] / df_metrics['Sum_Wt_Possessions']) * 100
        
        # Standardize Names
        df_metrics['Team'] = df_metrics['Team'].apply(standardize_team_name)
        
        print(f"[QEPC Opponent Processor] Calculated Weighted DRtg for {len(df_metrics)} teams.")
        return df_metrics[['Team', 'DRtg']]

    except Exception as e:
        print(f"❌ [QEPC Opponent Processor] CRITICAL FAILURE: {e}")
        return pd.DataFrame()
