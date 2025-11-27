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
# UPDATED: Use your canonical game-level team stats file in data/raw
RAW_TEAM_STATS_FILE = "Team_Stats.csv"
FTA_MULTIPLIER = 0.44 

# ACCURACY SETTINGS
HISTORY_DAYS = 730        # Look back up to 2 seasons
MIN_GAMES_PER_TEAM = 5    # Minimum games required for stable DRtg
HALF_LIFE_DAYS = 60       # Time decay half-life (recent games matter more)


# ---------------------------------------------------------------------------
# Time Decay Helper
# ---------------------------------------------------------------------------
def apply_time_decay(df: pd.DataFrame, date_col: str = "gameDate") -> pd.DataFrame:
    """
    Apply exponential time decay weights based on gameDate.
    Newer games get higher weight.
    """
    if date_col not in df.columns:
        raise ValueError(f"Expected date column '{date_col}' in DataFrame")

    df = df.copy()
    
    # Ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df = df.dropna(subset=[date_col])

    # Filter to history window
    cutoff_date = pd.Timestamp.utcnow() - pd.Timedelta(days=HISTORY_DAYS)
    df = df[df[date_col] >= cutoff_date].copy()

    if df.empty:
        print("[QEPC Opponent Processor] WARNING: No games within history window.")
        return df

    # Time difference in days
    df["DaysAgo"] = (pd.Timestamp.utcnow() - df[date_col]).dt.days

    # Exponential decay: weight = 0.5 ** (DaysAgo / HALF_LIFE_DAYS)
    df["Weight"] = 0.5 ** (df["DaysAgo"] / HALF_LIFE_DAYS)

    return df


# ---------------------------------------------------------------------------
# Core Opponent (Defensive) Metrics Processor
# ---------------------------------------------------------------------------
def load_and_process_opponent_data() -> pd.DataFrame:
    """
    Load raw team game stats and compute Weighted Defensive Rating (DRtg) 
    from the opponent perspective.
    
    Returns a DataFrame with:
        Team, DRtg
    """
    print("[QEPC Opponent Processor] Loading raw team data for Weighted DRtg...") 

    # NOTE: data/raw/Team_Stats.csv is your canonical game-level team stats file
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
        
        # 3. Possessions and Opponent Perspective
        #    Standard NBA possessions formula (approx.)
        df['Possessions'] = (
            df['FGA']
            + FTA_MULTIPLIER * df['FTA']
            - df['OR']
            + df['TO']
        )
        
        # Filter any non-sensical values
        df = df[(df['Possessions'] > 0) & (df['PointsAllowed'] >= 0)].copy()
        if df.empty:
            print("[QEPC Opponent Processor] WARNING: No valid rows after possessions filter.")
            return pd.DataFrame()
        
        # Build opponent perspective: i.e. what each Team allowed to Opponent
        # Here, we treat "TeamName" as the defensive team, so PointsAllowed
        # and Possessions become the basis of DRtg.
        opponent_view = df.copy()
        opponent_view['Team'] = opponent_view['TeamName']  # rename key
        
        # Weighted points and possessions allowed
        opponent_view['Wt_PointsAllowed'] = opponent_view['PointsAllowed'] * opponent_view['Weight']
        opponent_view['Wt_PossessionsAllowed'] = opponent_view['Possessions'] * opponent_view['Weight']
        
        # 4. Aggregate to team-level metrics
        df_metrics = opponent_view.groupby('Team').agg(
            Sum_Wt_Points=('Wt_PointsAllowed', 'sum'),
            Sum_Wt_Possessions=('Wt_PossessionsAllowed', 'sum'),
            Sum_Weights=('Weight', 'sum'),
            GamesPlayed=('PointsAllowed', 'count')
        ).reset_index()

        # Filter small samples
        df_metrics = df_metrics[df_metrics['GamesPlayed'] >= MIN_GAMES_PER_TEAM].copy()
        if df_metrics.empty:
            print("[QEPC Opponent Processor] WARNING: No teams met MIN_GAMES_PER_TEAM requirement.")
            return pd.DataFrame()
        
        # 5. Compute Defensive Rating (points allowed per 100 possessions)
        df_metrics['DRtg'] = (
            df_metrics['Sum_Wt_Points'] / df_metrics['Sum_Wt_Possessions']
        ) * 100
        
        # Standardize Names
        df_metrics['Team'] = df_metrics['Team'].apply(standardize_team_name)
        
        print(f"[QEPC Opponent Processor] Calculated Weighted DRtg for {len(df_metrics)} teams.")
        return df_metrics[['Team', 'DRtg']]

    except Exception as e:
        print(f"❌ [QEPC Opponent Processor] CRITICAL FAILURE: {e}")
        return pd.DataFrame()
