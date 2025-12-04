"""Data Loader for QEPC v2"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(r"C:\Users\wdors\qepc_project\experimental\CLAUDE_REWRITE")
DATA_DIR = BASE_DIR / "data"

def load_player_logs():
    """Load all player game logs"""
    path = DATA_DIR / "raw" / "player_logs" / "all_seasons.csv"
    return pd.read_csv(path)

def load_team_games():
    """Load team game data"""
    path = DATA_DIR / "raw" / "team_stats" / "team_games.csv"
    return pd.read_csv(path)

def save_predictions(df, filename):
    """Save predictions to logs"""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d")
    path = DATA_DIR / "logs" / "daily_predictions" / f"{timestamp}_{filename}"
    df.to_csv(path, index=False)
    return path
