"""
QEPC Module: sim.py
Handles NBA schedule loading and game-day filtering.
"""
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Union

from qepc.autoload import paths

# --- Configuration ---
# Original format columns (used when show='clean')
QEPC_SCHEDULE_COLS = ["Date", "Time", "Away Team", "Home Team"]
# VERIFIED FORMAT: MM/DD/YYYY h:MM PM/AM (slashes and space)
DATE_PARSE_FORMAT = "%m/%d/%Y %I:%M %p" 

_NBA_SCHEDULE_CACHE: Optional[pd.DataFrame] = None

def load_nba_schedule(reload: bool = False) -> Optional[pd.DataFrame]:
    """Loads the master NBA schedule (Games.csv), cleans it up, and caches it."""
    global _NBA_SCHEDULE_CACHE
    if _NBA_SCHEDULE_CACHE is not None and not reload:
        print("[QEPC NBA Sim] Schedule loaded from cache.")
        return _NBA_SCHEDULE_CACHE

    games_path = paths.get_games_path() 

    if not games_path.exists():
        print(f"[QEPC NBA Sim] ERROR: Games.csv not found at {games_path}")
        return None

    try:
        df = pd.read_csv(games_path)
        
        # --- ORIGINAL TWO-COLUMN PARSING LOGIC RESTORED ---
        # Combine the original 'Date' and 'Time' columns and parse them using the verified format.
        df["gameDate"] = pd.to_datetime(
            df["Date"].astype(str) + " " + df["Time"].astype(str),
            format=DATE_PARSE_FORMAT,
            errors="coerce",
        )
        # --- END RESTORATION ---

        # If we failed to parse all dates, throw an error
        if df["gameDate"].isnull().all() or df["gameDate"].empty:
            raise ValueError("All dates failed to parse or schedule is empty after cleaning.")
            
        # Ensure we keep the original Date and Time columns for the final clean output
        df.dropna(subset=["gameDate"], inplace=True)

        print(f"[QEPC NBA Sim] Successfully loaded and parsed {len(df)} games from original format.")
        _NBA_SCHEDULE_CACHE = df.copy()
        return df
    
    except Exception as e:
        # Verbose error printout for final debugging
        print(f"[QEPC NBA Sim] FAILED TO PROCESS SCHEDULE: {e}")
        return None

def _filter_and_clean_schedule(df: pd.DataFrame, show: str) -> Union[pd.DataFrame, None]:
    """Helper to apply 'clean' or 'raw' display logic as required by the QEPC spec."""
    if df.empty:
        return None
        
    if show == 'clean':
        return df[QEPC_SCHEDULE_COLS]
    elif show == 'raw':
        return df
    else:
        print(f"[QEPC NBA Sim] Invalid 'show' argument: {show}. Returning raw.")
        return df

def get_today_games(show: str = 'clean') -> Optional[pd.DataFrame]:
    """Returns today's NBA schedule based on your local date."""
    schedule = load_nba_schedule()
    if schedule is None: return None
    
    today = date.today()
    today_games = schedule[schedule["gameDate"].dt.date == today]
    
    return _filter_and_clean_schedule(today_games, show)

def get_tomorrow_games(show: str = 'clean') -> Optional[pd.DataFrame]:
    """Returns tomorrow's NBA schedule."""
    schedule = load_nba_schedule()
    if schedule is None: return None
    
    tomorrow = date.today() + timedelta(days=1)
    tomorrow_games = schedule[schedule["gameDate"].dt.date == tomorrow]
    
    return _filter_and_clean_schedule(tomorrow_games, show)

def get_upcoming_games(days: int = 7, show: str = 'clean') -> Optional[pd.DataFrame]:
    """Returns games scheduled within the next 'days' (starting from tomorrow)."""
    schedule = load_nba_schedule()
    if schedule is None: return None
    
    today = date.today()
    end_date = today + timedelta(days=days)
    
    upcoming_games = schedule[
        (schedule["gameDate"].dt.date > today) & 
        (schedule["gameDate"].dt.date <= end_date)
    ]
    
    return _filter_and_clean_schedule(upcoming_games, show)