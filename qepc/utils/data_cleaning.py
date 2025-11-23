"""
QEPC Module: data_cleaning.py
Utility functions for loading and preprocessing statistical input files.
Includes canonical mapping for team names.
"""
import pandas as pd
from typing import Optional, Dict

from qepc.autoload import paths

# --- Configuration for Data Loading (Remains the same) ---
TEAM_STATS_FILENAME = "Team_Stats.csv"
REQUIRED_STATS_COLS = ["Team", "ORtg", "DRtg"] 

# --- CRITICAL FIX: CANONICAL TEAM NAME MAPPING ---
# Maps various historical, abbreviated, or mismatched names to the current full name.
TEAM_NAME_MAP: Dict[str, str] = {
    # Current Abbreviated Names from your raw data
    "76ers": "Philadelphia 76ers",
    "Bucks": "Milwaukee Bucks",
    "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers",
    "Celtics": "Boston Celtics",
    "Clippers": "Los Angeles Clippers",
    "Nets": "Brooklyn Nets",
    # Add other abbreviated names commonly found in your data
    "Rockets": "Houston Rockets",
    "Thunder": "Oklahoma City Thunder",
    "Warriors": "Golden State Warriors",
    "Lakers": "Los Angeles Lakers",
    "Hornets": "Charlotte Hornets",
    "Heat": "Miami Heat",
    "Magic": "Orlando Magic",
    "Hawks": "Atlanta Hawks",
    "Pistons": "Detroit Pistons",
    "Pelicans": "New Orleans Pelicans",
    "Grizzlies": "Memphis Grizzlies",
    "Wizards": "Washington Wizards",
    "Suns": "Phoenix Suns",
    "Kings": "Sacramento Kings",
    "Raptors": "Toronto Raptors",
    # Historical/Non-NBA names found in raw data
    "Bobcats": "Charlotte Hornets", # Historical name for Hornets
    "Bullets": "Washington Wizards", # Historical name for Wizards
    # Add others that may be present (Blackhawks/Braves are often historical team names)
    "Blackhawks": "Unknown/Historical",
    "Braves": "Atlanta Hawks", # Historical name for Hawks
    "Spurs": "San Antonio Spurs",
    "Mavericks": "Dallas Mavericks",
}

def standardize_team_name(name: str) -> str:
    """Standardizes a team name to the canonical full name."""
    name = name.strip()
    
    # Check if the name is an exact match in the mapping
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # Use a simpler lookup for robustness against slight case variations (by converting everything to title case)
    for key, full_name in TEAM_NAME_MAP.items():
        if name.title() == key.title():
            return full_name
            
    # Default: Return the name as is (e.g., if it's already the full name or an international team)
    return name


# --- (load_team_stats function remains the same below) ---
def load_team_stats(reload: bool = False) -> Optional[pd.DataFrame]:
    """Loads and cleans the canonical team statistics from Team_Stats.csv."""
    stats_path = paths.get_data_dir() / TEAM_STATS_FILENAME
    if not stats_path.exists():
        print(f"[QEPC Data Clean] ERROR: Team stats file not found at {stats_path}. Cannot proceed with modeling.")
        return None
    try:
        df = pd.read_csv(stats_path)
        if not all(col in df.columns for col in REQUIRED_STATS_COLS):
            missing = [col for col in REQUIRED_STATS_COLS if col not in df.columns]
            print(f"[QEPC Data Clean] ERROR: Missing required columns in Team Stats: {missing}")
            return None
        for col in ['ORtg', 'DRtg']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=REQUIRED_STATS_COLS, inplace=True)
        df['Team'] = df['Team'].apply(standardize_team_name) # Apply standardization here too
        print(f"[QEPC Data Clean] Successfully loaded and cleaned stats for {len(df)} teams.")
        return df[REQUIRED_STATS_COLS]
    except Exception as e:
        print(f"[QEPC Data Clean] Failed to load or process Team_Stats.csv: {e}")
        return None