
import pandas as pd
from typing import Optional, Dict
from qepc.autoload import paths

TEAM_STATS_FILENAME = "Team_Stats.csv"
REQUIRED_STATS_COLS = ["Team", "ORtg", "DRtg"] 

TEAM_NAME_MAP = {
    "76ers": "Philadelphia 76ers", "Bucks": "Milwaukee Bucks", "Bulls": "Chicago Bulls",
    "Cavaliers": "Cleveland Cavaliers", "Celtics": "Boston Celtics", "Clippers": "Los Angeles Clippers",
    "Nets": "Brooklyn Nets", "Rockets": "Houston Rockets", "Thunder": "Oklahoma City Thunder",
    "Warriors": "Golden State Warriors", "Lakers": "Los Angeles Lakers", "Hornets": "Charlotte Hornets",
    "Heat": "Miami Heat", "Magic": "Orlando Magic", "Hawks": "Atlanta Hawks", "Pistons": "Detroit Pistons",
    "Pelicans": "New Orleans Pelicans", "Grizzlies": "Memphis Grizzlies", "Wizards": "Washington Wizards",
    "Suns": "Phoenix Suns", "Kings": "Sacramento Kings", "Raptors": "Toronto Raptors",
    "Knicks": "New York Knicks", "Pacers": "Indiana Pacers", "Nuggets": "Denver Nuggets",
    "Jazz": "Utah Jazz", "Timberwolves": "Minnesota Timberwolves", "Trail Blazers": "Portland Trail Blazers",
    "Mavericks": "Dallas Mavericks", "Spurs": "San Antonio Spurs",
    "Bobcats": "Charlotte Hornets", "Bullets": "Washington Wizards",
    "New Jersey Nets": "Brooklyn Nets", "Seattle SuperSonics": "Oklahoma City Thunder",
    "SuperSonics": "Oklahoma City Thunder", "Vancouver Grizzlies": "Memphis Grizzlies",
}

def standardize_team_name(name):
    if not isinstance(name, str): return str(name)
    name = name.strip()
    if name in TEAM_NAME_MAP: return TEAM_NAME_MAP[name]
    for key, full_name in TEAM_NAME_MAP.items():
        if name.lower() == key.lower(): return full_name
    return name

def load_team_stats(reload=False):
    stats_path = paths.get_data_dir() / TEAM_STATS_FILENAME
    if not stats_path.exists():
        print(f"[QEPC Data Clean] ERROR: Team stats file not found at {stats_path}.")
        return None
    try:
        df = pd.read_csv(stats_path)
        for col in ['ORtg', 'DRtg']:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'Team' in df.columns:
            df['Team'] = df['Team'].apply(standardize_team_name)
        print(f"[QEPC Data Clean] Successfully loaded stats for {len(df)} teams.")
        return df
    except Exception as e:
        print(f"[QEPC Data Clean] Failed to load or process Team_Stats.csv: {e}")
        return None
