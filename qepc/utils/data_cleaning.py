import pandas as pd
from typing import Optional, Dict
from qepc.autoload import paths

# We now rely on the NBA data_source helper, which in turn
# uses strengths_v2 + data/raw/Team_Stats.csv when needed.
from qepc.sports.nba.data_source import load_team_stats as _load_team_stats_source

# ---------------------------------------------------------------------------
# Team name standardization
# ---------------------------------------------------------------------------

TEAM_NAME_MAP: Dict[str, str] = {
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
    # Legacy / historical names mapped to modern franchises
    "Bobcats": "Charlotte Hornets", "Bullets": "Washington Wizards",
    "New Jersey Nets": "Brooklyn Nets", "Seattle SuperSonics": "Oklahoma City Thunder",
    "SuperSonics": "Oklahoma City Thunder", "Vancouver Grizzlies": "Memphis Grizzlies",
}


def standardize_team_name(name) -> str:
    """
    Normalize a variety of team name/abbreviation forms to a canonical string.

    This is used throughout QEPC (e.g. in opponent_data, diagnostics, etc.).
    """
    if not isinstance(name, str):
        return str(name)

    name = name.strip()
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]

    # Case-insensitive lookup
    for key, full_name in TEAM_NAME_MAP.items():
        if name.lower() == key.lower():
            return full_name

    return name


# ---------------------------------------------------------------------------
# Team stats loader (now delegates to data_source)
# ---------------------------------------------------------------------------

def load_team_stats(reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load team-level ratings (Team, ORtg, DRtg).

    Previously this read directly from data/Team_Stats.csv.
    Now it delegates to qepc.sports.nba.data_source.load_team_stats(),
    which will:
      - Prefer nba_api if available & allowed, OR
      - Fall back to strengths_v2 + data/raw/Team_Stats.csv.

    We then standardize the Team names and ensure ORtg/DRtg are numeric.
    """
    try:
        df = _load_team_stats_source()
    except Exception as e:
        print(f"[QEPC Data Clean] Failed to load team stats via data_source: {e}")
        return None

    if df is None or df.empty:
        print("[QEPC Data Clean] ERROR: No team stats data returned.")
        return None

    # Ensure required columns exist
    for col in ("Team", "ORtg", "DRtg"):
        if col not in df.columns:
            print(f"[QEPC Data Clean] ERROR: Missing expected column '{col}' in team stats.")
            return None

    # Clean numeric fields
    for col in ("ORtg", "DRtg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Standardize team names
    df["Team"] = df["Team"].apply(standardize_team_name)

    print(f"[QEPC Data Clean] Successfully loaded & normalized stats for {len(df)} teams.")
    return df
