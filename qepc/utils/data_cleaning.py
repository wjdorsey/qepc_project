import pandas as pd
from typing import Dict, Optional

from qepc.autoload import paths

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
    
    Parameters
    ----------
    name : str or other
        Team name or abbreviation to standardize
        
    Returns
    -------
    str
        Canonical team name (e.g. "Boston Celtics")
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
# Team stats loader (with fallback logic)
# ---------------------------------------------------------------------------

def load_team_stats(reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load team-level ratings (Team, ORtg, DRtg, Pace, Volatility).

    Uses strengths_v2 which handles:
      - Local CSV data with recency weighting
      - Live NBA API overlay when available
      - Multiple fallback sources

    Parameters
    ----------
    reload : bool, optional
        If True, forces reload (not currently used, kept for compatibility)

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame with columns: Team, ORtg, DRtg, Pace, Volatility, SOS
        Returns None if loading fails
    """
    try:
        # Import here to avoid circular imports
        from qepc.sports.nba.strengths_v2 import get_team_strengths
        
        # Load team strengths with auto source selection
        df = get_team_strengths(source="auto", verbose=False)
        
    except ImportError as e:
        print(f"[QEPC Data Clean] ⚠️ Could not import strengths_v2: {e}")
        return None
    except Exception as e:
        print(f"[QEPC Data Clean] ⚠️ Failed to load team stats: {e}")
        return None

    # Validate we got data
    if df is None or df.empty:
        print("[QEPC Data Clean] ❌ ERROR: No team stats data returned.")
        return None

    # Ensure required columns exist
    required_cols = ["Team", "ORtg", "DRtg"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"[QEPC Data Clean] ❌ ERROR: Missing expected columns: {missing_cols}")
        return None

    # Clean numeric fields
    for col in ["ORtg", "DRtg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Check for NaN values
        if df[col].isna().any():
            print(f"[QEPC Data Clean] ⚠️ Warning: {col} has NaN values, filling with league average")
            df[col].fillna(110.0, inplace=True)

    # Standardize team names
    df["Team"] = df["Team"].apply(standardize_team_name)

    print(f"[QEPC Data Clean] ✅ Successfully loaded & normalized stats for {len(df)} teams.")
    return df


def get_team_strength(team_name: str, team_stats_df: Optional[pd.DataFrame] = None) -> Optional[Dict]:
    """
    Get strength metrics for a single team.
    
    Parameters
    ----------
    team_name : str
        Team name (will be standardized)
    team_stats_df : Optional[pd.DataFrame]
        Pre-loaded team stats. If None, will call load_team_stats()
        
    Returns
    -------
    Optional[Dict]
        Dictionary with team metrics (ORtg, DRtg, Pace, Volatility, etc.)
        Returns None if team not found
    """
    if team_stats_df is None:
        team_stats_df = load_team_stats()
    
    if team_stats_df is None or team_stats_df.empty:
        return None
    
    # Standardize the input name
    standardized_name = standardize_team_name(team_name)
    
    # Look for the team
    team_row = team_stats_df[team_stats_df["Team"] == standardized_name]
    
    if team_row.empty:
        print(f"[QEPC Data Clean] ⚠️ Team '{standardized_name}' not found in stats")
        return None
    
    # Convert to dictionary
    return team_row.iloc[0].to_dict()