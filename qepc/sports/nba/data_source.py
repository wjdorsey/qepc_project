"""
QEPC NBA Data Source

Smart loaders that try to use the nba_api package when available and allowed,
but fall back to local CSVs when running in offline / cloud environments.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers to detect environment / capabilities
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """
    Try to reuse QEPC's existing project root resolver.
    Falls back to guessing based on this file's location.
    """
    try:
        from qepc.autoload import paths
        return paths.get_project_root()
    except Exception:
        return Path(__file__).resolve().parents[3]


def nba_api_available() -> bool:
    """Return True if the nba_api package can be imported."""
    try:
        import nba_api  # noqa: F401
        return True
    except Exception:
        return False


def online_mode_allowed() -> bool:
    """
    Decide if we are *allowed* to use online APIs.

    - If QEPC_OFFLINE=1 (or 'true'), we force offline mode.
    - Otherwise, we *try* online but still handle failures gracefully.
    """
    flag = os.environ.get("QEPC_OFFLINE", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return False
    return True


# ---------------------------------------------------------------------------
# CSV fallback loaders (what you already use today)
# ---------------------------------------------------------------------------

def load_team_stats_from_csv() -> pd.DataFrame:
    """
    Load team stats from your existing CSV file.

    This uses data/Team_Stats.csv, which is already part of your project.
    """
    root = _project_root()
    path = root / "data" / "Team_Stats.csv"

    if not path.exists():
        raise FileNotFoundError(f"Team_Stats.csv not found at {path}")

    df = pd.read_csv(path)
    # Basic cleaning to match your existing helper
    df["Team"] = df["Team"].astype(str)
    for col in ["ORtg", "DRtg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# NBA API loaders (only used when available)
# ---------------------------------------------------------------------------

def load_team_stats_from_nba_api(season: str = "2024-25") -> Optional[pd.DataFrame]:
    """
    Try to load team stats from the live NBA API via the nba_api package.

    Returns a DataFrame on success, or None if anything goes wrong.
    """
    if not nba_api_available():
        print("[QEPC NBA Data] nba_api package not installed; using CSV fallback.")
        return None

    if not online_mode_allowed():
        print("[QEPC NBA Data] Online mode disabled via QEPC_OFFLINE; using CSV fallback.")
        return None

    try:
        from nba_api.stats.endpoints import leaguedashteamstats

        print(f"[QEPC NBA Data] Fetching team stats from nba_api for season {season}...")
        endpoint = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame"
        )
        data = endpoint.get_data_frames()[0]

        # Normalize to something similar to Team_Stats.csv
        # nba_api uses 'TEAM_NAME', 'OFF_RATING', 'DEF_RATING', etc.
        cols_map = {
            "TEAM_NAME": "Team",
            "OFF_RATING": "ORtg",
            "DEF_RATING": "DRtg",
        }

        for src, dst in cols_map.items():
            if src not in data.columns:
                print(f"[QEPC NBA Data] Warning: expected column {src} missing from nba_api response.")
        df = data.rename(columns=cols_map)

        # Keep only the columns we care about for now
        keep_cols = [c for c in ["Team", "ORtg", "DRtg"] if c in df.columns]
        df = df[keep_cols].copy()

        print(f"[QEPC NBA Data] Loaded {len(df)} teams from nba_api.")
        return df

    except Exception as e:
        print(f"[QEPC NBA Data] Error while calling nba_api: {e}")
        print("[QEPC NBA Data] Falling back to CSV.")
        return None


# ---------------------------------------------------------------------------
# Public "smart" loader
# ---------------------------------------------------------------------------

def load_team_stats(prefer_api: bool = True, season: str = "2024-25") -> pd.DataFrame:
    """
    Smart team stats loader for QEPC.

    Behavior:
      - If prefer_api is True AND nba_api is installed AND not forced offline:
            Try nba_api first; if that fails, fall back to CSV.
      - Otherwise:
            Go straight to CSV, no online calls.

    This function does *not* modify any files; it only returns a DataFrame.
    """
    # Try API first if allowed
    if prefer_api and nba_api_available() and online_mode_allowed():
        df_api = load_team_stats_from_nba_api(season=season)
        if df_api is not None:
            return df_api

    # Fallback: use the CSV you already rely on
    return load_team_stats_from_csv()
