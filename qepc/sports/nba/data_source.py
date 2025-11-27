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

from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths


# ---------------------------------------------------------------------------
# Helpers to detect environment / capabilities
# ---------------------------------------------------------------------------
def _project_root() -> Path:
    """
    Try to find the project root (where the 'data' folder lives).

    This uses:
        - QEPC_PROJECT_ROOT env var, if set
        - Otherwise walks up from this file until it finds a 'data' dir
    """
    env_root = os.environ.get("QEPC_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if (root / "data").exists():
            return root

    # Fallback: walk upward from this file
    here = Path(__file__).resolve()
    for parent in [here] + list(here.parents):
        if (parent / "data").exists():
            return parent

    # Last resort: current working directory
    return Path.cwd().resolve()


def nba_api_available() -> bool:
    """
    Check whether the nba_api package is importable.
    """
    try:
        import nba_api  # type: ignore  # noqa: F401
        return True
    except Exception:
        return False


def online_mode_allowed() -> bool:
    """
    Check whether QEPC is allowed to use online APIs.

    Controlled via env var:
        QEPC_OFFLINE=1 -> force offline (no nba_api calls)
    """
    return os.environ.get("QEPC_OFFLINE", "0") != "1"


# ---------------------------------------------------------------------------
# CSV-based loaders (always available)
# ---------------------------------------------------------------------------
def load_team_stats_from_csv() -> pd.DataFrame:
    """
    Load team stats from your existing CSV file using the strengths_v2 helper.

    This builds team strengths from data/raw/Team_Stats.csv (game-level stats)
    and returns a DataFrame with columns like:
        Team, ORtg, DRtg, Pace, Volatility
    """
    return calculate_advanced_strengths()


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
        resp = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            measure_type_detailed_defense="Base",
        )
        data = resp.get_data_frames()[0]

        # Expect team abbreviation / name and ORtg/DRtg columns
        # We normalize to your internal schema: Team, ORtg, DRtg
        df = data.copy()

        # Try to build a nice 'Team' column
        if "TEAM_NAME" in df.columns:
            df["Team"] = df["TEAM_NAME"].astype(str)
        elif "TEAM_ABBREVIATION" in df.columns:
            df["Team"] = df["TEAM_ABBREVIATION"].astype(str)
        else:
            print("[QEPC NBA Data] ERROR: nba_api team stats missing TEAM_NAME/TEAM_ABBREVIATION.")
            return None

        # NBA API may provide OFF_RATING, DEF_RATING, etc.
        off_candidates = ["OFF_RATING", "OFFRTG", "OffRtg", "ORtg"]
        def_candidates = ["DEF_RATING", "DEFRTG", "DefRtg", "DRtg"]

        off_col = next((c for c in off_candidates if c in df.columns), None)
        def_col = next((c for c in def_candidates if c in df.columns), None)

        if off_col is None or def_col is None:
            print("[QEPC NBA Data] ERROR: nba_api frame missing clear ORtg/DRtg columns.")
            return None

        df = df[["Team", off_col, def_col]].copy()
        df = df.rename(columns={off_col: "ORtg", def_col: "DRtg"})

        # Basic cleaning
        df["Team"] = df["Team"].astype(str)
        df["ORtg"] = pd.to_numeric(df["ORtg"], errors="coerce")
        df["DRtg"] = pd.to_numeric(df["DRtg"], errors="coerce")

        print(f"[QEPC NBA Data] Loaded {len(df)} teams from nba_api.")
        return df

    except Exception as e:
        print(f"[QEPC NBA Data] ERROR during nba_api fetch: {e}")
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def load_team_stats(
    season: str = "2024-25",
    prefer_api: bool = True,
) -> pd.DataFrame:
    """
    High-level helper for getting team stats.

    Logic:
      - If prefer_api is True, nba_api is installed, and online mode is allowed:
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
