"""
QEPC NBA Team Strengths v2

This module builds a 30-team table of offensive and defensive strengths,
plus pace and a synthetic 'volatility' metric.

It supports two kinds of Team_Stats.csv:

1) Summary-style file with:
   - 'Team' column
   - ORtg/DRtg columns like 'ORtg', 'OffRtg', 'DRtg', 'DefRtg', etc.
   - optional 'Pace' column

2) Game-level file (your current setup) with columns like:
   - 'teamCity', 'teamName'
   - 'teamScore', 'opponentScore'
   - one row per game per team

In case (2), we:
   - build 'Team' = teamCity + " " + teamName
   - compute per-team offense = average teamScore
   - compute per-team defense = average opponentScore
   - use total points per game as a proxy for Pace
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple

import pandas as pd


# -----------------------------------------------------------------------------
# Helpers to locate project_root and load team stats safely
# -----------------------------------------------------------------------------

def _get_project_root() -> Path:
    """
    Try to get the QEPC project root in a robust way.

    1) If qepc.autoload.paths.get_project_root exists, use that.
    2) Otherwise, walk upwards from this file looking for data/Team_Stats.csv.
    3) Fallback to current working directory.
    """
    try:
        from qepc.autoload.paths import get_project_root  # type: ignore
        root = get_project_root()
        if isinstance(root, Path):
            return root
        return Path(root)
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / "data" / "Team_Stats.csv"
        if candidate.exists():
            return parent

    warnings.warn(
        "Could not automatically detect QEPC project root. "
        "Falling back to current working directory."
    )
    return Path.cwd()


def _detect_rating_columns(df: pd.DataFrame) -> Tuple[str | None, str | None]:
    """
    Detect offensive and defensive rating columns in Team_Stats.csv.

    Returns (off_col, def_col), each can be None if not found.
    """
    off_candidates = ["ORtg", "OffRtg", "Off_Rtg", "OffensiveRating", "Offensive_Rating"]
    def_candidates = ["DRtg", "DefRtg", "Def_Rtg", "DefensiveRating", "Defensive_Rating"]

    off_col = next((c for c in off_candidates if c in df.columns), None)
    def_col = next((c for c in def_candidates if c in df.columns), None)

    return off_col, def_col


def _detect_pace_column(df: pd.DataFrame) -> str | None:
    """Return the pace column name if it exists, else None."""
    for cand in ["Pace", "pace", "PossessionsPerGame"]:
        if cand in df.columns:
            return cand
    return None


# -----------------------------------------------------------------------------
# Main public function
# -----------------------------------------------------------------------------

def calculate_advanced_strengths() -> pd.DataFrame:
    """
    Build the advanced team strengths table from data/Team_Stats.csv.

    Returns
    -------
    strengths : DataFrame
        Columns:
            Team        - team name
            ORtg        - offensive strength (higher = better offense)
            DRtg        - defensive strength (lower = better defense)
            Pace        - pace proxy
            Volatility  - synthetic variance factor (8–14 range)
    """
    project_root = _get_project_root()
    team_stats_path = project_root / "data" / "raw" / "Team_Stats.csv"

    if not team_stats_path.exists():
        raise FileNotFoundError(
            f"Team_Stats.csv not found at {team_stats_path}. "
            "Make sure this file exists in your data/ folder."
        )

    team_stats = pd.read_csv(team_stats_path)
    cols = set(team_stats.columns)

    # -------------------------------------------------------------------------
    # CASE 1: Summary-style with 'Team' column already present
    # -------------------------------------------------------------------------
    if "Team" in cols:
        off_col, def_col = _detect_rating_columns(team_stats)
        pace_col = _detect_pace_column(team_stats)

        if off_col is None or def_col is None:
            raise ValueError(
                "Team_Stats.csv has a 'Team' column but no ORtg/DRtg-like columns.\n"
                f"Columns available: {list(team_stats.columns)}"
            )

        team_stats["Team"] = team_stats["Team"].astype(str).str.strip()

        strengths = pd.DataFrame()
        strengths["Team"] = team_stats["Team"]

        strengths["ORtg"] = pd.to_numeric(team_stats[off_col], errors="coerce")
        strengths["DRtg"] = pd.to_numeric(team_stats[def_col], errors="coerce")

        or_mean = strengths["ORtg"].mean()
        dr_mean = strengths["DRtg"].mean()
        strengths["ORtg"] = strengths["ORtg"].fillna(or_mean)
        strengths["DRtg"] = strengths["DRtg"].fillna(dr_mean)

        if pace_col is not None:
            strengths["Pace"] = pd.to_numeric(team_stats[pace_col], errors="coerce")
            pace_mean = strengths["Pace"].mean()
            strengths["Pace"] = strengths["Pace"].fillna(pace_mean)
        else:
            warnings.warn(
                "No pace column found in Team_Stats.csv; using a flat league-average pace."
            )
            strengths["Pace"] = 98.0

    # -------------------------------------------------------------------------
    # CASE 2: Game-level file (your current format) with teamCity/teamName
    # -------------------------------------------------------------------------
    elif {"teamCity", "teamName", "teamScore", "opponentScore"}.issubset(cols):
        # Build a Team label like "Boston Celtics"
        team_stats["teamCity"] = team_stats["teamCity"].astype(str).str.strip()
        team_stats["teamName"] = team_stats["teamName"].astype(str).str.strip()
        team_stats["Team"] = team_stats["teamCity"] + " " + team_stats["teamName"]

        # Convert scores to numeric
        team_stats["teamScore"] = pd.to_numeric(team_stats["teamScore"], errors="coerce")
        team_stats["opponentScore"] = pd.to_numeric(
            team_stats["opponentScore"], errors="coerce"
        )

        # Aggregate by Team across all games in the file
        agg = (
            team_stats
            .groupby("Team")
            .agg(
                ORtg=("teamScore", "mean"),          # offensive proxy = avg points scored
                DRtg=("opponentScore", "mean"),      # defensive proxy = avg points allowed
                Pace=("teamScore", "mean"),          # simple proxy; could use teamScore+oppScore
            )
            .reset_index()
        )

        strengths = agg.copy()

    else:
        raise ValueError(
            "Team_Stats.csv is in an unrecognized format.\n"
            "Supported formats:\n"
            "  1) Has 'Team' + ORtg/DRtg columns\n"
            "  2) Game-level with columns: 'teamCity', 'teamName', "
            "'teamScore', 'opponentScore'\n"
            f"Columns found: {list(team_stats.columns)}"
        )

    # -------------------------------------------------------------------------
    # Volatility: synthetic but data-driven
    # -------------------------------------------------------------------------
    strengths["Team"] = strengths["Team"].astype(str).str.strip()

    or_mean = strengths["ORtg"].mean()
    dr_mean = strengths["DRtg"].mean()

    diff_off = strengths["ORtg"] - or_mean
    diff_def = strengths["DRtg"] - dr_mean
    dist = (diff_off**2 + diff_def**2) ** 0.5

    if dist.max() == dist.min():
        strengths["Volatility"] = 10.0
    else:
        dist_min, dist_max = dist.min(), dist.max()
        strengths["Volatility"] = 8.0 + (dist - dist_min) / (dist_max - dist_min) * 6.0

    strengths = strengths.sort_values("Team").reset_index(drop=True)

    print("Built advanced team strengths from Team_Stats.csv")
    print(f"  Teams: {len(strengths)}")
    print(f"  ORtg range: {strengths['ORtg'].min():.1f} – {strengths['ORtg'].max():.1f}")
    print(f"  DRtg range: {strengths['DRtg'].min():.1f} – {strengths['DRtg'].max():.1f}")
    print(f"  Pace range: {strengths['Pace'].min():.1f} – {strengths['Pace'].max():.1f}")
    print(
        f"  Volatility range: {strengths['Volatility'].min():.2f} – "
        f"{strengths['Volatility'].max():.2f}"
    )

    return strengths
