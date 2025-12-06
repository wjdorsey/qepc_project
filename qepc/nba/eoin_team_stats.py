"""
QEPC NBA: Team stats builder from Eoin team_boxes_qepc.

This builds a Team_Stats-style table from the Eoin Kaggle team game logs,
so the rest of QEPC can use it as a drop-in data source.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .eoin_data_source import load_eoin_team_boxes, get_project_root

# ---------------------------------------------------------------------------
# Configuration: match these to your Eoin TeamStatistics schema
# ---------------------------------------------------------------------------

# From your column list:
#   'teamscore'       -> points scored by the team
#   'opponentscore'   -> points allowed
TEAM_POINTS_FOR_COL = "teamscore"
TEAM_POINTS_AGAINST_COL = "opponentscore"


def build_team_stats_from_eoin(
    team_boxes: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build a Team_Stats-style aggregate from Eoin team_boxes_qepc.

    Aggregates by team_id.

    Output columns (at minimum):
        - team_id
        - games_played
        - wins
        - losses
        - win_pct
        - pts_for
        - pts_against
        - pts_diff
        - off_ppg (points_for per game)
        - def_ppg (points_against per game)
    """
    if team_boxes is None:
        team_boxes = load_eoin_team_boxes(project_root)

    df = team_boxes.copy()

    # Basic sanity check
    required_cols = ["team_id", "win", TEAM_POINTS_FOR_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"team_boxes is missing required columns: {missing}. "
            "Check your normalization/rename step."
        )

    # Group by team_id (later you can add 'season' and group by ['season', 'team_id'])
    grouped = df.groupby("team_id", dropna=False)

    # Core aggregates
    games_played = grouped["game_id"].count()
    wins = grouped["win"].sum(min_count=1)
    pts_for = grouped[TEAM_POINTS_FOR_COL].sum(min_count=1)

    if TEAM_POINTS_AGAINST_COL in df.columns:
        pts_against = grouped[TEAM_POINTS_AGAINST_COL].sum(min_count=1)
    else:
        pts_against = None

    # Assemble into a DataFrame
    agg = pd.DataFrame({
        "team_id": games_played.index,
        "games_played": games_played.values,
        "wins": wins.values,
        "pts_for": pts_for.values,
    })

    agg["losses"] = agg["games_played"] - agg["wins"]
    agg["win_pct"] = agg["wins"] / agg["games_played"]

    if pts_against is not None:
        agg["pts_against"] = pts_against.values
        agg["pts_diff"] = agg["pts_for"] - agg["pts_against"]
        agg["off_ppg"] = agg["pts_for"] / agg["games_played"]
        agg["def_ppg"] = agg["pts_against"] / agg["games_played"]
    else:
        agg["pts_against"] = pd.NA
        agg["pts_diff"] = pd.NA
        agg["off_ppg"] = agg["pts_for"] / agg["games_played"]
        agg["def_ppg"] = pd.NA

    return agg


def save_team_stats_to_cache(
    team_stats: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    filename: str = "eoin_team_stats.parquet",
) -> Path:
    """
    Save the aggregated team stats to cache/imports as a parquet file.
    """
    if project_root is None:
        project_root = get_project_root()

    cache_dir = project_root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if team_stats is None:
        team_stats = build_team_stats_from_eoin(project_root=project_root)

    out_path = cache_dir / filename
    team_stats.to_parquet(out_path, index=False)
    return out_path
