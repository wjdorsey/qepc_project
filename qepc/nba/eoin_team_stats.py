"""
QEPC NBA: Team-level stats built from Eoin team_boxes_qepc.

We expect team_boxes_qepc (already normalized) to have columns like:

    - game_id
    - team_id
    - team_name
    - win              (1 if this team won the game, 0 otherwise)
    - teamscore        (this team's points in the game)
    - opponentscore    (opponent's points in the game)
    - reboundstotal    (this team's total rebounds in the game)
    - assists          (this team's total assists in the game)

We aggregate to a per-team table with:

    - games_played
    - wins, losses, win_pct
    - pts_for, pts_against, pts_diff
    - off_ppg, def_ppg
    - reb_total, reb_pg               (if reboundstotal exists)
    - ast_total, ast_pg               (if assists exists)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from qepc.utils.paths import get_project_root

from .eoin_data_source import load_eoin_team_boxes
from .schema import validate_team_boxes


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        return df.copy()
    if "game_datetime" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_datetime"]).dt.date
        return df
    raise ValueError("team_boxes is missing both 'game_date' and 'game_datetime'.")


def build_team_stats_from_eoin(
    team_boxes: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    start_date: Optional[pd.Timestamp] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Aggregate Eoin team_boxes_qepc into a per-team stats table.

    Parameters
    ----------
    team_boxes : DataFrame, optional
        If None, this will call load_eoin_team_boxes(project_root).
    project_root : Path, optional
        QEPC project root; only used if we need to load data.

    Returns
    -------
    DataFrame with at least:
        - team_id
        - games_played
        - wins
        - losses
        - win_pct
        - pts_for
        - pts_against
        - pts_diff
        - off_ppg
        - def_ppg
        - reb_total, reb_pg      (if reboundstotal exists)
        - ast_total, ast_pg      (if assists exists)
    """
    if team_boxes is None:
        team_boxes = load_eoin_team_boxes(project_root)

    df = validate_team_boxes(team_boxes.copy())
    df = _ensure_game_date(df)

    if start_date is not None:
        start = pd.to_datetime(start_date).date()
        df = df[df["game_date"] >= start]

    if cutoff_date is not None:
        end = pd.to_datetime(cutoff_date).date()
        df = df[df["game_date"] < end]

    # Required columns we expect from your normalized team_boxes_qepc
    required_cols = [
        "team_id",
        "game_id",
        "win",
        "teamscore",
        "opponentscore",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"team_boxes is missing required columns: {missing}. "
            "Check your normalization/rename step."
        )

    has_reb = "reboundstotal" in df.columns
    has_ast = "assists" in df.columns

    # Group by team_id (you can later add season and group by [season, team_id])
    group_keys = ["team_id"]
    grouped = df.groupby(group_keys)

    agg = pd.DataFrame(index=grouped.size().index)

    # Basic counts
    agg["games_played"] = grouped["game_id"].nunique()
    agg["wins"] = grouped["win"].sum(min_count=1)
    agg["losses"] = agg["games_played"] - agg["wins"]
    agg["win_pct"] = agg["wins"] / agg["games_played"]

    # Points for / against
    agg["pts_for"] = grouped["teamscore"].sum(min_count=1)
    agg["pts_against"] = grouped["opponentscore"].sum(min_count=1)
    agg["pts_diff"] = agg["pts_for"] - agg["pts_against"]
    agg["off_ppg"] = agg["pts_for"] / agg["games_played"]
    agg["def_ppg"] = agg["pts_against"] / agg["games_played"]

    # Rebounds per game (if available)
    if has_reb:
        agg["reb_total"] = grouped["reboundstotal"].sum(min_count=1)
        agg["reb_pg"] = agg["reb_total"] / agg["games_played"]
    else:
        agg["reb_total"] = pd.NA
        agg["reb_pg"] = pd.NA

    # Assists per game (if available)
    if has_ast:
        agg["ast_total"] = grouped["assists"].sum(min_count=1)
        agg["ast_pg"] = agg["ast_total"] / agg["games_played"]
    else:
        agg["ast_total"] = pd.NA
        agg["ast_pg"] = pd.NA

    # Move team_id back to a normal column
    agg = agg.reset_index()

    return agg


def save_team_stats_to_cache(
    team_stats: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    filename: str = "eoin_team_stats.parquet",
) -> Path:
    """
    Save the per-team stats table to cache/imports as parquet.
    """
    if project_root is None:
        project_root = get_project_root()

    cache_dir = project_root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if team_stats is None:
        team_stats = build_team_stats_from_eoin(project_root=project_root)

    out_path = cache_dir / filename
    team_stats.to_parquet(out_path, index=False)
    print(f"Saved team_stats to: {out_path}")
    return out_path
