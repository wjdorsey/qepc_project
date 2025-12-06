"""
QEPC NBA: Player usage and baseline stats from Eoin player_boxes_qepc.

This builds a per-player table with:
- games_played
- avg_minutes
- avg_points, avg_rebounds, avg_assists
- mean_points_share       (share of team points in games played)
- mean_rebounds_share     (share of team rebounds)
- mean_assists_share      (share of team assists)
- player_name             (if first/last name are available)

We aggregate by (player_id, team_name) so a player who changes teams
appears once per team.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .eoin_data_source import load_eoin_player_boxes, get_project_root


def build_player_usage_from_eoin(
    player_boxes: Optional[pd.DataFrame] = None,
    min_games: int = 10,
    project_root: Optional[Path] = None,
    cutoff_date: str = "2024-10-01",
) -> pd.DataFrame:
    """
    Build per-player usage stats from Eoin player_boxes_qepc.

    Parameters
    ----------
    player_boxes : DataFrame, optional
        If None, this will call load_eoin_player_boxes(project_root).
    min_games : int
        Minimum number of games required to keep a player/team row.
    project_root : Path, optional
        QEPC project root; used only if we need to load data.
    cutoff_date : str
        Only use games on/after this date (YYYY-MM-DD), if 'game_date' exists.

    Returns
    -------
    DataFrame with columns like:
        - player_id
        - player_name (if name columns exist)
        - team_name
        - games_played
        - avg_points
        - avg_rebounds
        - avg_assists
        - avg_minutes
        - mean_points_share
        - mean_rebounds_share
        - mean_assists_share
    """
    if player_boxes is None:
        player_boxes = load_eoin_player_boxes(project_root)

    df = player_boxes.copy()

    # --- Required columns ---
    required_cols = ["player_id", "team_name", "game_id", "points"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"player_boxes is missing required columns: {missing}. "
            "Check your normalization/rename step."
        )

    # --- Optional stat/name columns ---
    has_reb = "reboundstotal" in df.columns
    has_ast = "assists" in df.columns
    has_min = "numminutes" in df.columns
    has_first = "firstname" in df.columns
    has_last = "lastname" in df.columns

    # --- Optional: limit to recent games only ---
    if "game_date" in df.columns and cutoff_date is not None:
        cutoff = pd.to_datetime(cutoff_date).date()
        df = df[df["game_date"] >= cutoff].copy()

    # If we filtered everything out, bail early
    if df.empty:
        raise ValueError(
            f"No player box rows remain after applying cutoff_date={cutoff_date}. "
            "You may want to loosen the cutoff_date or check game_date values."
        )

    # --- Team keys & team totals per game ---
    df["team_key"] = df["team_name"]

    # Team game points (for points share)
    df["team_game_points"] = df.groupby(
        ["game_id", "team_key"]
    )["points"].transform("sum")

    # Avoid div-by-zero: if team_game_points is 0, treat denom as 1
    df["points_share"] = df["points"] / df["team_game_points"].where(
        df["team_game_points"] != 0, other=1
    )

    # Team game rebounds (for rebounds share)
    if has_reb:
        df["team_game_rebounds"] = df.groupby(
            ["game_id", "team_key"]
        )["reboundstotal"].transform("sum")
        df["rebounds_share"] = df["reboundstotal"] / df["team_game_rebounds"].where(
            df["team_game_rebounds"] != 0, other=1
        )
    else:
        df["rebounds_share"] = pd.NA

    # Team game assists (for assists share)
    if has_ast:
        df["team_game_assists"] = df.groupby(
            ["game_id", "team_key"]
        )["assists"].transform("sum")
        df["assists_share"] = df["assists"] / df["team_game_assists"].where(
            df["team_game_assists"] != 0, other=1
        )
    else:
        df["assists_share"] = pd.NA

    # --- Aggregate by (player_id, team) ---

    group_keys = ["player_id", "team_key"]

    # Use older-pandas-friendly agg: column -> function
    agg_dict = {
        "game_id": "nunique",
        "points": "mean",
        "points_share": "mean",
        "rebounds_share": "mean",
        "assists_share": "mean",
    }

    if has_reb:
        agg_dict["reboundstotal"] = "mean"
    if has_ast:
        agg_dict["assists"] = "mean"
    if has_min:
        agg_dict["numminutes"] = "mean"
    if has_first:
        agg_dict["firstname"] = "first"
    if has_last:
        agg_dict["lastname"] = "first"

    grouped = df.groupby(group_keys).agg(agg_dict).reset_index()

    # --- Rename to nicer column names ---

    grouped = grouped.rename(
        columns={
            "game_id": "games_played",
            "team_key": "team_name",
            "points": "avg_points",
            "reboundstotal": "avg_rebounds",
            "assists": "avg_assists",
            "numminutes": "avg_minutes",
            "points_share": "mean_points_share",
            "rebounds_share": "mean_rebounds_share",
            "assists_share": "mean_assists_share",
        }
    )

    # --- Build a display name if possible ---
    if has_first and has_last:
        grouped["player_name"] = (
            grouped["firstname"].astype(str).str.strip()
            + " "
            + grouped["lastname"].astype(str).str.strip()
        )

    # --- Filter out fringe guys with very few games ---
    grouped = grouped[grouped["games_played"] >= min_games].reset_index(drop=True)

    return grouped


def save_player_usage_to_cache(
    player_usage: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    filename: str = "eoin_player_usage.parquet",
) -> Path:
    """
    Save the per-player usage table to cache/imports as parquet.
    """
    if project_root is None:
        project_root = get_project_root()

    cache_dir = project_root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if player_usage is None:
        player_usage = build_player_usage_from_eoin(project_root=project_root)

    out_path = cache_dir / filename
    player_usage.to_parquet(out_path, index=False)
    print(f"Saved player usage to: {out_path}")
    return out_path
