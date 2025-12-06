"""
QEPC NBA: Player usage and baseline stats from Eoin player_boxes_qepc.

This builds a per-player table with:
- games_played
- avg_minutes
- avg_points, avg_rebounds, avg_assists
- mean_points_share (share of team scoring in games played)
- mean_rebounds_share (share of team rebounds)
- mean_assists_share (share of team assists)
- player_name (if first/last name are available)
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
) -> pd.DataFrame:
    """
    Build per-player usage stats from Eoin player_boxes_qepc.

    Aggregates by (player_id, team_name) so players who change teams
    appear separately for each team.

    Requires columns in player_boxes:
        - player_id
        - team_name
        - game_id
        - points

    Optional columns:
        - reboundstotal
        - assists
        - numminutes
        - firstname
        - lastname
        - game_date (for filtering recent seasons)
    """
    if player_boxes is None:
        player_boxes = load_eoin_player_boxes(project_root)

    df = player_boxes.copy()

    required_cols = ["player_id", "team_name", "game_id", "points"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"player_boxes is missing required columns: {missing}. "
            "Check your normalization/rename step."
        )

    # Optional stat/name columns
    has_reb = "reboundstotal" in df.columns
    has_ast = "assists" in df.columns
    has_min = "numminutes" in df.columns
    has_first = "firstname" in df.columns
    has_last = "lastname" in df.columns

    # OPTIONAL: limit to recent games only (tune this date as you like)
    if "game_date" in df.columns:
        cutoff = pd.to_datetime("2024-10-01").date()  # start of 24-25-ish season
        df = df[df["game_date"] >= cutoff].copy()

    # Compute team-level points per game so we can get shares:
    df["team_key"] = df["team_name"]

    df["team_game_points"] = df.groupby(
        ["game_id", "team_key"]
    )["points"].transform("sum")

    # Avoid div-by-zero: if team_game_points is 0, share = 0
    df["points_share"] = df["points"] / df["team_game_points"].where(
        df["team_game_points"] != 0, other=1
    )

    # Rebound and assist shares, if available
    if has_reb:
        df["team_game_rebounds"] = df.groupby(
            ["game_id", "team_key"]
        )["reboundstotal"].transform("sum")
        df["rebounds_share"] = df["reboundstotal"] / df["team_game_rebounds"].where(
            df["team_game_rebounds"] != 0, other=1
        )
    else:
        df["rebounds_share"] = pd.NA

    if has_ast:
        df["team_game_assists"] = df.groupby(
            ["game_id", "team_key"]
        )["assists"].transform("sum")
        df["assists_share"] = df["assists"] / df["team_game_assists"].where(
            df["team_game_assists"] != 0, other=1
        )
    else:
        df["assists_share"] = pd.NA

    group_keys = ["player_id", "team_key"]

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

    # Rename columns to more friendly names
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

    # Build a display name if possible
    if has_first and has_last:
        grouped["player_name"] = (
            grouped["firstname"].astype(str).str.strip()
            + " "
            + grouped["lastname"].astype(str).str.strip()
        )

    # Filter out fringe guys with very few games
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
