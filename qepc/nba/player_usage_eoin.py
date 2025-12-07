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

NEW (quantum-flavored):
    Optionally use recency-weighted averages via an exponential
    "decoherence" time scale tau_days, so recent games matter more.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .eoin_data_source import load_eoin_player_boxes, get_project_root
from qepc.quantum.decoherence import exponential_time_weights


def build_player_usage_from_eoin(
    player_boxes: Optional[pd.DataFrame] = None,
    min_games: int = 10,
    project_root: Optional[Path] = None,
    cutoff_date: str = "2024-10-01",
    use_recency_weights: bool = True,
    tau_points_days: float = 30.0,
    tau_rebounds_days: Optional[float] = None,
    tau_assists_days: Optional[float] = None,
    clip_days: float = 120.0,
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
    use_recency_weights : bool
        If True and 'game_date' exists, use exponential time decay to weight
        games (decoherence). If False, use simple unweighted means.
    tau_points_days : float
        Time constant (days) for points-related weighting.
    tau_rebounds_days : float or None
        Time constant for rebounds. If None, uses tau_points_days.
    tau_assists_days : float or None
        Time constant for assists. If None, uses tau_points_days.
    clip_days : float
        Cap Δdays at this many days; older games all get ~the same small weight.

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

    # --- Required base columns ---
    required_cols = ["player_id", "team_name", "game_id", "points"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"player_boxes is missing required columns: {missing}. "
            "Check your normalization/rename step."
        )

    # --- Optional stat/name/date columns ---
    has_reb = "reboundstotal" in df.columns
    has_ast = "assists" in df.columns
    has_min = "numminutes" in df.columns
    has_first = "firstname" in df.columns
    has_last = "lastname" in df.columns
    has_date = "game_date" in df.columns

    # Optional: limit to recent games only
    if has_date and cutoff_date is not None:
        cutoff = pd.to_datetime(cutoff_date).date()
        df = df[df["game_date"] >= cutoff].copy()

    if df.empty:
        raise ValueError(
            f"No player box rows remain after applying cutoff_date={cutoff_date}. "
            "You may want to loosen the cutoff_date or check game_date values."
        )

    # --- Build per-game team totals so we can compute shares ---
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
        df["rebounds_share"] = np.nan

    # Team game assists (for assists share)
    if has_ast:
        df["team_game_assists"] = df.groupby(
            ["game_id", "team_key"]
        )["assists"].transform("sum")
        df["assists_share"] = df["assists"] / df["team_game_assists"].where(
            df["team_game_assists"] != 0, other=1
        )
    else:
        df["assists_share"] = np.nan

    # --- Recency weights (decoherence) ---
    use_rw = use_recency_weights and has_date
    if use_rw:
        # If not provided, reuse points tau
        if tau_rebounds_days is None:
            tau_rebounds_days = tau_points_days
        if tau_assists_days is None:
            tau_assists_days = tau_points_days

        # Raw (unnormalized) weights per row; we normalize within groups later
        w_pts = exponential_time_weights(
            df["game_date"],
            ref_date=None,
            tau_days=tau_points_days,
            clip_days=clip_days,
            normalize=False,
        )
        df["w_pts"] = w_pts

        if has_reb:
            w_reb = exponential_time_weights(
                df["game_date"],
                ref_date=None,
                tau_days=tau_rebounds_days,
                clip_days=clip_days,
                normalize=False,
            )
            df["w_reb"] = w_reb
        else:
            df["w_reb"] = np.nan

        if has_ast:
            w_ast = exponential_time_weights(
                df["game_date"],
                ref_date=None,
                tau_days=tau_assists_days,
                clip_days=clip_days,
                normalize=False,
            )
            df["w_ast"] = w_ast
        else:
            df["w_ast"] = np.nan
    else:
        df["w_pts"] = 1.0
        df["w_reb"] = 1.0
        df["w_ast"] = 1.0

    # --- Aggregate by (player_id, team) with weighted means ---
    group_keys = ["player_id", "team_key"]

    rows = []
    grouped = df.groupby(group_keys)

    for keys, g in grouped:
        player_id, team_key = keys

        # games_played is still unweighted count of distinct games
        games_played = g["game_id"].nunique()

        if games_played < min_games:
            continue

        row = {
            "player_id": player_id,
            "team_name": team_key,
            "games_played": int(games_played),
        }

        # Names
        if has_first:
            row["firstname"] = str(g["firstname"].iloc[0]).strip()
        if has_last:
            row["lastname"] = str(g["lastname"].iloc[0]).strip()

        # Points-related weighted averages
        w_pts = g["w_pts"].to_numpy(dtype=float)
        total_w_pts = w_pts.sum()

        if total_w_pts > 0:
            pts = g["points"].to_numpy(dtype=float)
            pts_share = g["points_share"].to_numpy(dtype=float)

            row["avg_points"] = float((w_pts * pts).sum() / total_w_pts)
            row["mean_points_share"] = float((w_pts * pts_share).sum() / total_w_pts)
        else:
            row["avg_points"] = np.nan
            row["mean_points_share"] = np.nan

        # Rebounds-related
        if has_reb:
            w_reb = g["w_reb"].to_numpy(dtype=float)
            total_w_reb = w_reb.sum()
            if total_w_reb > 0:
                reb = g["reboundstotal"].to_numpy(dtype=float)
                reb_share = g["rebounds_share"].to_numpy(dtype=float)
                row["avg_rebounds"] = float((w_reb * reb).sum() / total_w_reb)
                row["mean_rebounds_share"] = float(
                    (w_reb * reb_share).sum() / total_w_reb
                )
            else:
                row["avg_rebounds"] = np.nan
                row["mean_rebounds_share"] = np.nan
        else:
            row["avg_rebounds"] = np.nan
            row["mean_rebounds_share"] = np.nan

        # Assists-related
        if has_ast:
            w_ast = g["w_ast"].to_numpy(dtype=float)
            total_w_ast = w_ast.sum()
            if total_w_ast > 0:
                ast = g["assists"].to_numpy(dtype=float)
                ast_share = g["assists_share"].to_numpy(dtype=float)
                row["avg_assists"] = float((w_ast * ast).sum() / total_w_ast)
                row["mean_assists_share"] = float(
                    (w_ast * ast_share).sum() / total_w_ast
                )
            else:
                row["avg_assists"] = np.nan
                row["mean_assists_share"] = np.nan
        else:
            row["avg_assists"] = np.nan
            row["mean_assists_share"] = np.nan

        # Minutes: we can just use points weights as a proxy if available
        if has_min:
            if use_rw and total_w_pts > 0:
                mins = g["numminutes"].to_numpy(dtype=float)
                row["avg_minutes"] = float((w_pts * mins).sum() / total_w_pts)
            else:
                row["avg_minutes"] = float(g["numminutes"].astype(float).mean())
        else:
            row["avg_minutes"] = np.nan

        rows.append(row)

    grouped_usage = pd.DataFrame(rows).reset_index(drop=True)

    # Build a display name if possible
    if has_first and has_last and not grouped_usage.empty:
        grouped_usage["player_name"] = (
            grouped_usage["firstname"].astype(str).str.strip()
            + " "
            + grouped_usage["lastname"].astype(str).str.strip()
        )

    return grouped_usage


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
