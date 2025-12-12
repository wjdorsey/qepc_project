"""
QEPC Module: player_usage_eoin.py

Builds player usage profiles from the Eoin Kaggle dataset (QEPC-normalized
player_boxes_qepc), including:

- Season-level averages for points / rebounds / assists
- Usage shares (points / team points, etc.)
- Recency features (last-N game rolling averages, e.g. last 5)
- Optional vs-opponent splits (how a player performs vs specific teams)

All of this is designed to feed into QEPC's λ builders for props and
entangled multiverse simulations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qepc.utils.paths import get_project_root

from .eoin_data_source import load_eoin_player_boxes


# ---------------------------------------------------------------------------
# Config dataclass (nice for future tweaks)
# ---------------------------------------------------------------------------

@dataclass
class PlayerUsageConfig:
    """
    Configuration for building player usage from Eoin data.
    """
    min_games: int = 10          # minimum games with this team to keep
    recent_window: int = 5       # window for last-N recency averages
    min_team_games: int = 10     # team-level filter if needed in future


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _ensure_datetime(player_boxes: pd.DataFrame) -> pd.DataFrame:
    df = player_boxes.copy()
    if "game_date" not in df.columns:
        raise KeyError("player_boxes must have a 'game_date' column (QEPC-normalized Eoin data).")
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def _add_team_totals_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (game_id, team_name), compute team totals for points / rebs / asts
    and merge them back so we can compute usage shares.
    """
    required = ["game_id", "team_name", "points", "reboundstotal", "assists"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_boxes is missing required columns: {missing}")

    team_totals = (
        df.groupby(["game_id", "team_name"], as_index=False)[
            ["points", "reboundstotal", "assists"]
        ]
        .sum()
        .rename(
            columns={
                "points": "team_points",
                "reboundstotal": "team_rebounds",
                "assists": "team_assists",
            }
        )
    )

    merged = df.merge(team_totals, on=["game_id", "team_name"], how="left")

    # Avoid division by zero in usage shares
    for col in ["team_points", "team_rebounds", "team_assists"]:
        merged[col] = merged[col].replace(0, np.nan)

    return merged


def _add_usage_shares(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add per-game usage shares: player stat / team stat for that game.
    """
    df = df.copy()
    df["points_share"] = df["points"] / df["team_points"]
    df["rebounds_share"] = df["reboundstotal"] / df["team_rebounds"]
    df["assists_share"] = df["assists"] / df["team_assists"]
    return df


def _add_recency_features(
    df: pd.DataFrame,
    recent_window: int = 5,
) -> pd.DataFrame:
    """
    For each (player_id, team_name), compute rolling last-N averages for
    points / rebounds / assists. Uses shift(1) so we only look at *prior*
    games relative to each row (no peeking into the future).
    """
    df = df.copy()

    df = df.sort_values(["player_id", "team_name", "game_date", "game_id"])

    group_keys = ["player_id", "team_name"]

    def rolling(stat_col: str, out_col: str) -> None:
        df[out_col] = (
            df.groupby(group_keys)[stat_col]
              .shift(1)  # only previous games
              .rolling(window=recent_window, min_periods=1)
              .mean()
        )

    rolling("points", "pts_avg_lastN")
    rolling("reboundstotal", "reb_avg_lastN")
    rolling("assists", "ast_avg_lastN")

    return df


def _build_basic_usage(
    df_with_shares_and_recency: pd.DataFrame,
    config: PlayerUsageConfig,
) -> pd.DataFrame:
    """
    Aggregate per (player_id, team_name) to build the core usage table.
    """
    df = df_with_shares_and_recency.copy()

    # First, basic aggregates
    group_cols = ["player_id", "team_name"]
    agg = df.groupby(group_cols).agg(
        games_played=("game_id", "nunique"),
        avg_points=("points", "mean"),
        avg_rebounds=("reboundstotal", "mean"),
        avg_assists=("assists", "mean"),
        mean_points_share=("points_share", "mean"),
        mean_rebounds_share=("rebounds_share", "mean"),
        mean_assists_share=("assists_share", "mean"),
    )

    agg = agg.reset_index()

    # Filter by minimum games with this team
    agg = agg[agg["games_played"] >= config.min_games].copy()

    # Now we want the *latest* recency values per player/team
    # (i.e., last row chronologically)
    df = df.sort_values(["player_id", "team_name", "game_date", "game_id"])
    latest_rows = (
        df.groupby(["player_id", "team_name"], as_index=False)
          .tail(1)[
              ["player_id", "team_name", "pts_avg_lastN", "reb_avg_lastN", "ast_avg_lastN"]
          ]
    )

    usage = agg.merge(latest_rows, on=["player_id", "team_name"], how="left")

    # Also attach player_name (from firstname/lastname)
    name_cols = []
    if "firstname" in df.columns:
        name_cols.append("firstname")
    if "lastname" in df.columns:
        name_cols.append("lastname")

    if name_cols:
        names = (
            df.groupby(["player_id", "team_name"], as_index=False)[name_cols]
              .agg(lambda x: x.iloc[0])
        )

        if "firstname" in names.columns and "lastname" in names.columns:
            names["player_name"] = (
                names["firstname"].fillna("").str.strip()
                + " "
                + names["lastname"].fillna("").str.strip()
            ).str.strip()
        elif "firstname" in names.columns:
            names["player_name"] = names["firstname"].astype(str)
        elif "lastname" in names.columns:
            names["player_name"] = names["lastname"].astype(str)
        else:
            names["player_name"] = ""

        usage = usage.merge(
            names[["player_id", "team_name", "player_name"]],
            on=["player_id", "team_name"],
            how="left",
        )

    # Order nicely: by team, then descending avg_points
    usage = usage.sort_values(
        ["team_name", "avg_points"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return usage


# ---------------------------------------------------------------------------
# Public API – main usage table
# ---------------------------------------------------------------------------

def build_player_usage_from_eoin(
    player_boxes: Optional[pd.DataFrame] = None,
    config: Optional[PlayerUsageConfig] = None,
) -> pd.DataFrame:
    """
    Build a player usage table from QEPC-ready Eoin player_boxes_qepc.

    Columns include:
      - player_id, team_name, player_name
      - games_played
      - avg_points, avg_rebounds, avg_assists
      - mean_points_share, mean_rebounds_share, mean_assists_share
      - pts_avg_lastN, reb_avg_lastN, ast_avg_lastN  (recency features)

    Parameters
    ----------
    player_boxes : DataFrame, optional
        If not provided, this will call load_eoin_player_boxes().
    config : PlayerUsageConfig, optional
        Controls min_games and recent_window.
    """
    if config is None:
        config = PlayerUsageConfig()

    if player_boxes is None:
        player_boxes = load_eoin_player_boxes()

    df = _ensure_datetime(player_boxes)
    df = _add_team_totals_per_game(df)
    df = _add_usage_shares(df)
    df = _add_recency_features(df, recent_window=config.recent_window)

    usage = _build_basic_usage(df, config=config)
    return usage


# ---------------------------------------------------------------------------
# Public API – vs-opponent splits
# ---------------------------------------------------------------------------

def build_player_vs_opponent_splits(
    player_boxes: Optional[pd.DataFrame] = None,
    min_games_vs_opp: int = 3,
) -> pd.DataFrame:
    """
    Build a table of per-player vs-opponent splits.

    Output columns:
      - player_id
      - team_name
      - opp_team_name
      - player_name
      - games_vs_opp
      - pts_vs_opp
      - reb_vs_opp
      - ast_vs_opp
    """
    if player_boxes is None:
        player_boxes = load_eoin_player_boxes()

    df = _ensure_datetime(player_boxes).copy()

    required = ["player_id", "team_name", "opp_team_name", "points", "reboundstotal", "assists"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_boxes is missing required columns: {missing}")

    group_cols = ["player_id", "team_name", "opp_team_name"]

    splits = (
        df.groupby(group_cols)
        .agg(
            games_vs_opp=("game_id", "nunique"),
            pts_vs_opp=("points", "mean"),
            reb_vs_opp=("reboundstotal", "mean"),
            ast_vs_opp=("assists", "mean"),
        )
        .reset_index()
    )

    splits = splits[splits["games_vs_opp"] >= min_games_vs_opp].copy()

    # Attach player_name
    name_cols = []
    if "firstname" in df.columns:
        name_cols.append("firstname")
    if "lastname" in df.columns:
        name_cols.append("lastname")

    if name_cols:
        names = (
            df.groupby(["player_id", "team_name"], as_index=False)[name_cols]
              .agg(lambda x: x.iloc[0])
        )

        if "firstname" in names.columns and "lastname" in names.columns:
            names["player_name"] = (
                names["firstname"].fillna("").str.strip()
                + " "
                + names["lastname"].fillna("").str.strip()
            ).str.strip()
        elif "firstname" in names.columns:
            names["player_name"] = names["firstname"].astype(str)
        elif "lastname" in names.columns:
            names["player_name"] = names["lastname"].astype(str)
        else:
            names["player_name"] = ""

        splits = splits.merge(
            names[["player_id", "team_name", "player_name"]],
            on=["player_id", "team_name"],
            how="left",
        )

    splits = splits.sort_values(
        ["player_name", "games_vs_opp"],
        ascending=[True, False],
    ).reset_index(drop=True)

    return splits


# ---------------------------------------------------------------------------
# Cache helpers (parquet in cache/imports)
# ---------------------------------------------------------------------------

def _get_cache_dir() -> Path:
    root = get_project_root()
    cache_dir = root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def save_player_usage_to_cache(
    usage: pd.DataFrame,
    filename: str = "eoin_player_usage.parquet",
) -> Path:
    cache_dir = _get_cache_dir()
    out_path = cache_dir / filename
    usage.to_parquet(out_path, index=False)
    return out_path


def load_player_usage_from_cache(
    filename: str = "eoin_player_usage.parquet",
) -> pd.DataFrame:
    cache_dir = _get_cache_dir()
    path = cache_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Player usage cache file not found: {path}")
    return pd.read_parquet(path)


def save_player_vs_opp_to_cache(
    splits: pd.DataFrame,
    filename: str = "eoin_player_vs_opp.parquet",
) -> Path:
    cache_dir = _get_cache_dir()
    out_path = cache_dir / filename
    splits.to_parquet(out_path, index=False)
    return out_path


def load_player_vs_opp_from_cache(
    filename: str = "eoin_player_vs_opp.parquet",
) -> pd.DataFrame:
    cache_dir = _get_cache_dir()
    path = cache_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Player vs-opp cache file not found: {path}")
    return pd.read_parquet(path)
