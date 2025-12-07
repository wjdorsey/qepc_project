"""
QEPC Quantum: Entanglement / co-movement utilities.

The idea here is to measure how much two players' stats "move together"
within the same team context – a crude, but useful, stand-in for the
"entangled" relationship in the QEPC metaphor.

We start simple:
    - For a given team_name and stat (e.g. points, assists),
      look at all games in player_boxes_qepc.
    - Build a game-by-player matrix of that stat.
    - Compute a correlation matrix across players.
    - Return a tidy DataFrame of pairwise correlations
      with the number of shared games.

Later, we can:
    - Do cross-stat entanglement (e.g., Tatum PTS vs Brown AST).
    - Condition on scripts, fatigue states, etc.
"""

from __future__ import annotations

from typing import Optional, List, Tuple
from datetime import date, datetime

import numpy as np
import pandas as pd


DateLike = date | datetime | str | pd.Timestamp


def _make_player_name_map(df_team: pd.DataFrame) -> dict[int, str]:
    """
    Build a mapping player_id -> player_name (or fallback to string(player_id)).
    """
    has_first = "firstname" in df_team.columns
    has_last = "lastname" in df_team.columns

    if has_first and has_last:
        names = (
            df_team["firstname"].astype(str).str.strip()
            + " "
            + df_team["lastname"].astype(str).str.strip()
        )
        mapping = (
            pd.DataFrame({"player_id": df_team["player_id"], "player_name": names})
            .drop_duplicates("player_id")
            .set_index("player_id")["player_name"]
            .to_dict()
        )
    else:
        mapping = (
            df_team["player_id"]
            .drop_duplicates()
            .astype(int)
            .apply(lambda pid: f"player_{pid}")
            .to_dict()
        )

    return mapping


def build_team_entanglement(
    player_boxes: pd.DataFrame,
    team_name: str,
    stat_col: str = "points",
    min_shared_games: int = 10,
    date_col: Optional[str] = "game_date",
    cutoff_date: Optional[DateLike] = None,
) -> pd.DataFrame:
    """
    Build an "entanglement" table for a single team and stat.

    Parameters
    ----------
    player_boxes : DataFrame
        Eoin player_boxes_qepc with at least:
            - game_id
            - team_name
            - player_id
            - <stat_col>
        Optionally:
            - game_date
            - firstname, lastname
    team_name : str
        Team name (e.g., "Celtics") to filter by.
    stat_col : str
        Which stat to use for co-movement (e.g., "points", "assists").
    min_shared_games : int
        Minimum number of games two players must both appear in to compute
        a correlation. Pairs with fewer shared games will have NaN corr.
    date_col : str or None
        If provided and cutoff_date is not None, we filter out games before
        cutoff_date using this column.
    cutoff_date : date-like or None
        If provided, ignore games before this date.

    Returns
    -------
    DataFrame with columns:
        - player_id_a
        - player_id_b
        - corr          (Pearson correlation of game-level stat)
        - n_shared_games
        - player_name_a
        - player_name_b
    """
    df = player_boxes.copy()

    needed = ["game_id", "team_name", "player_id", stat_col]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"player_boxes is missing required columns for entanglement: {missing}"
        )

    # Filter to the given team
    df = df[df["team_name"] == team_name].copy()
    if df.empty:
        raise ValueError(f"No rows found for team_name={team_name!r} in player_boxes.")

    # Optional date filter
    if date_col is not None and cutoff_date is not None and date_col in df.columns:
        cutoff = pd.to_datetime(cutoff_date).date()
        df = df[df[date_col] >= cutoff].copy()
        if df.empty:
            raise ValueError(
                f"No rows remain for team_name={team_name!r} after cutoff_date={cutoff_date}."
            )

    # Build name map for later
    name_map = _make_player_name_map(df)

    # Pivot: index = game_id, columns = player_id, values = stat_col
    pivot = df.pivot_table(
        index="game_id",
        columns="player_id",
        values=stat_col,
        aggfunc="sum",  # if duplicate rows, sum within game
    )

    # How many games each pair both have non-NA?
    valid_mask = ~pivot.isna()
    shared_counts = valid_mask.T @ valid_mask

    # Correlation matrix with min_shared_games threshold
    corr_mat = pivot.corr(min_periods=min_shared_games)

    # Give distinct names to the axes so reset_index doesn't collide
    corr_mat.index.name = "player_id_a"
    corr_mat.columns.name = "player_id_b"

    # Tidy long-form DataFrame
    corr_long = corr_mat.stack(dropna=False).reset_index(name="corr")

    # Filter out self-corr and NaNs, keep each pair once (a < b)
    corr_long = corr_long[
        (corr_long["player_id_a"] < corr_long["player_id_b"])
        & corr_long["corr"].notna()
    ].copy()

    # How many games each pair both have non-NA?
    valid_mask = ~pivot.isna()
    shared_counts = valid_mask.T @ valid_mask

    n_shared = (
        shared_counts.stack()
        .reset_index()
        .rename(
            columns={
                0: "n_shared_games",
                "level_0": "player_id_a",
                "level_1": "player_id_b",
            }
        )
    )

    corr_long = corr_long.merge(
        n_shared, on=["player_id_a", "player_id_b"], how="left"
    )

    # Attach names if available
    def map_name(pid: int) -> str:
        try:
            return name_map[int(pid)]
        except Exception:
            return f"player_{int(pid)}"

    corr_long["player_name_a"] = corr_long["player_id_a"].astype(int).map(map_name)
    corr_long["player_name_b"] = corr_long["player_id_b"].astype(int).map(map_name)

    # Sort by absolute correlation, strongest first
    corr_long["abs_corr"] = corr_long["corr"].abs()
    corr_long = corr_long.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    return corr_long[
        [
            "player_id_a",
            "player_name_a",
            "player_id_b",
            "player_name_b",
            "corr",
            "abs_corr",
            "n_shared_games",
        ]
    ]
