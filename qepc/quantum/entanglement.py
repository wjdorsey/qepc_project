"""
QEPC Quantum: Entanglement / co-movement utilities.

We measure how much two players' stats "move together" within the same
team context — a simple stand-in for the "entangled" relationship.

Implementation here is deliberately explicit and avoids fancy
stack/reset_index tricks so it behaves consistently across pandas versions.
"""

from __future__ import annotations

from typing import Optional, Union
from datetime import date, datetime

import numpy as np
import pandas as pd


DateLike = Union[date, datetime, str, pd.Timestamp]


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
        a correlation. Pairs with fewer shared games will be skipped.
    date_col : str or None
        If provided and cutoff_date is not None, we filter out games before
        cutoff_date using this column.
    cutoff_date : date-like or None
        If provided, ignore games before this date.

    Returns
    -------
    DataFrame with columns:
        - player_id_a
        - player_name_a
        - player_id_b
        - player_name_b
        - corr          (Pearson correlation of game-level stat)
        - abs_corr
        - n_shared_games
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

    # If we have fewer than 2 players with data, nothing to correlate
    if pivot.shape[1] < 2:
        return pd.DataFrame(
            columns=[
                "player_id_a",
                "player_name_a",
                "player_id_b",
                "player_name_b",
                "corr",
                "abs_corr",
                "n_shared_games",
            ]
        )

    # Compute correlation matrix (as a DataFrame)
    corr_mat = pivot.corr(min_periods=min_shared_games)

    # Compute shared game counts (as a plain numpy matrix)
    valid_mask = ~pivot.isna()
    shared_counts = valid_mask.T @ valid_mask  # DataFrame

    player_ids = list(pivot.columns)
    n_players = len(player_ids)

    rows = []

    for i in range(n_players):
        for j in range(i + 1, n_players):
            pid_a = int(player_ids[i])
            pid_b = int(player_ids[j])

            # How many games both have valid stats in
            n_shared = int(shared_counts.iloc[i, j])
            if n_shared < min_shared_games:
                continue

            corr = float(corr_mat.iloc[i, j])
            if np.isnan(corr):
                continue

            name_a = name_map.get(pid_a, f"player_{pid_a}")
            name_b = name_map.get(pid_b, f"player_{pid_b}")

            rows.append(
                {
                    "player_id_a": pid_a,
                    "player_name_a": name_a,
                    "player_id_b": pid_b,
                    "player_name_b": name_b,
                    "corr": corr,
                    "abs_corr": abs(corr),
                    "n_shared_games": n_shared,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "player_id_a",
                "player_name_a",
                "player_id_b",
                "player_name_b",
                "corr",
                "abs_corr",
                "n_shared_games",
            ]
        )

    ent_df = pd.DataFrame(rows)
    ent_df = ent_df.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return ent_df


def get_player_entanglement_view(
    ent_df: pd.DataFrame,
    player_id: int,
    min_shared_games: int = 5,
) -> pd.DataFrame:
    """
    Given a team entanglement DataFrame, extract all pairs involving `player_id`
    and return a tidy view with the "partner" player and correlation.

    Parameters
    ----------
    ent_df : DataFrame
        Output of build_team_entanglement.
    player_id : int
        Player to focus on.
    min_shared_games : int
        Filter out pairs with fewer shared games than this.

    Returns
    -------
    DataFrame with:
        - player_id
        - player_name
        - partner_id
        - partner_name
        - corr
        - abs_corr
        - n_shared_games
    """
    pid = int(player_id)

    mask_a = ent_df["player_id_a"] == pid
    mask_b = ent_df["player_id_b"] == pid
    sub = ent_df[mask_a | mask_b].copy()

    if sub.empty:
        return sub

    rows = []
    for _, row in sub.iterrows():
        if row["player_id_a"] == pid:
            player_name = row["player_name_a"]
            partner_id = row["player_id_b"]
            partner_name = row["player_name_b"]
        else:
            player_name = row["player_name_b"]
            partner_id = row["player_id_a"]
            partner_name = row["player_name_a"]

        rows.append(
            {
                "player_id": pid,
                "player_name": player_name,
                "partner_id": int(partner_id),
                "partner_name": partner_name,
                "corr": float(row["corr"]),
                "abs_corr": float(row["abs_corr"]),
                "n_shared_games": int(row["n_shared_games"]),
            }
        )

    out = pd.DataFrame(rows)
    out = out[out["n_shared_games"] >= min_shared_games].copy()
    out = out.sort_values("abs_corr", ascending=False).reset_index(drop=True)
    return out
