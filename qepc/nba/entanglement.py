"""Leakage-free rolling state vectors for the QEPC NBA multiverse."""

from __future__ import annotations

import pandas as pd

from .schema import validate_team_boxes, validate_games


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df
    if "game_datetime" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_datetime"]).dt.date
        return df
    raise ValueError("DataFrame is missing both 'game_date' and 'game_datetime'.")


def build_team_state_vectors(
    team_boxes: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10, 20),
) -> pd.DataFrame:
    """Compute rolling, leakage-free team state vectors.

    Returns one row per (game_id, team_id) containing the prior rolling
    averages for points for/against across multiple windows.
    """

    df = _ensure_game_date(validate_team_boxes(team_boxes.copy()))
    df = df.sort_values(["team_id", "game_date", "game_id"])

    features = []
    for stat in ("teamscore", "opponentscore"):
        for w in windows:
            col = f"{stat}_avg_prev{w}"
            features.append(col)
            df[col] = (
                df.groupby("team_id")[stat]
                .shift(1)
                .rolling(window=w, min_periods=1)
                .mean()
            )

    out_cols = ["game_id", "team_id", "game_date", *features]
    return df[out_cols].copy()


def attach_team_state_to_games(
    games_df: pd.DataFrame,
    team_state: pd.DataFrame,
) -> pd.DataFrame:
    """Attach home/away team state vectors to a games table."""

    games = _ensure_game_date(validate_games(games_df.copy()))
    state = _ensure_game_date(team_state.copy())

    # Home merge
    merged = games.merge(
        state.add_prefix("home_"),
        left_on=["game_id", "home_team_id"],
        right_on=["home_game_id", "home_team_id"],
        how="left",
    )

    # Away merge
    merged = merged.merge(
        state.add_prefix("away_"),
        left_on=["game_id", "away_team_id"],
        right_on=["away_game_id", "away_team_id"],
        how="left",
    )

    # Clean redundant key columns
    for col in ["home_game_id", "away_game_id"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])

    return merged
