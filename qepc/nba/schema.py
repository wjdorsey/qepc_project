"""Lightweight schema validators for QEPC NBA data universes.

Each validator raises a ``ValueError`` with an actionable hint if the
expected columns are missing.  These are intentionally minimal so the
checks are cheap and friendly to notebook workflows.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd


def _check_columns(df: pd.DataFrame, required: Sequence[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{label} is missing required columns: {missing}. "
            "Did you run the Eoin fetch/transform notebooks to build the QEPC parquet?"
        )


def validate_games(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the games table from the Eoin pipeline."""

    required_any_date = ["game_id", "home_team_id", "away_team_id"]
    _check_columns(df, required_any_date, "games")

    if "game_date" not in df.columns and "game_datetime" not in df.columns:
        raise ValueError(
            "games requires either 'game_date' or 'game_datetime'. "
            "Use notebooks/brain/00_fetch_kaggle_eoin_dataset.ipynb to regenerate."
        )

    return df


def validate_team_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the team_boxes table."""

    required = [
        "game_id",
        "team_id",
        "opp_team_id",
        "teamscore",
        "opponentscore",
        "win",
    ]
    _check_columns(df, required, "team_boxes")
    return df


def validate_player_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the player_boxes table."""

    required = [
        "game_id",
        "team_name",
        "player_id",
        "points",
        "reboundstotal",
        "assists",
    ]
    _check_columns(df, required, "player_boxes")
    return df

