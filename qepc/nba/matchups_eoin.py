"""
QEPC NBA: Matchup view built from Eoin games + team strengths.

This module builds a per-game matchup table that joins:
- games_qepc from Eoin (home/away teams, dates, scores)
- team strength scores from team_strengths_eoin

Usage example from a notebook:

    from qepc.nba.matchups_eoin import build_matchups_for_date

    matchups = build_matchups_for_date("2025-12-05")
    print(matchups.head())
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Union

import pandas as pd

from .eoin_data_source import load_eoin_games
from .team_strengths_eoin import calculate_advanced_strengths_from_eoin


DateLike = Union[str, date]


@dataclass
class MatchupConfig:
    """
    Simple config for matchup building.
    """
    include_final_games: bool = False  # False = upcoming games only


def _normalize_date(d: DateLike) -> date:
    """
    Convert a string or date into a date object.
    Accepts "YYYY-MM-DD" or anything pandas can parse.
    """
    if isinstance(d, date):
        return d
    # Assume string-like
    return pd.to_datetime(d).date()


def build_matchups_for_date(
    target_date: DateLike,
    games: Optional[pd.DataFrame] = None,
    strengths: Optional[pd.DataFrame] = None,
    config: Optional[MatchupConfig] = None,
) -> pd.DataFrame:
    """
    Build a matchup table for a given game_date.

    Columns in the output include:
        - game_id
        - game_datetime
        - game_date
        - home_team_id
        - away_team_id
        - home_strength_score
        - home_strength_rank
        - away_strength_score
        - away_strength_rank
        - strength_diff (home - away)
        - is_final
        - home_score (if available)
        - away_score (if available)
    """
    if config is None:
        config = MatchupConfig()

    if games is None:
        games = load_eoin_games()

    if strengths is None:
        strengths = calculate_advanced_strengths_from_eoin(verbose=False)

    target = _normalize_date(target_date)

    # Filter games to the target date
    if "game_date" not in games.columns:
        # Safety: build game_date if missing
        games = games.copy()
        games["game_datetime"] = pd.to_datetime(
            games["game_datetime"], errors="coerce", utc=True
        )
        games["game_date"] = games["game_datetime"].dt.date

    mask = games["game_date"] == target

    if not config.include_final_games and "is_final" in games.columns:
        mask &= ~games["is_final"]

    games_day = games.loc[mask].copy()

    if games_day.empty:
        print(f"No games found for date {target}.")
        return games_day

        # Select core columns
    keep_cols = [
        "game_id",
        "game_datetime",
        "game_date",
        "home_team_id",
        "away_team_id",
    ]

    # Include team names/cities if available in games frame
    for col in [
        "home_team_name",
        "home_team_city",
        "away_team_name",
        "away_team_city",
    ]:
        if col in games_day.columns:
            keep_cols.append(col)

    score_cols = []
    for col in ["home_score", "away_score", "winner", "is_final"]:
        if col in games_day.columns:
            keep_cols.append(col)
            score_cols.append(col)

    games_day = games_day[keep_cols]


    games_day = games_day[keep_cols]

    # Prepare strength frame for joins
    strength_cols = ["team_id", "strength_score", "strength_rank"]
    strengths_core = strengths[strength_cols].copy()

    # Join home team strength
    matchups = games_day.merge(
        strengths_core.add_prefix("home_"),
        left_on="home_team_id",
        right_on="home_team_id",
        how="left",
    )

    # Join away team strength
    strengths_core_away = strengths_core.add_prefix("away_")
    matchups = matchups.merge(
        strengths_core_away,
        left_on="away_team_id",
        right_on="away_team_id",
        how="left",
    )

    # Compute strength differential (home - away)
    matchups["strength_diff"] = (
        matchups["home_strength_score"] - matchups["away_strength_score"]
    )

    # Sort by tipoff time
    matchups = matchups.sort_values("game_datetime").reset_index(drop=True)

    return matchups
