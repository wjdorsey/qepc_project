# qepc/nba/matchups_eoin.py

"""
QEPC NBA – Eoin Kaggle matchup builder.

This module builds matchup-level expected team points for a given date using:
- The Eoin Kaggle dataset (games + team boxscores),
- Modern-era team strengths (from 2022-10-01 onward),
- A simple strengths model: off_ppg / def_ppg symmetric blend,
- Schedule effects: home-court advantage and back-to-backs,
- A linear calibration layer fit in the backtest notebook.

Usage from a notebook:

    from qepc.nba.matchups_eoin import build_matchups_for_date

    matchups = build_matchups_for_date("2025-12-05")
    display(matchups)

The returned DataFrame includes both raw and calibrated expected points:
- exp_home_pts_raw / exp_away_pts_raw
- exp_home_pts / exp_away_pts (calibrated)
"""

from __future__ import annotations

import datetime as dt
from typing import Union

import numpy as np
import pandas as pd

from .eoin_data_source import load_eoin_games, load_eoin_team_boxes
from .eoin_team_stats import build_team_stats_from_eoin
from .team_strengths_eoin import calculate_advanced_strengths_from_eoin
from .calibration_eoin import calibrate_team_totals

DateLike = Union[str, dt.date, dt.datetime]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_date(d: DateLike) -> dt.date:
    """Convert string/datetime/date into a plain date object."""
    if isinstance(d, dt.datetime):
        return d.date()
    if isinstance(d, dt.date):
        return d
    return pd.to_datetime(d).date()


def _ensure_game_date(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a games/team_boxes DataFrame has a 'game_date' column as datetime.date."""
    if "game_date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["game_date"]):
            df = df.copy()
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        return df

    # Fallback: derive from 'game_datetime' if present
    if "game_datetime" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_datetime"]).dt.date
        return df

    raise ValueError("DataFrame is missing 'game_date' and 'game_datetime' columns.")


def _build_rest_index(games: pd.DataFrame) -> pd.DataFrame:
    """
    Build a rest / B2B index with one row per (game_id, team_id).

    Columns:
    - game_date
    - is_home
    - prev_date
    - days_since_prev
    - is_b2b
    """
    games = _ensure_game_date(games)

    rows = []
    for _, row in games.iterrows():
        gid = row["game_id"]
        gdate = row["game_date"]

        # Home entry
        rows.append({
            "game_id": gid,
            "team_id": row["home_team_id"],
            "is_home": True,
            "game_date": gdate,
        })
        # Away entry
        rows.append({
            "game_id": gid,
            "team_id": row["away_team_id"],
            "is_home": False,
            "game_date": gdate,
        })

    team_games = pd.DataFrame(rows).sort_values(["team_id", "game_date"])

    team_games["prev_date"] = team_games.groupby("team_id")["game_date"].shift(1)

    team_games["days_since_prev"] = (
        pd.to_datetime(team_games["game_date"])
        - pd.to_datetime(team_games["prev_date"])
    ).dt.days

    team_games["days_since_prev"] = team_games["days_since_prev"].fillna(999)
    team_games["is_b2b"] = team_games["days_since_prev"] == 1

    return team_games.set_index(["game_id", "team_id"])


def _build_modern_strengths(
    team_boxes: pd.DataFrame,
    modern_cutoff: dt.date = dt.date(2022, 10, 1),
    cutoff_date: dt.date | None = None,
) -> pd.DataFrame:
    """
    Build team strengths using only games on/after modern_cutoff.
    Returns strengths DataFrame indexed by team_id (via caller).
    """
    team_boxes = _ensure_game_date(team_boxes)

    modern = team_boxes[team_boxes["game_date"] >= modern_cutoff].copy()
    if cutoff_date is not None:
        modern = modern[modern["game_date"] < cutoff_date]

    if modern.empty:
        raise ValueError(
            f"No team boxes found on/after {modern_cutoff}. "
            "Check your Eoin data or cutoff date."
        )

    team_stats_modern = build_team_stats_from_eoin(modern)
    strengths_modern = calculate_advanced_strengths_from_eoin(
        team_stats_modern,
        start_date=modern_cutoff,
        cutoff_date=cutoff_date,
        verbose=False,
    )
    return strengths_modern


def _predict_team_points_with_schedule(
    game_row: pd.Series,
    strengths_index: pd.DataFrame,
    rest_index: pd.DataFrame,
    home_bonus: float = 1.5,
    away_penalty: float = 0.5,
    b2b_penalty: float = 1.5,
) -> tuple[float, float]:
    """
    Predict raw (uncalibrated) team points using:
    - team off_ppg / def_ppg from strengths_index,
    - symmetric blend,
    - home-court bonus and away penalty,
    - back-to-back penalty based on rest_index.
    """
    home_id = game_row["home_team_id"]
    away_id = game_row["away_team_id"]

    if home_id not in strengths_index.index or away_id not in strengths_index.index:
        return np.nan, np.nan

    home = strengths_index.loc[home_id]
    away = strengths_index.loc[away_id]

    home_off = home["off_ppg"]
    home_def = home["def_ppg"]
    away_off = away["off_ppg"]
    away_def = away["def_ppg"]

    # Base symmetric model
    home_raw = (home_off + away_def) / 2.0
    away_raw = (away_off + home_def) / 2.0

    # Home-court tweaks
    home_raw += home_bonus
    away_raw -= away_penalty

    # Back-to-back tweaks
    key_home = (game_row["game_id"], home_id)
    key_away = (game_row["game_id"], away_id)

    if key_home in rest_index.index:
        if bool(rest_index.loc[key_home, "is_b2b"]):
            home_raw -= b2b_penalty

    if key_away in rest_index.index:
        if bool(rest_index.loc[key_away, "is_b2b"]):
            away_raw -= b2b_penalty

    return home_raw, away_raw


def _extract_actual_score(game_row: pd.Series, prefix: str) -> float | None:
    candidates = [
        f"{prefix}_score",
        f"{prefix}score",
        f"{prefix}_points",
        f"{prefix}points",
    ]
    for cand in candidates:
        if cand in game_row.index:
            return game_row[cand]
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_matchups_for_date(
    target_date: DateLike,
    modern_cutoff: dt.date = dt.date(2022, 10, 1),
    cutoff_date: dt.date | None = None,
    home_bonus: float = 1.5,
    away_penalty: float = 0.5,
    b2b_penalty: float = 1.5,
    use_calibration: bool = True,
) -> pd.DataFrame:
    """
    Build a matchup DataFrame for all games on a given date,
    using Eoin data + QEPC strengths + schedule + optional calibration.

    Parameters
    ----------
    target_date : str | date | datetime
        Date of games to build matchups for (e.g. "2025-12-05").
    modern_cutoff : date
        Earliest game date to use when building strengths. Default 2022-10-01.
    home_bonus : float
        Home-court bonus (points added to home λ).
    away_penalty : float
        Away penalty (points subtracted from away λ).
    b2b_penalty : float
        Penalty applied if team is on second night of back-to-back.
    use_calibration : bool
        If True, apply calibration_eoin.calibrate_team_totals and return
        both raw and calibrated expectations.

    Returns
    -------
    pd.DataFrame
        Columns include (at minimum):

        - game_id
        - game_date
        - game_datetime (if available)
        - home_team_id, away_team_id
        - home_team_name, away_team_name (if available)
        - home_strength_score, away_strength_score
        - home_off_ppg, home_def_ppg
        - away_off_ppg, away_def_ppg
        - home_is_b2b, away_is_b2b
        - exp_home_pts_raw, exp_away_pts_raw
        - exp_home_pts, exp_away_pts
    """
    date_obj = _parse_date(target_date)
    if cutoff_date is None:
        cutoff_date = date_obj

    # Load base data
    games = load_eoin_games()
    games = _ensure_game_date(games)

    # Filter to games on this date
    day_games = games[games["game_date"] == date_obj].copy()
    if day_games.empty:
        # Return a nicely-structured but empty DataFrame
        return pd.DataFrame(
            columns=[
                "game_id",
                "game_date",
                "game_datetime",
                "home_team_id",
                "away_team_id",
                "home_team_name",
                "away_team_name",
                "home_strength_score",
                "away_strength_score",
                "home_off_ppg",
                "home_def_ppg",
                "away_off_ppg",
                "away_def_ppg",
                "home_is_b2b",
                "away_is_b2b",
                "exp_home_pts_raw",
                "exp_away_pts_raw",
                "exp_home_pts",
                "exp_away_pts",
            ]
        )

    # Build strengths from modern window using only *prior* games
    team_boxes = load_eoin_team_boxes()
    strengths_modern = _build_modern_strengths(
        team_boxes,
        modern_cutoff=modern_cutoff,
        cutoff_date=cutoff_date,
    )
    strengths_idx = strengths_modern.set_index("team_id")

    # Rest / B2B index from games up to and including the target date
    rest_source = games[games["game_date"] <= date_obj].copy()
    rest_idx = _build_rest_index(rest_source)

    rows = []
    for _, g in day_games.iterrows():
        raw_home, raw_away = _predict_team_points_with_schedule(
            g,
            strengths_index=strengths_idx,
            rest_index=rest_idx,
            home_bonus=home_bonus,
            away_penalty=away_penalty,
            b2b_penalty=b2b_penalty,
        )

        if np.isnan(raw_home) or np.isnan(raw_away):
            continue

        # Grab strength info (for inspection)
        home_team = strengths_idx.loc[g["home_team_id"]]
        away_team = strengths_idx.loc[g["away_team_id"]]

        # B2B flags
        home_key = (g["game_id"], g["home_team_id"])
        away_key = (g["game_id"], g["away_team_id"])

        home_is_b2b = bool(rest_idx.loc[home_key, "is_b2b"]) if home_key in rest_idx.index else False
        away_is_b2b = bool(rest_idx.loc[away_key, "is_b2b"]) if away_key in rest_idx.index else False

        # Calibrated expectations
        if use_calibration:
            cal_home, cal_away = calibrate_team_totals(raw_home, raw_away)
        else:
            cal_home, cal_away = raw_home, raw_away

        row_out = {
            "game_id": g["game_id"],
            "game_date": g["game_date"],
        }

        # Optional datetime if present
        if "game_datetime" in g.index:
            row_out["game_datetime"] = g["game_datetime"]
        else:
            row_out["game_datetime"] = pd.NaT

        # Team IDs & names
        row_out["home_team_id"] = g["home_team_id"]
        row_out["away_team_id"] = g["away_team_id"]

        # Use whatever name fields exist in games
        row_out["home_team_name"] = g.get("home_team_name", g.get("hometeamname", None))
        row_out["away_team_name"] = g.get("away_team_name", g.get("awayteamname", None))

        # Actual scores if present (useful for leakage-free backtests)
        row_out["actual_home_pts"] = _extract_actual_score(g, "home")
        row_out["actual_away_pts"] = _extract_actual_score(g, "away")

        # Strength-level info
        row_out["home_strength_score"] = home_team.get("strength_score", np.nan)
        row_out["away_strength_score"] = away_team.get("strength_score", np.nan)

        row_out["home_off_ppg"] = home_team.get("off_ppg", np.nan)
        row_out["home_def_ppg"] = home_team.get("def_ppg", np.nan)
        row_out["away_off_ppg"] = away_team.get("off_ppg", np.nan)
        row_out["away_def_ppg"] = away_team.get("def_ppg", np.nan)

        # Schedule flags
        row_out["home_is_b2b"] = home_is_b2b
        row_out["away_is_b2b"] = away_is_b2b

        # Raw vs calibrated expectations
        row_out["exp_home_pts_raw"] = raw_home
        row_out["exp_away_pts_raw"] = raw_away
        row_out["exp_home_pts"] = cal_home
        row_out["exp_away_pts"] = cal_away

        rows.append(row_out)

    matchups = pd.DataFrame(rows)

    # Sort by tip time if we have it, else by game_id
    if "game_datetime" in matchups.columns:
        matchups = matchups.sort_values(["game_date", "game_datetime", "game_id"])
    else:
        matchups = matchups.sort_values(["game_date", "game_id"])

    matchups.reset_index(drop=True, inplace=True)
    return matchups
