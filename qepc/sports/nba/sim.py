"""
QEPC Module: sim.py
Handles NBA schedule loading and game-day filtering.
"""

from __future__ import annotations

import pandas as pd
from datetime import date, timedelta
from typing import Optional, Union

from qepc.autoload import paths

# --- Configuration ---

# Original format columns (used when show='clean')
QEPC_SCHEDULE_COLS = ["Date", "Time", "Away Team", "Home Team"]

# Primary expected format: "10/21/2025 7:30 PM"
DATE_PARSE_FORMAT = "%m/%d/%Y %I:%M %p"

# Simple cache so we don't keep re-reading Games.csv
_NBA_SCHEDULE_CACHE: Optional[pd.DataFrame] = None


def _build_game_datetime(df: pd.DataFrame) -> pd.Series:
    """
    Build a unified gameDate column from the schedule DataFrame.

    Strategy:
    1. If 'gameDate' already exists and is parseable -> use that.
    2. Else, if 'Date' + 'Time' exist -> try strict DATE_PARSE_FORMAT.
       If that fails for all rows, fall back to flexible to_datetime with infer.
    3. Else, if only 'Date' exists -> try to_datetime on 'Date' alone.
    """
    # 1) Already-present gameDate
    if "gameDate" in df.columns:
        game_dt = pd.to_datetime(df["gameDate"], errors="coerce")
        if not game_dt.isna().all():
            return game_dt

    # 2) Date + Time, strict format first
    if {"Date", "Time"}.issubset(df.columns):
        combined = df["Date"].astype(str) + " " + df["Time"].astype(str)

        # strict, known-good format
        game_dt = pd.to_datetime(
            combined,
            format=DATE_PARSE_FORMAT,
            errors="coerce",
        )
        if not game_dt.isna().all():
            return game_dt

        # fallback: flexible parsing
        game_dt_flex = pd.to_datetime(combined, errors="coerce", infer_datetime_format=True)
        if not game_dt_flex.isna().all():
            return game_dt_flex

    # 3) Only Date
    if "Date" in df.columns:
        game_dt = pd.to_datetime(df["Date"], errors="coerce", infer_datetime_format=True)
        if not game_dt.isna().all():
            return game_dt

    # If we reach here, everything failed
    raise ValueError(
        "Unable to construct gameDate from schedule. "
        "Checked: existing 'gameDate', 'Date' + 'Time', and 'Date' alone."
    )


def load_nba_schedule(reload: bool = False) -> Optional[pd.DataFrame]:
    """
    Load the master NBA schedule (Games.csv), clean it up, and cache it.

    Returns
    -------
    pd.DataFrame or None
        DataFrame with at least a 'gameDate' column plus the original fields,
        or None if the file cannot be loaded / parsed.
    """
    global _NBA_SCHEDULE_CACHE

    if _NBA_SCHEDULE_CACHE is not None and not reload:
        print("[QEPC NBA Sim] Schedule loaded from cache.")
        return _NBA_SCHEDULE_CACHE

    games_path = paths.get_games_path()

    if not games_path.exists():
        print(f"[QEPC NBA Sim] ERROR: Games.csv not found at {games_path}")
        return None

    try:
        df = pd.read_csv(games_path)

        # Build or fix gameDate column
        game_dt = _build_game_datetime(df)
        df["gameDate"] = game_dt

        # Drop rows where we still couldn't parse a date
        df.dropna(subset=["gameDate"], inplace=True)

        if df.empty:
            raise ValueError("Schedule is empty after date parsing / cleaning.")

        print(
            f"[QEPC NBA Sim] Successfully loaded and parsed "
            f"{len(df)} games from Games.csv."
        )

        _NBA_SCHEDULE_CACHE = df.copy()
        return df

    except Exception as e:
        # Verbose error printout for final debugging
        print(f"[QEPC NBA Sim] FAILED TO PROCESS SCHEDULE: {e}")
        return None


def _filter_and_clean_schedule(df: pd.DataFrame, show: str) -> Union[pd.DataFrame, None]:
    """
    Apply 'clean' or 'raw' display logic as required by the QEPC spec.

    Parameters
    ----------
    df : pd.DataFrame
        Filtered schedule (already subset to a date range).
    show : {'clean', 'raw'}
        'clean' -> return QEPC_SCHEDULE_COLS if available.
        'raw'   -> return the full DataFrame.

    Returns
    -------
    pd.DataFrame or None
        None if the filtered DataFrame is empty, otherwise a DataFrame.
    """
    if df is None or df.empty:
        return None

    if show == "clean":
        # Only keep columns that actually exist
        cols = [c for c in QEPC_SCHEDULE_COLS if c in df.columns]
        if not cols:
            # Fallback: nothing matches; return raw
            return df
        return df[cols]

    elif show == "raw":
        return df

    else:
        print(f"[QEPC NBA Sim] Invalid 'show' argument: {show!r}. Returning raw.")
        return df


def get_today_games(show: str = "clean") -> Optional[pd.DataFrame]:
    """
    Returns today's NBA schedule based on your local date.

    Parameters
    ----------
    show : {'clean', 'raw'}
        Output format preference.

    Returns
    -------
    pd.DataFrame or None
    """
    schedule = load_nba_schedule()
    if schedule is None:
        return None

    today = date.today()
    today_games = schedule[schedule["gameDate"].dt.date == today]

    return _filter_and_clean_schedule(today_games, show)


def get_tomorrow_games(show: str = "clean") -> Optional[pd.DataFrame]:
    """
    Returns tomorrow's NBA schedule based on your local date.

    Parameters
    ----------
    show : {'clean', 'raw'}
        Output format preference.

    Returns
    -------
    pd.DataFrame or None
    """
    schedule = load_nba_schedule()
    if schedule is None:
        return None

    tomorrow = date.today() + timedelta(days=1)
    tomorrow_games = schedule[schedule["gameDate"].dt.date == tomorrow]

    return _filter_and_clean_schedule(tomorrow_games, show)


def get_upcoming_games(days: int = 7, show: str = "clean") -> Optional[pd.DataFrame]:
    """
    Returns games scheduled within the next 'days' days (starting from tomorrow).

    Parameters
    ----------
    days : int, default 7
        Number of days ahead to look (excludes today).
    show : {'clean', 'raw'}
        Output format preference.

    Returns
    -------
    pd.DataFrame or None
    """
    schedule = load_nba_schedule()
    if schedule is None:
        return None

    today = date.today()
    end_date = today + timedelta(days=days)

    upcoming_games = schedule[
        (schedule["gameDate"].dt.date > today)
        & (schedule["gameDate"].dt.date <= end_date)
    ]

    return _filter_and_clean_schedule(upcoming_games, show)
