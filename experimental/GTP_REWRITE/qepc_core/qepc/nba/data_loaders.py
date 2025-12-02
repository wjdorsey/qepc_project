# qepc/nba/data_loaders.py
#
# NBA data loading utilities for QEPC experimental core.
#
# Responsibilities:
#   - Find the right raw team game log file under data/raw
#   - Parse dates robustly (handles timezone / "Z" suffix)
#   - Ensure a canonical schema for downstream modules:
#       gameDate (datetime, naive)
#       gameId   (string)
#       teamCity, teamName
#       opponentTeamCity, opponentTeamName (if available)
#       teamScore, opponentScore
#       home (0/1)
#       Season (int)
#
# This module is intentionally conservative: it prefers your big
# all-seasons NBA_API_QEPC_Format file if present, then falls back
# to TeamStatistics / Team_Stats only if needed.

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from qepc.config import QEPCConfig
from qepc.logging_utils import qstep, qwarn


# ---------------------------------------------------------------------
# File selection
# ---------------------------------------------------------------------


def _find_nba_team_logs(raw_root: Path) -> Optional[Path]:
    """
    Look for your main multi-season team log.

    Priority order:
      1) NBA_API_QEPC_Format*.csv / .xls  (10-year all-seasons log)
      2) TeamStatistics.csv               (team game logs)
      3) Team_Stats.csv                   (season stats in same shape)

    Returns:
        Path to chosen file, or None if nothing is found.
    """
    # 1) Highest priority: your all-seasons file
    for pat in ["NBA_API_QEPC_Format*.csv", "NBA_API_QEPC_Format*.xls"]:
        matches = list(raw_root.glob(pat))
        if matches:
            matches = sorted(matches, key=lambda p: str(p).lower())
            best = matches[0]
            qstep(f"Using NBA team logs file (NBA_API_QEPC_Format priority): {best}")
            return best

    # 2) Fallbacks: older team logs
    for pat in ["TeamStatistics.csv", "Team_Stats.csv"]:
        matches = list(raw_root.glob(pat))
        if matches:
            matches = sorted(matches, key=lambda p: str(p).lower())
            best = matches[0]
            qstep(f"Using NBA team logs file (fallback pattern {pat}): {best}")
            return best

    qwarn(f"No NBA team logs found under {raw_root}")
    return None


def _read_any(path: Path) -> pd.DataFrame:
    """
    Read a CSV or XLS file into a DataFrame.
    """
    suffix = path.suffix.lower()
    if suffix in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    # default: CSV-style
    return pd.read_csv(path)


# ---------------------------------------------------------------------
# Canonicalization helpers
# ---------------------------------------------------------------------


def _normalize_dates(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Normalize 'gameDate' in-place:
      - Find a date-like column (gameDate / GAME_DATE / Date / date)
      - Parse with utc=True to avoid mixed timezone dtype
      - Drop rows with invalid dates
      - Strip timezone info (naive datetime) for easier .dt access
    """
    date_col = None
    for cand in ["gameDate", "GAME_DATE", "Date", "date"]:
        if cand in df.columns:
            date_col = cand
            break

    if date_col is None:
        raise ValueError(f"No date-like column found in {filename}")

    # Parse with utc=True to satisfy pandas' mixed-timezone warning
    df["gameDate"] = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    before = len(df)
    df = df[df["gameDate"].notna()].copy()
    dropped = before - len(df)
    if dropped > 0:
        qwarn(f"Dropped {dropped} rows with invalid dates in {filename}")

    # Convert to naive datetime (no tz) so .dt works cleanly everywhere
    df["gameDate"] = df["gameDate"].dt.tz_convert(None)

    return df


def _ensure_game_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'gameId' column exists.

    If not present, we synthesize one from date + teamCity + teamName.
    This isn't as clean as a real gameId, but it's stable enough for
    grouping home/away pairs in backtests.
    """
    if "gameId" in df.columns:
        return df

    for c in ["teamCity", "teamName"]:
        if c not in df.columns:
            raise ValueError(
                f"Cannot synthesize gameId; required column '{c}' is missing."
            )

    df["gameId"] = (
        df["gameDate"].dt.strftime("%Y%m%d")
        + "_"
        + df["teamCity"].astype(str)
        + "_"
        + df["teamName"].astype(str)
    )
    return df


def _ensure_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'Season' column exists.

    If it's missing, infer it from gameDate.year. If a 'season' column
    exists, normalize it to 'Season'.
    """
    if "Season" in df.columns:
        return df

    # If they used a lowercase 'season' column, reuse that
    if "season" in df.columns:
        df["Season"] = df["season"]
        return df

    if "gameDate" not in df.columns:
        raise ValueError("Cannot infer Season; 'gameDate' column missing.")

    df["Season"] = df["gameDate"].dt.year
    return df


def _canonicalize_team_logs(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """
    Take a raw NBA team logs DataFrame and ensure it has all columns
    QEPC needs downstream, with consistent naming.
    """
    df = _normalize_dates(df, filename)
    df = _ensure_game_id(df)
    df = _ensure_season(df)

    required = ["teamCity", "teamName", "teamScore"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in {filename}. "
            "Expected at least teamCity, teamName, teamScore."
        )

    # If opponent info is missing, we can still use these logs for strengths
    # but backtest pairing will need gameId+home logic.
    for col in ["opponentTeamCity", "opponentTeamName", "opponentScore"]:
        if col not in df.columns:
            qwarn(f"Opponent column '{col}' missing in {filename}; some features may be limited.")

    # Ensure 'home' exists and is 0/1
    if "home" not in df.columns:
        qwarn(
            f"'home' column missing in {filename}; "
            "backtest pairing will treat all rows as unknown home/away."
        )
    else:
        # Coerce to 0/1 ints if possible
        df["home"] = df["home"].astype(float).round().astype(int)

    return df


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def load_nba_team_logs(config: QEPCConfig) -> pd.DataFrame:
    """
    Load multi-season NBA team game logs into a canonical format.

    Returns a DataFrame with (at minimum):

        gameDate   : datetime64[ns] (naive, no tz)
        gameId     : string
        teamCity   : string
        teamName   : string
        teamScore  : numeric
        opponentScore      : numeric (if available)
        opponentTeamCity   : string  (if available)
        opponentTeamName   : string  (if available)
        home       : 0/1 (if available)
        Season     : int

    This is the primary input for:
        - team strengths computation
        - backtest game construction
    """
    raw_root = config.raw_root
    path = _find_nba_team_logs(raw_root)
    if path is None:
        raise FileNotFoundError(
            f"Could not locate NBA team logs file under {raw_root}. "
            "Expected something like NBA_API_QEPC_Format.csv or TeamStatistics.csv."
        )

    df_raw = _read_any(path)
    qstep(f"Loaded raw NBA team logs: {len(df_raw)} rows, {len(df_raw.columns)} columns")

    df = _canonicalize_team_logs(df_raw, path.name)

    # Small debug / sanity print
    qstep(
        f"NBA team logs canonicalized: {len(df)} rows, "
        f"dates {df['gameDate'].min().date()} → {df['gameDate'].max().date()}, "
        f"seasons {df['Season'].min()} → {df['Season'].max()}"
    )

    return df
