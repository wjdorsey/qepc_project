"""
QEPC Data Loader – Team Game Logs

Canonical source: data/raw/team_game_logs.csv

This module provides a small, stable API for:
  - Locating the canonical team logs file
  - Loading it with standard dtypes
  - Filtering by season/date/home-only for backtests and strengths

Usage examples
--------------

from qepc.data_team_logs import (
    TeamLogsFilter,
    load_team_logs,
    load_team_logs_for_backtest,
)

# Load everything
df_all = load_team_logs()

# Load last 3 seasons of home games only
bt_logs = load_team_logs_for_backtest(lookback_seasons=3, home_only=True)

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd

# ---------------------------------------------------------------------
# Project root / config helpers
# ---------------------------------------------------------------------

try:
    # Preferred path: use QEPC's config if available
    from qepc.config import detect_project_root, QEPCConfig  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - fallback if config isn't there
    QEPCConfig = None  # type: ignore[assignment]

    def detect_project_root() -> Path:
        """
        Fallback: assume this file lives in qepc/ and project root is its parent.
        """
        return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------
# Filter configuration
# ---------------------------------------------------------------------


@dataclass
class TeamLogsFilter:
    """
    Options for subsetting team_game_logs.

    seasons:
        Explicit list of season strings like ["2021-22", "2022-23"].
        If provided, overrides min_season / max_season.

    min_season / max_season:
        Inclusive bounds on Season (string). Example: min_season="2021-22".

    start_date / end_date:
        Date window on gameDate (inclusive start, inclusive end).

    home_only:
        If True, keep only rows where home == 1.
    """

    seasons: Optional[Sequence[str]] = None
    min_season: Optional[str] = None
    max_season: Optional[str] = None
    start_date: Optional[Union[str, pd.Timestamp]] = None
    end_date: Optional[Union[str, pd.Timestamp]] = None
    home_only: bool = False


# ---------------------------------------------------------------------
# Core path + load helpers
# ---------------------------------------------------------------------


def get_team_logs_path(config: Optional["QEPCConfig"] = None) -> Path:
    """
    Resolve the canonical team logs CSV path.

    Prefers config.raw_data_dir if a QEPCConfig is provided; otherwise
    uses <project_root>/data/raw/team_game_logs.csv.
    """
    if config is not None and hasattr(config, "raw_data_dir"):
        raw_root = Path(config.raw_data_dir)
    else:
        project_root = detect_project_root()
        raw_root = project_root / "data" / "raw"

    path = raw_root / "team_game_logs.csv"
    return path


def _coerce_timestamp(value: Optional[Union[str, pd.Timestamp]]) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    if isinstance(value, pd.Timestamp):
        return value
    return pd.to_datetime(value, errors="coerce")


def load_team_logs(
    config: Optional["QEPCConfig"] = None,
    flt: Optional[TeamLogsFilter] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load the canonical team game logs with optional filtering.

    Parameters
    ----------
    config:
        Optional QEPCConfig. If provided, its raw_data_dir is used.
    flt:
        Optional TeamLogsFilter specifying seasons / dates / home_only.
    verbose:
        If True, prints a short summary of what was loaded.

    Returns
    -------
    pd.DataFrame with at least the following columns:
      - Season (str)
      - gameId
      - gameDate (Timestamp)
      - teamId, teamAbbrev, teamName
      - opponentTeamId, opponentTeamAbbrev, opponentTeamName
      - home (0/1), win (0/1)
      - pts, reboundsTotal, assists, steals, blocks, turnovers, foulsPersonal
      - fieldGoalsMade / Attempted / Percentage
      - threePointersMade / Attempted / Percentage
      - freeThrowsMade / Attempted / Percentage
      - reboundsOffensive / Defensive
    """
    path = get_team_logs_path(config)
    if not path.exists():
        raise FileNotFoundError(
            f"team_game_logs.csv not found at {path}.\n"
            "Make sure you've run the team logs build/updater notebooks "
            "to create this canonical file."
        )

    df = pd.read_csv(path, low_memory=False)

    # Standardize core types
    if "gameDate" in df.columns:
        df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
    if "Season" in df.columns:
        df["Season"] = df["Season"].astype(str)

    # Ensure home / win are numeric-ish
    for col in ["home", "win"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Apply filters if requested
    if flt is not None:
        if flt.seasons is not None:
            df = df[df["Season"].isin(list(flt.seasons))]

        else:
            # Min/max season bounds, using simple string comparison on "YYYY-YY"
            if flt.min_season is not None:
                df = df[df["Season"] >= flt.min_season]
            if flt.max_season is not None:
                df = df[df["Season"] <= flt.max_season]

        # Date filters
        start_ts = _coerce_timestamp(flt.start_date)
        end_ts = _coerce_timestamp(flt.end_date)

        if start_ts is not None:
            df = df[df["gameDate"] >= start_ts]
        if end_ts is not None:
            df = df[df["gameDate"] <= end_ts]

        # Home-only filter
        if flt.home_only and "home" in df.columns:
            df = df[df["home"] == 1]

    df = df.reset_index(drop=True)

    if verbose:
        seasons = sorted(df["Season"].unique()) if "Season" in df.columns else ["(unknown)"]
        print("[TeamLogs] Loaded team_game_logs.csv")
        print("  Path:    ", path)
        print("  Shape:    ", df.shape)
        if "gameDate" in df.columns:
            print("  Date range:", df["gameDate"].min(), "→", df["gameDate"].max())
        print("  Seasons:  ", seasons)

    return df


# ---------------------------------------------------------------------
# Convenience: backtest-oriented loader
# ---------------------------------------------------------------------


def _season_sort_key(season_str: str) -> int:
    """
    Turn '2021-22' into 2021 for easy sorting.
    """
    if not isinstance(season_str, str):
        return -9999
    try:
        return int(season_str.split("-")[0])
    except Exception:
        return -9999


def load_team_logs_for_backtest(
    config: Optional["QEPCConfig"] = None,
    lookback_seasons: int = 3,
    cutoff_date: Optional[Union[str, pd.Timestamp]] = None,
    home_only: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load a subset of team_game_logs tailored for backtesting.

    Parameters
    ----------
    lookback_seasons:
        How many most-recent seasons to keep (e.g., 3 → last 3 seasons).
        If <= 0, all seasons are kept.
    cutoff_date:
        Optional date cut. If provided, we keep only games with
        gameDate <= cutoff_date.
    home_only:
        If True, filter to home==1 rows (one row per game).
    verbose:
        Print a summary of what's returned.

    Returns
    -------
    pd.DataFrame subset of team_game_logs.
    """
    # Load everything first (no filter)
    df = load_team_logs(config=config, flt=None, verbose=False)

    # Season filtering
    if "Season" in df.columns and lookback_seasons > 0:
        all_seasons = sorted(df["Season"].unique(), key=_season_sort_key)
        if all_seasons:
            keep_seasons = all_seasons[-lookback_seasons:]
            df = df[df["Season"].isin(keep_seasons)]

    # Cutoff date
    cutoff_ts = _coerce_timestamp(cutoff_date)
    if cutoff_ts is not None and "gameDate" in df.columns:
        df = df[df["gameDate"] <= cutoff_ts]

    # Home-only
    if home_only and "home" in df.columns:
        df = df[df["home"] == 1]

    df = df.reset_index(drop=True)

    if verbose:
        print("[TeamLogs] Backtest subset")
        print("  Shape:   ", df.shape)
        if "gameDate" in df.columns:
            print("  Date range:", df["gameDate"].min(), "→", df["gameDate"].max())
        if "Season" in df.columns:
            print("  Seasons:", sorted(df["Season"].unique(), key=_season_sort_key))

    return df


# ---------------------------------------------------------------------
# CLI-style smoke test
# ---------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    print("=== QEPC team_game_logs smoke test ===")
    try:
        df_all = load_team_logs(verbose=True)
        print("\nHead:")
        print(df_all.head())

        df_bt = load_team_logs_for_backtest(lookback_seasons=3, home_only=True, verbose=True)
        print("\nBacktest subset head:")
        print(df_bt.head())
    except Exception as e:
        print("⚠️ Error in smoke test:", e)
