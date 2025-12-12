"""
QEPC NBA: Eoin Kaggle data source helpers.

This module is the single place that knows:
- where the Eoin-based parquet files live
- how to load them into clean DataFrames
- how to summarize them for quick sanity checks

Usage from a notebook or script:

    from qepc.nba.eoin_data_source import (
        load_eoin_games,
        load_eoin_player_boxes,
        load_eoin_team_boxes,
        print_eoin_summary,
    )

    games = load_eoin_games()
    players = load_eoin_player_boxes()
    teams = load_eoin_team_boxes()
    print_eoin_summary(games, players, teams)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from qepc.utils.paths import get_project_root

from .schema import (
    validate_games,
    validate_player_boxes,
    validate_team_boxes,
)

def get_cache_imports_dir(project_root: Optional[Path] = None) -> Path:
    if project_root is None:
        project_root = get_project_root()
    cache_dir = project_root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def get_eoin_parquet_paths(project_root: Optional[Path] = None) -> dict:
    cache_dir = get_cache_imports_dir(project_root)
    return {
        "games": cache_dir / "eoin_games_qepc.parquet",
        "player_boxes": cache_dir / "eoin_player_boxes_qepc.parquet",
        "team_boxes": cache_dir / "eoin_team_boxes_qepc.parquet",
    }


# ---------------------------------------------------------------------------
# Core loaders
# ---------------------------------------------------------------------------

def _check_file_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"[Eoin loader] Expected {label} parquet not found at:\n"
            f"  {path}\n"
            f"Did you run the Eoin fetch/transform notebook to generate it?"
        )


def _read_parquet_safe(path: Path) -> pd.DataFrame:
    """Read parquet with a clear message if dependencies are missing."""

    try:
        return pd.read_parquet(path)
    except ImportError as exc:  # pyarrow / fastparquet missing
        raise ImportError(
            "Reading parquet requires optional dependency 'pyarrow' or 'fastparquet'. "
            "Install one of them (e.g., pip install pyarrow) to load QEPC caches."
        ) from exc


def load_eoin_games(project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the QEPC-ready games table from Eoin parquet.

    Columns include:
        - game_id
        - game_datetime (datetime64[ns, UTC])
        - game_date (python date)
        - home_team_id, away_team_id
        - home_score, away_score
        - winner_team_id
        - is_final (bool)
        plus any additional passthrough Eoin columns.
    """
    paths = get_eoin_parquet_paths(project_root)
    path = paths["games"]
    _check_file_exists(path, "games")
    df = _read_parquet_safe(path)
    return validate_games(df)


def load_eoin_player_boxes(project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the QEPC-ready player box scores from Eoin parquet.

    Columns include:
        - game_id
        - game_datetime (datetime64[ns, UTC])
        - game_date
        - player_id
        - team_name, team_city
        - opp_team_name, opp_team_city
        - box score stats (pts, reb, ast, etc.)
    """
    paths = get_eoin_parquet_paths(project_root)
    path = paths["player_boxes"]
    _check_file_exists(path, "player_boxes")
    df = _read_parquet_safe(path)
    return validate_player_boxes(df)


def load_eoin_team_boxes(project_root: Optional[Path] = None) -> pd.DataFrame:
    """
    Load the QEPC-ready team game logs from Eoin parquet.

    Columns include:
        - game_id
        - game_datetime (datetime64[ns, UTC])
        - game_date
        - team_id, team_city, team_name
        - opp_team_id, opp_team_city, opp_team_name
        - points, rebounds, etc. for the team as a whole
    """
    paths = get_eoin_parquet_paths(project_root)
    path = paths["team_boxes"]
    _check_file_exists(path, "team_boxes")
    df = _read_parquet_safe(path)
    return validate_team_boxes(df)


# ---------------------------------------------------------------------------
# Convenience helpers / quick diagnostics
# ---------------------------------------------------------------------------

def print_eoin_summary(
    games: Optional[pd.DataFrame] = None,
    player_boxes: Optional[pd.DataFrame] = None,
    team_boxes: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
) -> None:
    """
    Print a quick summary of the Eoin data universe: row counts, date ranges, seasons.

    If any of the three DataFrames is None, it will be loaded automatically.
    """
    if games is None:
        games = load_eoin_games(project_root)
    if player_boxes is None:
        player_boxes = load_eoin_player_boxes(project_root)
    if team_boxes is None:
        team_boxes = load_eoin_team_boxes(project_root)

    def _date_range(series: pd.Series) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        if series.isna().all():
            return None, None
        return series.min(), series.max()

    print("=== Eoin / QEPC Data Summary ===")
    print(f"Games:         {games.shape[0]:>8} rows, {games.shape[1]} columns")
    g_min, g_max = _date_range(pd.to_datetime(games["game_datetime"], errors="coerce", utc=True))
    print(f"  game_datetime: {g_min}  →  {g_max}")

    print(f"Player boxes:  {player_boxes.shape[0]:>8} rows, {player_boxes.shape[1]} columns")
    p_min, p_max = _date_range(pd.to_datetime(player_boxes["game_datetime"], errors="coerce", utc=True))
    print(f"  game_datetime: {p_min}  →  {p_max}")

    print(f"Team boxes:    {team_boxes.shape[0]:>8} rows, {team_boxes.shape[1]} columns")
    t_min, t_max = _date_range(pd.to_datetime(team_boxes["game_datetime"], errors="coerce", utc=True))
    print(f"  game_datetime: {t_min}  →  {t_max}")

    if "seasonwins" in team_boxes.columns and "seasonlosses" in team_boxes.columns:
        season_wins = team_boxes["seasonwins"].max()
        season_losses = team_boxes["seasonlosses"].max()
        print(f"Max season record seen in team_boxes: {season_wins}–{season_losses} (approx)")
