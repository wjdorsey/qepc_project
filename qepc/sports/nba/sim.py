"""
QEPC Module: sim.py
Schedule helpers for NBA using nba_api when available, with safe fallbacks.

Priority:
1. Live nba_api ScoreBoard for today's games.
2. nba_api LeagueGameLog for season schedule.
3. Local CSVs (data/Games.csv, data/raw/Games.csv, LeagueSchedule24_25/25_26).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import pandas as pd

from qepc.autoload.paths import get_games_path

try:  # nba_api may not be installed in every environment
    from nba_api.stats.endpoints import leaguegamelog, scoreboard
except Exception:  # pragma: no cover
    leaguegamelog = None
    scoreboard = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _looks_like_git_lfs_stub(df: pd.DataFrame) -> bool:
    """
    Detect if a CSV is actually a Git LFS pointer file rather than real data.
    """
    header_str = " ".join(str(c) for c in df.columns)
    if "git-lfs.github.com/spec/v1" in header_str:
        return True
    try:
        first_values = " ".join(str(v) for v in df.iloc[0].values)
        if "git-lfs.github.com/spec/v1" in first_values:
            return True
    except Exception:
        pass
    return False


def _standardize_schedule_df(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Try to coerce whatever columns exist into:
        - gameDate (datetime)
        - Home Team
        - Away Team

    If we can't find a reasonable date column, return None.
    """
    if df is None or df.empty:
        return None

    # 1) Find a date column
    date_col = None
    for cand in ["gameDate", "GAME_DATE", "Game Date", "DATE", "Date"]:
        if cand in df.columns:
            date_col = cand
            break

    if date_col is None:
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break

    if date_col is None:
        print("[QEPC sim] Schedule CSV has no recognizable date column; cannot standardize.")
        return None

    df = df.copy()
    df["gameDate"] = pd.to_datetime(df[date_col], errors="coerce")

    # 2) Home / Away team columns
    def _find_team_col(candidates):
        for cand in candidates:
            if cand in df.columns:
                return cand
        return None

    home_source = _find_team_col(
        ["Home Team", "HOME_TEAM", "home_team", "Home", "HOME", "HOME_TEAM_NAME"]
    )
    away_source = _find_team_col(
        ["Away Team", "AWAY_TEAM", "away_team", "Away", "Visitor", "VISITOR_TEAM_NAME"]
    )

    if "Home Team" not in df.columns and home_source is not None:
        df["Home Team"] = df[home_source]
    if "Away Team" not in df.columns and away_source is not None:
        df["Away Team"] = df[away_source]

    if "Home Team" not in df.columns or "Away Team" not in df.columns:
        print(
            "[QEPC sim] Warning: could not standardize Home Team / Away Team columns "
            "from schedule CSV. Some downstream features may not work."
        )

    return df


def _load_schedule_from_file() -> Optional[pd.DataFrame]:
    """
    Fallback: load schedule from local CSVs.

    Search order:
    1) data/Games.csv
    2) data/raw/Games.csv
    3) data/raw/LeagueSchedule24_25.csv
    4) data/raw/LeagueSchedule25_26.csv
    """
    from pathlib import Path

    data_dir = get_games_path().parent

    def _try_path(path: Path, label: str) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            print(f"[QEPC sim] Failed to read {label} at {path}: {exc}")
            return None
        if _looks_like_git_lfs_stub(df):
            print(
                f"[QEPC sim] {label} at {path} looks like a Git LFS pointer, "
                "not real schedule data."
            )
            return None
        df_std = _standardize_schedule_df(df)
        if df_std is None:
            print(f"[QEPC sim] Could not standardize {label} at {path}.")
            return None
        print(f"[QEPC sim] Loaded schedule from {label} at {path}.")
        return df_std

    # Primary canonical file
    primary = _try_path(get_games_path(), "data/Games.csv")
    if primary is not None:
        return primary

    # Alternative raw schedules
    alt_files = [
        ("data/raw/Games.csv", data_dir / "raw" / "Games.csv"),
        ("data/raw/LeagueSchedule24_25.csv", data_dir / "raw" / "LeagueSchedule24_25.csv"),
        ("data/raw/LeagueSchedule25_26.csv", data_dir / "raw" / "LeagueSchedule25_26.csv"),
    ]
    for label, path in alt_files:
        df = _try_path(path, label)
        if df is not None:
            return df

    print("[QEPC sim] No usable local schedule found in CSV files.")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_nba_schedule(season: Optional[str] = None) -> pd.DataFrame:
    """
    Load an NBA schedule table.

    Priority:
    1. nba_api.leaguegamelog for the given season (if available).
    2. Local schedule CSVs via _load_schedule_from_file().
    3. Empty DataFrame with standard columns.
    """
    # 1) Try LeagueGameLog for a rich season-long table
    if leaguegamelog is not None:
        try:
            kwargs = {}
            if season is not None:
                kwargs["season"] = season

            log = leaguegamelog.LeagueGameLog(**kwargs)
            df = log.get_data_frames()[0]

            if "GAME_DATE" in df.columns:
                df["gameDate"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
            elif "gameDate" in df.columns:
                df["gameDate"] = pd.to_datetime(df["gameDate"], errors="coerce")
            else:
                for cand in df.columns:
                    if "date" in cand.lower():
                        df["gameDate"] = pd.to_datetime(df[cand], errors="coerce")
                        break

            print("[QEPC sim] Fetched schedule from nba_api LeagueGameLog.")
            return df

        except Exception as exc:
            print(f"[QEPC sim] LeagueGameLog failed, will fall back to CSV: {exc}")

    # 2) Fallback: local CSVs
    file_df = _load_schedule_from_file()
    if file_df is not None:
        return file_df

    # 3) Final fallback: empty but well-shaped DataFrame
    print("[QEPC sim] No schedule source available; returning empty DataFrame.")
    return pd.DataFrame(columns=["gameDate", "Home Team", "Away Team", "gameId"])


def _load_today_from_scoreboard() -> Optional[pd.DataFrame]:
    """
    Use the nba_api ScoreBoard endpoint to get *today's* games with game IDs.
    """
    if scoreboard is None:
        return None

    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        records = []

        for game in games:
            game_date_raw = (
                game.get("gameTimeUTC")
                or game.get("gameDate")
                or game.get("GAME_DATE")
            )

            home = game.get("homeTeam", {}) or {}
            away = game.get("awayTeam", {}) or {}

            record = {
                "gameDate": pd.to_datetime(game_date_raw, errors="coerce"),
                "gameId": game.get("gameId") or game.get("GAME_ID"),
                "Home Team": home.get("teamName") or home.get("TEAM_NAME"),
                "Away Team": away.get("teamName") or away.get("TEAM_NAME"),
            }
            records.append(record)

        df = pd.DataFrame.from_records(records)
        if df.empty:
            return None

        print("[QEPC sim] Fetched today's schedule from nba_api ScoreBoard.")
        return df

    except Exception as exc:
        print(f"[QEPC sim] ScoreBoard API failed, will fall back to LeagueGameLog/CSV: {exc}")
        return None


def get_today_games(with_lineups: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of *today's* NBA games.

    Priority:
    1. ScoreBoard (live)
    2. LeagueGameLog (season) via load_nba_schedule()
    3. Local CSVs via _load_schedule_from_file()
    """
    today = date.today()

    # 1) Try live scoreboard
    today_df = _load_today_from_scoreboard()

    # 2) Fallback to full schedule
    if today_df is None:
        full_schedule = load_nba_schedule()

        if "gameDate" not in full_schedule.columns:
            print("[QEPC sim] Schedule has no gameDate column; returning empty DataFrame.")
            return pd.DataFrame()

        full_schedule = full_schedule.copy()
        full_schedule["gameDate"] = pd.to_datetime(
            full_schedule["gameDate"], errors="coerce"
        )

        if full_schedule["gameDate"].isna().all():
            print(
                "[QEPC sim] gameDate column is not datetimelike or all NaT; "
                "returning empty DataFrame."
            )
            return pd.DataFrame()

        today_df = full_schedule[full_schedule["gameDate"].dt.date == today].copy()

    if today_df.empty:
        print(f"[QEPC sim] No games found for {today.isoformat()}.")
        return today_df

    if with_lineups:
        try:
            from qepc.sports.nba.lineups import get_lineup
        except Exception as exc:
            print(f"[QEPC sim] Could not import get_lineup; returning games without lineups: {exc}")
            return today_df

        if "Home Lineup" not in today_df.columns:
            today_df["Home Lineup"] = ""
        if "Away Lineup" not in today_df.columns:
            today_df["Away Lineup"] = ""

        for idx, row in today_df.iterrows():
            home_team = row.get("Home Team")
            away_team = row.get("Away Team")
            game_id = row.get("gameId") if "gameId" in today_df.columns else None

            home_lineup = get_lineup(home_team, game_id)
            away_lineup = get_lineup(away_team, game_id)

            today_df.at[idx, "Home Lineup"] = ", ".join(home_lineup)
            today_df.at[idx, "Away Lineup"] = ", ".join(away_lineup)

    return today_df


def get_tomorrow_games(with_lineups: bool = False) -> pd.DataFrame:
    """
    Return a DataFrame of *tomorrow's* NBA games.
    """
    target = date.today() + timedelta(days=1)
    full_schedule = load_nba_schedule()

    if "gameDate" not in full_schedule.columns:
        print("[QEPC sim] Schedule has no gameDate column; returning empty DataFrame.")
        return pd.DataFrame()

    full_schedule = full_schedule.copy()
    full_schedule["gameDate"] = pd.to_datetime(
        full_schedule["gameDate"], errors="coerce"
    )

    if full_schedule["gameDate"].isna().all():
        print(
            "[QEPC sim] gameDate column is not datetimelike or all NaT; "
            "returning empty DataFrame."
        )
        return pd.DataFrame()

    df = full_schedule[full_schedule["gameDate"].dt.date == target].copy()

    if df.empty:
        print(f"[QEPC sim] No games found for {target.isoformat()}.")
        return df

    if with_lineups:
        try:
            from qepc.sports.nba.lineups import get_lineup
        except Exception as exc:
            print(f"[QEPC sim] Could not import get_lineup; returning games without lineups: {exc}")
            return df

        if "Home Lineup" not in df.columns:
            df["Home Lineup"] = ""
        if "Away Lineup" not in df.columns:
            df["Away Lineup"] = ""

        for idx, row in df.iterrows():
            home_team = row.get("Home Team")
            away_team = row.get("Away Team")

            home_lineup = get_lineup(home_team, None)
            away_lineup = get_lineup(away_team, None)

            df.at[idx, "Home Lineup"] = ", ".join(home_lineup)
            df.at[idx, "Away Lineup"] = ", ".join(away_lineup)

    return df


def get_upcoming_games(days: int = 7) -> pd.DataFrame:
    """
    Return all games from today through `days` days ahead (inclusive).
    """
    if days < 0:
        raise ValueError("days must be non-negative")

    full_schedule = load_nba_schedule()

    if "gameDate" not in full_schedule.columns:
        print("[QEPC sim] Schedule has no gameDate column; returning empty DataFrame.")
        return pd.DataFrame()

    full_schedule = full_schedule.copy()
    full_schedule["gameDate"] = pd.to_datetime(
        full_schedule["gameDate"], errors="coerce"
    )

    if full_schedule["gameDate"].isna().all():
        print(
            "[QEPC sim] gameDate column is not datetimelike or all NaT; "
            "returning empty DataFrame."
        )
        return pd.DataFrame()

    start = date.today()
    end = start + timedelta(days=days)

    mask = (full_schedule["gameDate"].dt.date >= start) & (
        full_schedule["gameDate"].dt.date <= end
    )
    df = full_schedule.loc[mask].copy().sort_values("gameDate")

    if df.empty:
        print(
            f"[QEPC sim] No upcoming games found between "
            f"{start.isoformat()} and {end.isoformat()}."
        )

    return df
