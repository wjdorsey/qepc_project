# qepc/data_player_logs.py
#
# Utilities for working with the experimental 5-year lean player logs.
# These are stored alongside the notebooks in:
#   qepc_core/notebooks/data/player_logs_5yr_lean.csv
#
# This module now includes:
#   - load_player_logs_5yr()
#   - update_player_logs_from_nba_api()
#
# Safe behavior:
#   • If nba_api is missing -> clear ImportError with explanation.
#   • Only appends *new* rows (gameDate > last_date in existing CSV).
#   • Dedupe on (gameId, playerId).

from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import List, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _get_experimental_data_dir() -> Path:
    """
    Returns the notebooks/data directory inside the experimental qepc_core tree.

    Assumes this file lives at:
      qepc_project/experimental/GTP_REWRITE/qepc_core/qepc/data_player_logs.py

    So:
      __file__           -> .../qepc_core/qepc/data_player_logs.py
      .parents[1]        -> .../qepc_core
      / "notebooks/data" -> .../qepc_core/notebooks/data
    """
    core_root = Path(__file__).resolve().parents[1]  # qepc_core
    data_dir = core_root / "notebooks" / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def get_player_logs_path() -> Path:
    """
    Path to the lean 5-year player logs CSV used by the experimental engine.
    """
    return _get_experimental_data_dir() / "player_logs_5yr_lean.csv"


# ---------------------------------------------------------------------------
# Load existing lean 5-year logs
# ---------------------------------------------------------------------------

def load_player_logs_5yr(parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the experimental 5-year lean player logs.

    Expects columns (some may be missing depending on builder):
      - gameId, gameDate, Season
      - playerId, playerName
      - teamId, teamAbbrev, teamName
      - opponentTeamAbbrev
      - home, win
      - minutes
      - pts, reboundsTotal, assists, steals, blocks, turnovers, foulsPersonal
      - fieldGoalsMade, fieldGoalsAttempted, fieldGoalsPercentage
      - threePointersMade, threePointersAttempted, threePointersPercentage
      - freeThrowsMade, freeThrowsAttempted, freeThrowsPercentage
      - plusMinusPoints
    """
    path = get_player_logs_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Lean player logs not found at {path}.\n"
            "Run your 'build_player_logs_5yr' notebook first to create it."
        )

    parse_cols = ["gameDate"] if parse_dates else None
    df = pd.read_csv(path, parse_dates=parse_cols, low_memory=False)
    return df


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------

def _infer_nba_season_from_date(dt: pd.Timestamp) -> str:
    """
    Given a game date, infer NBA season string like '2021-22'.

    NBA seasons roll over around late October. We assume:
      - If month >= 8 (Aug+): season_start_year = dt.year
      - Else:                  season_start_year = dt.year - 1
    """
    if pd.isna(dt):
        raise ValueError("Cannot infer season from NaT date")

    start_year = dt.year if dt.month >= 8 else dt.year - 1
    end_year_short = str(start_year + 1)[-2:]
    return f"{start_year}-{end_year_short}"


def _get_current_nba_season() -> str:
    """
    Infer the *current* NBA season string from today's date.
    """
    today = datetime.utcnow()
    ts = pd.Timestamp(today)
    return _infer_nba_season_from_date(ts)


# ---------------------------------------------------------------------------
# Mapping raw nba_api -> lean schema
# ---------------------------------------------------------------------------

def _parse_matchup_to_flags(matchup: str):
    """
    Parse nba_api MATCHUP string into (opponentTeamAbbrev, home_flag).

    Examples:
      'BOS vs. NYK' -> ('NYK', 1)
      'BOS @ NYK'   -> ('NYK', 0)
    """
    if not isinstance(matchup, str):
        return None, None

    if " vs. " in matchup:
        parts = matchup.split(" vs. ")
        if len(parts) == 2:
            return parts[1].strip(), 1

    if " @ " in matchup:
        parts = matchup.split(" @ ")
        if len(parts) == 2:
            return parts[1].strip(), 0

    return None, None


def _map_leaguegamelog_to_lean(df_raw: pd.DataFrame, season_str: str) -> pd.DataFrame:
    """
    Transform nba_api.stats.endpoints.leaguegamelog output into our lean schema.
    """
    if df_raw.empty:
        return df_raw

    df_raw = df_raw.copy()

    # 1) Normalize dates
    if "GAME_DATE" in df_raw.columns:
        df_raw["gameDate"] = pd.to_datetime(df_raw["GAME_DATE"], errors="coerce")
    else:
        raise ValueError("Missing GAME_DATE column from leaguegamelog output")

    # 2) Derive opponentTeamAbbrev + home flag from MATCHUP
    if "MATCHUP" in df_raw.columns:
        opp_list = []
        home_list = []
        for m in df_raw["MATCHUP"]:
            opp, home = _parse_matchup_to_flags(m)
            opp_list.append(opp)
            home_list.append(home)
        df_raw["opponentTeamAbbrev"] = opp_list
        df_raw["home"] = home_list
    else:
        df_raw["opponentTeamAbbrev"] = None
        df_raw["home"] = None

    # 3) Win/Loss -> 1/0
    if "WL" in df_raw.columns:
        df_raw["win"] = df_raw["WL"].map({"W": 1, "L": 0}).astype("Int64")
    else:
        df_raw["win"] = pd.NA

    # 4) Tag season string
    df_raw["Season"] = season_str

    # 5) Column mapping to lean schema
    col_map = {
        "GAME_ID": "gameId",
        "gameDate": "gameDate",
        "Season": "Season",
        "PLAYER_ID": "playerId",
        "PLAYER_NAME": "playerName",
        "TEAM_ID": "teamId",
        "TEAM_ABBREVIATION": "teamAbbrev",
        "TEAM_NAME": "teamName",
        "opponentTeamAbbrev": "opponentTeamAbbrev",
        "home": "home",
        "win": "win",
        "MIN": "minutes",
        "PTS": "pts",
        "REB": "reboundsTotal",
        "AST": "assists",
        "STL": "steals",
        "BLK": "blocks",
        "TOV": "turnovers",
        "PF": "foulsPersonal",
        "FGM": "fieldGoalsMade",
        "FGA": "fieldGoalsAttempted",
        "FG_PCT": "fieldGoalsPercentage",
        "FG3M": "threePointersMade",
        "FG3A": "threePointersAttempted",
        "FG3_PCT": "threePointersPercentage",
        "FTM": "freeThrowsMade",
        "FTA": "freeThrowsAttempted",
        "FT_PCT": "freeThrowsPercentage",
        "PLUS_MINUS": "plusMinusPoints",
    }

    used_map = {src: dst for src, dst in col_map.items() if src in df_raw.columns}
    df_lean = df_raw[list(used_map.keys())].rename(columns=used_map)

    # 6) Numeric coercion
    num_fields = [
        "minutes",
        "pts",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "foulsPersonal",
        "fieldGoalsMade",
        "fieldGoalsAttempted",
        "fieldGoalsPercentage",
        "threePointersMade",
        "threePointersAttempted",
        "threePointersPercentage",
        "freeThrowsMade",
        "freeThrowsAttempted",
        "freeThrowsPercentage",
        "plusMinusPoints",
    ]
    for c in num_fields:
        if c in df_lean.columns:
            df_lean[c] = pd.to_numeric(df_lean[c], errors="coerce")

    # 7) Sort for stability
    sort_cols = [c for c in ["gameDate", "gameId", "playerId"] if c in df_lean.columns]
    if sort_cols:
        df_lean = df_lean.sort_values(sort_cols)

    return df_lean


# ---------------------------------------------------------------------------
# Incremental updater using nba_api
# ---------------------------------------------------------------------------

def update_player_logs_from_nba_api(
    seasons: Optional[List[str]] = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Incrementally update the lean 5-year player logs using nba_api.

    Steps:
      1) Load existing lean logs (player_logs_5yr_lean.csv).
      2) Determine last_date = max(gameDate).
      3) Decide which seasons to fetch:
           - If seasons is None, infer from last_date → current season.
      4) For each season: call leaguegamelog (player mode), map → lean schema.
      5) Keep only rows with gameDate > last_date.
      6) Append + dedupe on (gameId, playerId).
      7) Save back to CSV (unless dry_run=True).

    Returns:
      Updated DataFrame (even in dry_run mode).
    """
    try:
        from nba_api.stats.endpoints import leaguegamelog
    except ImportError as e:
        raise ImportError(
            "nba_api is required to update player logs.\n"
            "Install with: pip install nba_api"
        ) from e

    # 1) Load existing
    df_existing = load_player_logs_5yr(parse_dates=True)
    if verbose:
        print("=== update_player_logs_from_nba_api ===")
        print("Existing rows:", len(df_existing))
        print(
            "Existing date range:",
            df_existing["gameDate"].min(),
            "→",
            df_existing["gameDate"].max(),
        )

    last_date = df_existing["gameDate"].max()
    if pd.isna(last_date):
        raise ValueError("Existing lean logs have no valid gameDate values.")

    # 2) Decide seasons to fetch
    if seasons is None:
        def season_start_year(s: str) -> int:
            return int(s.split("-")[0])

        last_season = _infer_nba_season_from_date(last_date)
        current_season = _get_current_nba_season()

        sy_last = season_start_year(last_season)
        sy_curr = season_start_year(current_season)

        if verbose:
            print(f"Last season in logs: {last_season}")
            print(f"Current NBA season: {current_season}")

        if sy_curr <= sy_last:
            # Same season; just refetch that season and filter by date
            seasons = [last_season]
        else:
            # Span from last_season up to current_season
            seasons = [
                f"{y}-{str(y + 1)[-2:]}"
                for y in range(sy_last, sy_curr + 1)
            ]

        seasons = sorted(set(seasons), key=lambda s: int(s.split("-")[0]))

    if verbose:
        print("Seasons to fetch:", seasons)

    # 3) Fetch new data from nba_api
    new_lean_chunks: List[pd.DataFrame] = []

    for season_str in seasons:
        if verbose:
            print(f"\nFetching leaguegamelog for Season={season_str} (players)...")

        lg = leaguegamelog.LeagueGameLog(
            season=season_str,
            season_type_all_star="Regular Season",
            player_or_team="Player",
        )
        df_raw = lg.get_data_frames()[0]

        if df_raw.empty:
            if verbose:
                print(f"  ⚠️ No rows returned for {season_str}")
            continue

        if verbose:
            print("  Raw rows:", len(df_raw))

        df_lean_new = _map_leaguegamelog_to_lean(df_raw, season_str)
        if verbose:
            print("  Lean rows:", len(df_lean_new))

        new_lean_chunks.append(df_lean_new)

    if not new_lean_chunks:
        if verbose:
            print("\nNo new data fetched from nba_api; returning existing logs unchanged.")
        return df_existing

    df_new_all = pd.concat(new_lean_chunks, ignore_index=True)

    # 4) Keep only strictly newer games
    df_new_all = df_new_all[df_new_all["gameDate"] > last_date].copy()
    if df_new_all.empty:
        if verbose:
            print("\nFetched data, but nothing is newer than last_date; nothing to append.")
        return df_existing

    if verbose:
        print("\nNew rows after date filter:", len(df_new_all))
        print(
            "New date range:",
            df_new_all["gameDate"].min(),
            "→",
            df_new_all["gameDate"].max(),
        )

    # 5) Dedupe on (gameId, playerId)
    for col in ["gameId", "playerId"]:
        if col not in df_existing.columns or col not in df_new_all.columns:
            raise ValueError(
                f"Expected '{col}' column in both existing and new data for dedupe."
            )

    df_combined = pd.concat([df_existing, df_new_all], ignore_index=True)
    df_combined = (
        df_combined
        .drop_duplicates(subset=["gameId", "playerId"], keep="last")
        .sort_values(["gameDate", "gameId", "playerId"])
        .reset_index(drop=True)
    )

    if verbose:
        print("\nCombined rows after dedupe:", len(df_combined))

    # 6) Save if not dry_run
    if not dry_run:
        out_path = get_player_logs_path()
        if verbose:
            print(f"Writing updated lean player logs to: {out_path}")
        df_combined.to_csv(out_path, index=False)

    if verbose:
        print("=== Done update_player_logs_from_nba_api ===")

    return df_combined
