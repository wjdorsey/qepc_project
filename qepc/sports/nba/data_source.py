"""
QEPC NBA Data Sources
---------------------
Unified helpers for "today's games" and live data.

Priority order for schedule:
    1) data/live/games_today_nba_api.csv        (your notebook snapshot)
    2) nba_api ScoreboardV2                     (official-ish live source)
    3) ESPN scoreboard API                      (backup live source)
    4) local data/Games.csv                     (canonical schedule)
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# QEPC path helpers
# ---------------------------------------------------------------------------

try:
    from qepc.autoload.paths import get_data_dir, get_games_path
except Exception:
    # Fallbacks if autoload is not available
    def get_data_dir() -> Path:
        return Path.cwd() / "data"

    def get_games_path() -> Path:
        return get_data_dir() / "Games.csv"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _ensure_live_dir() -> Path:
    data_dir = Path(get_data_dir())
    live_dir = data_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    return live_dir


def _normalize_today_filter(df: pd.DataFrame, date_col: str, target_date: date) -> pd.DataFrame:
    """Ensure date_col is datetime and filter rows where .dt.date == target_date."""
    if date_col not in df.columns:
        return pd.DataFrame()

    df = df.copy()
    df["gameDate"] = pd.to_datetime(df[date_col], errors="coerce")

    if df["gameDate"].isna().all():
        return pd.DataFrame()

    return df[df["gameDate"].dt.date == target_date].copy()


def _finalize_schedule_df(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Ensure df has the canonical columns:
        gameDate, Home Team, Away Team, gameId
    and drop rows missing team names.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["gameDate", "Home Team", "Away Team", "gameId"])

    df = df.copy()

    # Ensure gameId column exists
    if "gameId" not in df.columns:
        if "GAME_ID" in df.columns:
            df["gameId"] = df["GAME_ID"]
        else:
            df["gameId"] = pd.NA

    # Ensure gameDate is datetime
    if "gameDate" not in df.columns:
        # Try common date columns if needed
        for cand in ["GAME_DATE_EST", "GAME_DATE", "Date"]:
            if cand in df.columns:
                df["gameDate"] = pd.to_datetime(df[cand], errors="coerce")
                break

    if "gameDate" not in df.columns:
        df["gameDate"] = pd.NaT

    # Only keep rows with usable names
    df = df[
        df["Home Team"].notna()
        & df["Away Team"].notna()
    ].copy()

    if df.empty and verbose:
        print("[QEPC data_sources] Schedule DataFrame ended up empty after normalization.")

    return df[["gameDate", "Home Team", "Away Team", "gameId"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 1) data/live/games_today_nba_api.csv
# ---------------------------------------------------------------------------

def load_today_from_live_csv(
    target_date: Optional[date] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load today's games from data/live/games_today_nba_api.csv

    Expected columns (from your live notebook):
        - GAME_DATE_EST (or similar)
        - HOME_TEAM_QEPC / AWAY_TEAM_QEPC   (preferred)
        - or Home Team / Away Team

    Returns canonical columns via _finalize_schedule_df.
    """
    if target_date is None:
        target_date = date.today()

    live_dir = _ensure_live_dir()
    live_path = live_dir / "games_today_nba_api.csv"

    if not live_path.exists():
        if verbose:
            print(f"[QEPC data_sources] No live CSV at {live_path}; skipping.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(live_path)
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] Failed to read {live_path}: {exc}")
        return pd.DataFrame()

    if df.empty:
        if verbose:
            print("[QEPC data_sources] Live CSV is empty; skipping.")
        return pd.DataFrame()

    # Find a date column
    date_col = None
    for cand in ["gameDate", "GAME_DATE_EST", "GAME_DATE"]:
        if cand in df.columns:
            date_col = cand
            break
    if date_col is None:
        # fall back to any column with 'date' in name
        for c in df.columns:
            if "date" in str(c).lower():
                date_col = c
                break

    if date_col is None:
        if verbose:
            print("[QEPC data_sources] Live CSV has no recognizable date column; skipping.")
        return pd.DataFrame()

    df_today = _normalize_today_filter(df, date_col, target_date)
    if df_today.empty:
        if verbose:
            print(f"[QEPC data_sources] Live CSV has no rows for {target_date.isoformat()}; skipping.")
        return pd.DataFrame()

    # Map team names
    if "Home Team" not in df_today.columns:
        if "HOME_TEAM_QEPC" in df_today.columns:
            df_today["Home Team"] = df_today["HOME_TEAM_QEPC"]
        elif "HOME_TEAM_NAME" in df_today.columns:
            df_today["Home Team"] = df_today["HOME_TEAM_NAME"]

    if "Away Team" not in df_today.columns:
        if "AWAY_TEAM_QEPC" in df_today.columns:
            df_today["Away Team"] = df_today["AWAY_TEAM_QEPC"]
        elif "AWAY_TEAM_NAME" in df_today.columns:
            df_today["Away Team"] = df_today["AWAY_TEAM_NAME"]

    df_today = _finalize_schedule_df(df_today, verbose=verbose)
    if verbose and not df_today.empty:
        print("[QEPC data_sources] Loaded today's schedule from games_today_nba_api.csv.")

    return df_today


# ---------------------------------------------------------------------------
# 2) nba_api ScoreboardV2
# ---------------------------------------------------------------------------

def load_today_from_nba_api(
    target_date: Optional[date] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load today's games from nba_api ScoreboardV2.

    Returns canonical schedule columns on success, or empty DataFrame on failure.
    """
    if target_date is None:
        target_date = date.today()

    try:
        from nba_api.stats.endpoints import ScoreboardV2
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] nba_api not available: {exc}")
        return pd.DataFrame()

    try:
        sb = ScoreboardV2(game_date=target_date.strftime("%Y-%m-%d"))
        frames = sb.get_data_frames()
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] ScoreboardV2 error: {exc}")
        return pd.DataFrame()

    if not frames or len(frames) < 2:
        if verbose:
            print("[QEPC data_sources] ScoreboardV2 returned no frames.")
        return pd.DataFrame()

    game_header = frames[0]  # GAME_HEADER
    line_score = frames[1]   # LINE_SCORE

    if game_header.empty or line_score.empty:
        if verbose:
            print("[QEPC data_sources] ScoreboardV2 data empty.")
        return pd.DataFrame()

    # Merge header + line score to identify home vs away
    merged = line_score.merge(
        game_header[["GAME_ID", "GAME_DATE_EST", "HOME_TEAM_ID", "VISITOR_TEAM_ID"]],
        on="GAME_ID",
        how="left",
    )

    rows = []
    for game_id in merged["GAME_ID"].unique():
        g = merged[merged["GAME_ID"] == game_id].copy()
        if g.empty:
            continue

        game_date = pd.to_datetime(g["GAME_DATE_EST"].iloc[0], errors="coerce")

        home_id = g["HOME_TEAM_ID"].iloc[0]
        away_id = g["VISITOR_TEAM_ID"].iloc[0]

        home_row = g[g["TEAM_ID"] == home_id]
        away_row = g[g["TEAM_ID"] == away_id]

        if home_row.empty or away_row.empty:
            # Fallback – sometimes TEAM_ID mismatch
            # Just take two rows sorted by TEAM_ID as home/away
            g_sorted = g.sort_values("TEAM_ID")
            if len(g_sorted) >= 2:
                home_row = g_sorted.iloc[[0]]
                away_row = g_sorted.iloc[[1]]
            else:
                continue

        home_row = home_row.iloc[0]
        away_row = away_row.iloc[0]

        # TEAM_NAME is usually "Boston Celtics"
        home_name = home_row.get("TEAM_NAME") or home_row.get("TEAM_ABBREVIATION")
        away_name = away_row.get("TEAM_NAME") or away_row.get("TEAM_ABBREVIATION")

        rows.append(
            {
                "gameDate": game_date,
                "Home Team": home_name,
                "Away Team": away_name,
                "gameId": game_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        if verbose:
            print("[QEPC data_sources] ScoreboardV2 produced no matchups.")
        return pd.DataFrame()

    df = _normalize_today_filter(df, "gameDate", target_date)
    df = _finalize_schedule_df(df, verbose=verbose)

    if verbose and not df.empty:
        print("[QEPC data_sources] Loaded today's schedule from nba_api ScoreboardV2.")

    return df


# ---------------------------------------------------------------------------
# 3) ESPN scoreboard
# ---------------------------------------------------------------------------

def load_today_from_espn(
    target_date: Optional[date] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load today's games from ESPN's unofficial NBA scoreboard API.

    Endpoint:
      https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard
    """
    if target_date is None:
        target_date = date.today()

    try:
        import requests
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] requests not available for ESPN: {exc}")
        return pd.DataFrame()

    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    params = {"dates": target_date.strftime("%Y%m%d")}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] ESPN scoreboard error: {exc}")
        return pd.DataFrame()

    events = data.get("events", []) or []
    rows = []

    for ev in events:
        game_id = ev.get("id")
        ev_date_raw = ev.get("date")
        game_dt = pd.to_datetime(ev_date_raw, errors="coerce")

        comps = ev.get("competitions", []) or []
        if not comps:
            continue

        comp = comps[0]
        competitors = comp.get("competitors", []) or []
        if len(competitors) != 2:
            continue

        home_team_name = None
        away_team_name = None

        for c in competitors:
            team_info = c.get("team", {}) or {}
            display_name = team_info.get("displayName")  # e.g. "Boston Celtics"
            home_away = c.get("homeAway")

            if home_away == "home":
                home_team_name = display_name
            elif home_away == "away":
                away_team_name = display_name

        if not home_team_name or not away_team_name:
            continue

        rows.append(
            {
                "gameDate": game_dt,
                "Home Team": home_team_name,
                "Away Team": away_team_name,
                "gameId": game_id,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        if verbose:
            print("[QEPC data_sources] ESPN scoreboard returned no games.")
        return pd.DataFrame()

    df = _normalize_today_filter(df, "gameDate", target_date)
    df = _finalize_schedule_df(df, verbose=verbose)

    if verbose and not df.empty:
        print("[QEPC data_sources] Loaded today's schedule from ESPN scoreboard.")

    return df


# ---------------------------------------------------------------------------
# 4) Local data/Games.csv
# ---------------------------------------------------------------------------

def load_today_from_local_csv(
    target_date: Optional[date] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load today's games from the canonical local schedule: data/Games.csv
    (and its 'gameDate' column, which your diagnostics already verified).
    """
    if target_date is None:
        target_date = date.today()

    games_path = Path(get_games_path())
    if not games_path.exists():
        if verbose:
            print(f"[QEPC data_sources] Local Games.csv not found at {games_path}.")
        return pd.DataFrame()

    try:
        df = pd.read_csv(games_path)
    except Exception as exc:
        if verbose:
            print(f"[QEPC data_sources] Failed to read {games_path}: {exc}")
        return pd.DataFrame()

    # Ensure we have a gameDate column
    if "gameDate" not in df.columns:
        # Build it from Date if needed
        if "Date" in df.columns:
            df = df.copy()
            df["gameDate"] = pd.to_datetime(df["Date"], errors="coerce")
        else:
            if verbose:
                print("[QEPC data_sources] Local Games.csv missing gameDate/Date columns.")
            return pd.DataFrame()

    df_today = _normalize_today_filter(df, "gameDate", target_date)
    if df_today.empty:
        if verbose:
            print(f"[QEPC data_sources] No local games for {target_date.isoformat()} in Games.csv.")
        return pd.DataFrame()

    # Map columns to canonical names
    if "Home Team" not in df_today.columns and "Home_Team" in df_today.columns:
        df_today["Home Team"] = df_today["Home_Team"]
    if "Away Team" not in df_today.columns and "Away_Team" in df_today.columns:
        df_today["Away Team"] = df_today["Away_Team"]

    df_today = df_today.rename(
        columns={
            "Home Team": "Home Team",
            "Away Team": "Away Team",
        }
    )

    df_today = _finalize_schedule_df(df_today, verbose=verbose)

    if verbose and not df_today.empty:
        print(f"[QEPC data_sources] Loaded schedule from local Games.csv at {games_path}.")

    return df_today


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def get_today_schedule(
    target_date: Optional[date] = None,
    priority: tuple[str, ...] = ("live_csv", "nba_api", "espn", "local_csv"),
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Master function: try multiple sources in priority order and return
    a standardized schedule DataFrame.

    Canonical output columns:
        gameDate, Home Team, Away Team, gameId
    """
    if target_date is None:
        target_date = date.today()

    sources = {
        "live_csv": load_today_from_live_csv,
        "nba_api": load_today_from_nba_api,
        "espn": load_today_from_espn,
        "local_csv": load_today_from_local_csv,
    }

    for key in priority:
        func = sources.get(key)
        if func is None:
            continue

        if verbose:
            print(f"[QEPC data_sources] Trying source: {key}")

        df = func(target_date=target_date, verbose=verbose)
        if df is not None and not df.empty:
            if verbose:
                print(f"[QEPC data_sources] Using schedule from source: {key}")
            return df

    if verbose:
        print("[QEPC data_sources] No schedule found from any source; returning empty DataFrame.")

    return pd.DataFrame(columns=["gameDate", "Home Team", "Away Team", "gameId"])
