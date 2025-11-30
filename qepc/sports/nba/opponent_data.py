"""
QEPC Module: opponent_data.py

Safe, local-only opponent defensive metrics.

The older version tried to use the nba_api `leaguedashopponentstats` endpoint,
which no longer exists in newer nba_api builds and caused import failures.

This version:
- Loads local team stats (Team_Stats.csv) using the data directory inferred
  from qepc.autoload.paths.get_games_path().
- Extracts a "DRtg"-like defensive rating per team.
- Returns a small DataFrame used by props.py and (optionally) strengths_v2.
"""

from __future__ import annotations

from typing import Optional, List

import pandas as pd

from qepc.autoload.paths import get_games_path


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Find the first column in df whose name matches one of the candidates
    (case-sensitive first, then case-insensitive).
    """
    for cand in candidates:
        if cand in df.columns:
            return cand
    lower_map = {str(c).lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def process_opponent_metrics() -> pd.DataFrame:
    """
    Return a DataFrame with columns:

        - Team
        - DRtg  (defensive rating or closest available proxy)

    Data source:
        data/raw/Team_Stats.csv, where "data" is the directory that
        contains Games.csv (resolved via get_games_path()).

    If we cannot find a suitable DRtg column, we fall back to an empty
    DataFrame and let callers handle defaults.
    """
    data_dir = get_games_path().parent
    team_stats_path = data_dir / "raw" / "Team_Stats.csv"

    try:
        df = pd.read_csv(team_stats_path)
    except Exception as exc:
        print(f"[opponent_data] Failed to read Team_Stats from {team_stats_path}: {exc}")
        return pd.DataFrame(columns=["Team", "DRtg"])

    if df.empty:
        print("[opponent_data] Team_Stats.csv is empty.")
        return pd.DataFrame(columns=["Team", "DRtg"])

    team_col = _find_column(df, ["Team", "TEAM_NAME", "team_name"])
    if not team_col:
        print("[opponent_data] Could not find a Team column in Team_Stats.csv.")
        return pd.DataFrame(columns=["Team", "DRtg"])

    # Try to find a defensive rating-like column
    drtg_col = _find_column(
        df,
        ["DRtg", "DEF_RATING", "DRTG", "DefRtg", "DEFRTG", "DEF_RTG"],
    )
    if not drtg_col:
        print(
            "[opponent_data] Could not find a defensive rating column in Team_Stats.csv. "
            "Returning empty metrics; callers should handle with a default (e.g., league average)."
        )
        return pd.DataFrame(columns=["Team", "DRtg"])

    opp_df = df[[team_col, drtg_col]].copy()
    opp_df.rename(columns={team_col: "Team", drtg_col: "DRtg"}, inplace=True)

    # Drop obvious duplicates
    opp_df = opp_df.drop_duplicates(subset=["Team"]).reset_index(drop=True)

    print(f"[opponent_data] Loaded opponent metrics for {len(opp_df)} teams from local Team_Stats.")
    return opp_df
