# qepc/core/strengths.py
#
# Team strengths engine for QEPC experimental core (NBA-first).
#
# Input: output of qepc.nba.data_loaders.load_nba_team_logs(...)
# Output: strengths_df with one row per team:
#   Team, SeasonMin, SeasonMax, Games, ORtg, DRtg, Pace, Volatility

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qepc.logging_utils import qstep, qwarn


# ---------------------------------------------------------------------
# Config + helpers
# ---------------------------------------------------------------------


@dataclass
class TeamStrengthsConfig:
    lookback_seasons: int = 3          # how many seasons back from latest
    min_games: int = 50                # minimum games required for a team
    recency_half_life_games: int = 60  # exponential half-life in games
    use_season_year_from_season_col: bool = True  # parse "2014-15" -> 2014


def _exp_weights(n: int, half_life: int) -> np.ndarray:
    """Exponential decay weights over n observations (most recent highest)."""
    if n <= 0:
        return np.array([])
    idx = np.arange(n)[::-1]  # 0 = most recent
    lam = np.log(2.0) / max(1, half_life)
    w = np.exp(-lam * idx)
    return w / w.sum()


def _infer_season_year_column(df: pd.DataFrame, cfg: TeamStrengthsConfig) -> pd.Series:
    """
    Return a numeric 'SeasonYear' Series for lookback filters.
    Prefer parsing Season like "2014-15" -> 2014; fall back to gameDate.year.
    """
    if "Season" in df.columns and cfg.use_season_year_from_season_col:
        season_str = df["Season"].astype(str)
        year_str = season_str.str.extract(r"(\d{4})", expand=False)
        season_year = pd.to_numeric(year_str, errors="coerce")

        if season_year.notna().any():
            season_year = season_year.ffill().bfill()
            if season_year.notna().all():
                return season_year.astype(int)

        qwarn("Could not reliably parse Season; falling back to gameDate.year")

    if "gameDate" not in df.columns:
        raise ValueError("Cannot infer season year; 'gameDate' missing.")

    return df["gameDate"].dt.year


def _repair_opponent_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    If opponentScore looks basically zero everywhere (as in NBA_API_QEPC_Format),
    repair it by using the other team's teamScore within each gameId.
    """
    if "opponentScore" not in df.columns or "teamScore" not in df.columns:
        return df
    if "gameId" not in df.columns:
        return df

    df = df.copy()

    nonzero_frac = (df["opponentScore"] != 0).mean()
    if nonzero_frac > 0.1:
        # Already has meaningful opponent scores; nothing to do.
        return df

    qwarn(
        "opponentScore appears mostly zero; repairing from gameId/teamScore "
        "pairs (using the other team's score)."
    )

    def repair_group(g: pd.DataFrame) -> pd.DataFrame:
        if len(g) != 2:
            # Can't safely repair weird games; leave as-is.
            return g
        s0 = g.iloc[0]["teamScore"]
        s1 = g.iloc[1]["teamScore"]
        g.loc[g.index[0], "opponentScore"] = s1
        g.loc[g.index[1], "opponentScore"] = s0
        return g

    df = df.groupby("gameId", group_keys=False).apply(repair_group)
    return df


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------


def compute_team_strengths(
    game_logs: pd.DataFrame,
    config: Optional[TeamStrengthsConfig] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute team strengths from multi-season NBA game logs.

    Returns a DataFrame with:
        Team, SeasonMin, SeasonMax, Games, ORtg, DRtg, Pace, Volatility
    """
    if config is None:
        config = TeamStrengthsConfig()

    # Copy and repair opponent scores if needed
    df = game_logs.copy()
    df = _repair_opponent_scores(df)

    # 1) Apply cutoff date if requested
    if cutoff_date is not None:
        df = df[df["gameDate"] <= cutoff_date].copy()

    # 2) Infer numeric season year for lookback filters
    df["SeasonYear"] = _infer_season_year_column(df, config)

    latest_year = int(df["SeasonYear"].max())
    if config.lookback_seasons > 0:
        min_year = latest_year - config.lookback_seasons + 1
    else:
        min_year = int(df["SeasonYear"].min())

    df = df[(df["SeasonYear"] >= min_year) & (df["SeasonYear"] <= latest_year)].copy()

    if verbose:
        qstep(
            f"Computing team strengths from seasons {min_year}–{latest_year} "
            f"({df['SeasonYear'].nunique()} seasons, {len(df)} team-rows)"
        )

    # 3) Canonical team key
    df["TeamKey"] = df["teamCity"].astype(str).str.strip() + " " + df["teamName"].astype(str).str.strip()

    strengths_rows = []

    for team_key, group in df.groupby("TeamKey"):
        group = group.sort_values("gameDate")
        n_games = len(group)
        if n_games < config.min_games:
            continue

        w = _exp_weights(n_games, config.recency_half_life_games)

        pts_for = group["teamScore"].to_numpy(dtype=float)
        pts_against = group["opponentScore"].to_numpy(dtype=float)

        # Weighted means (points per game)
        ORtg = float((pts_for * w).sum())
        DRtg = float((pts_against * w).sum())

        total_points = pts_for + pts_against
        Pace = float((total_points * w).sum() / 2.0)  # per-team scoring rate

        # Weighted volatility of teamScore (std dev)
        mean_for = ORtg
        var_for = float(((pts_for - mean_for) ** 2 * w).sum())
        Volatility = float(np.sqrt(max(var_for, 0.0)))

        strengths_rows.append(
            {
                "Team": team_key,
                "SeasonMin": int(group["SeasonYear"].min()),
                "SeasonMax": int(group["SeasonYear"].max()),
                "Games": int(n_games),
                "ORtg": ORtg,
                "DRtg": DRtg,
                "Pace": Pace,
                "Volatility": Volatility,
            }
        )

    strengths_df = pd.DataFrame(strengths_rows).sort_values("Team").reset_index(drop=True)

    if verbose:
        qstep(f"Computed strengths for {len(strengths_df)} teams (min_games={config.min_games})")
        if not strengths_df.empty:
            desc_vol = strengths_df["Volatility"].describe()
            qstep(
                "Volatility summary: "
                f"min={desc_vol['min']:.2f}, "
                f"median={desc_vol['50%']:.2f}, "
                f"max={desc_vol['max']:.2f}"
            )

    return strengths_df
