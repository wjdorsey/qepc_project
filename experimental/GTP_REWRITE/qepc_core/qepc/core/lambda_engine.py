# qepc/core/lambda_engine.py
#
# λ-builder for QEPC experimental core (NBA-first).
#
# Input:
#   - strengths_df: output of compute_team_strengths(...)
#       columns: Team, ORtg, DRtg, Pace, Volatility, Games, SeasonMin, SeasonMax
#   - schedule_df: a DataFrame with at least:
#       "Home Team", "Away Team"  (strings matching strengths_df["Team"])
#
# Output:
#   - schedule_with_lambda: original schedule columns plus:
#       lambda_home, lambda_away, vol_home, vol_away
#
# Design:
#   - We treat strengths_df["Pace"] as per-team expected points per game.
#   - We combine ORtg (offense) of home with ORtg of away to split points.
#   - We apply a simple home advantage by shifting a few points from away
#     to home (configurable).
#
# This is intentionally simple and transparent so we can tune later.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from qepc.logging_utils import qstep, qwarn


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------


@dataclass
class LambdaConfig:
    """
    Configuration for mapping team strengths to per-game λ.

    Attributes
    ----------
    home_advantage_points : float
        Average extra points we give to home team vs away, by shifting
        that many points from away to home (split as ± home_adv/2).
    min_lambda_total : float
        Minimum total points per game (safety floor).
    max_lambda_total : float
        Maximum total points per game (safety ceiling).
    use_net_rating_adjust : bool
        Placeholder for later: adjust λ slightly based on net rating diff.
        For now, we wire the hook but keep the weight small.
    net_rating_weight : float
        How many points of spread per 10 points of net rating diff.
    """

    home_advantage_points: float = 2.5
    min_lambda_total: float = 190.0
    max_lambda_total: float = 260.0

    use_net_rating_adjust: bool = True
    net_rating_weight: float = 0.5  # points of spread per 10 net rating diff


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------


def _build_team_index(strengths_df: pd.DataFrame) -> Dict[str, int]:
    """
    Build a mapping from normalized Team string -> row index in strengths_df.

    We normalize by stripping extra whitespace and lowercasing, so
    'Portland Trail Blazers' and 'portland trail blazers  ' match.
    """
    index: Dict[str, int] = {}

    for i, team in enumerate(strengths_df["Team"]):
        key = str(team).strip().lower()
        # If duplicates exist, last one wins; that's fine for NBA (1 row per team).
        index[key] = i

    return index


def _normalize_team_name(name: str) -> str:
    """Normalize team name for lookup."""
    return str(name).strip().lower()


def _compute_raw_totals(
    home_row: pd.Series,
    away_row: pd.Series,
    cfg: LambdaConfig,
) -> Tuple[float, float, float]:
    """
    Compute base λ_home, λ_away, λ_total before home-advantage shifts.

    We:
      - Use Pace_home + Pace_away as baseline game total.
      - Split that total using ORtg_home vs ORtg_away.
      - Optionally nudge spread by net rating diff.
    """
    ORtg_home = float(home_row["ORtg"])
    ORtg_away = float(away_row["ORtg"])
    Pace_home = float(home_row["Pace"])
    Pace_away = float(away_row["Pace"])

    # Baseline total from pace
    lambda_total = Pace_home + Pace_away

    # Safety clamps
    lambda_total = max(cfg.min_lambda_total, min(cfg.max_lambda_total, lambda_total))

    # Offensive share: how much of the total we give the home team
    if ORtg_home + ORtg_away > 0:
        home_share = ORtg_home / (ORtg_home + ORtg_away)
    else:
        home_share = 0.5

    lambda_home = lambda_total * home_share
    lambda_away = lambda_total - lambda_home

    # Optional: net rating adjustment
    if cfg.use_net_rating_adjust and "DRtg" in home_row and "DRtg" in away_row:
        net_home = float(home_row["ORtg"] - home_row["DRtg"])
        net_away = float(away_row["ORtg"] - away_row["DRtg"])
        net_diff = net_home - net_away  # positive if home is stronger overall

        # scale to a modest spread shift (points)
        spread_adjust = (net_diff / 10.0) * cfg.net_rating_weight  # points
        # Apply by shifting half from away to home
        lambda_home += spread_adjust / 2.0
        lambda_away -= spread_adjust / 2.0

    return lambda_home, lambda_away, lambda_total


def _apply_home_advantage(
    lambda_home: float,
    lambda_away: float,
    cfg: LambdaConfig,
) -> Tuple[float, float]:
    """
    Apply home advantage by shifting some points from away to home,
    keeping the total roughly similar.
    """
    shift = cfg.home_advantage_points / 2.0
    # Only apply if away has enough points to spare
    if lambda_away > shift:
        lambda_home += shift
        lambda_away -= shift
    return lambda_home, lambda_away


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------


def compute_lambda_for_schedule(
    schedule_df: pd.DataFrame,
    strengths_df: pd.DataFrame,
    config: Optional[LambdaConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Attach λ_home, λ_away, vol_home, vol_away to a schedule.

    Parameters
    ----------
    schedule_df : DataFrame
        Must contain at least:
          - "Home Team"
          - "Away Team"
        with values that match strengths_df["Team"].
    strengths_df : DataFrame
        Output of compute_team_strengths (one row per team).
    config : LambdaConfig, optional
        Controls home advantage, total clamps, net rating weight.
    verbose : bool
        If True, prints summary and counts of missing teams.

    Returns
    -------
    schedule_with_lambda : DataFrame
        Original schedule columns plus:
          - lambda_home, lambda_away
          - vol_home, vol_away
          - has_strengths (bool flag)
    """
    if config is None:
        config = LambdaConfig()

    if "Team" not in strengths_df.columns:
        raise ValueError("strengths_df must have a 'Team' column")

    missing_cols = [c for c in ["Home Team", "Away Team"] if c not in schedule_df.columns]
    if missing_cols:
        raise ValueError(f"schedule_df is missing required columns: {missing_cols}")

    # Build index for fast lookup
    team_index = _build_team_index(strengths_df)

    rows = []
    missing_home = 0
    missing_away = 0

    for _, game in schedule_df.iterrows():
        home_name = game["Home Team"]
        away_name = game["Away Team"]

        home_key = _normalize_team_name(home_name)
        away_key = _normalize_team_name(away_name)

        i_home = team_index.get(home_key, None)
        i_away = team_index.get(away_key, None)

        if i_home is None:
            missing_home += 1
        if i_away is None:
            missing_away += 1

        if i_home is None or i_away is None:
            rows.append(
                {
                    **game.to_dict(),
                    "lambda_home": np.nan,
                    "lambda_away": np.nan,
                    "vol_home": np.nan,
                    "vol_away": np.nan,
                    "has_strengths": False,
                }
            )
            continue

        home_row = strengths_df.iloc[i_home]
        away_row = strengths_df.iloc[i_away]

        lambda_home, lambda_away, _ = _compute_raw_totals(home_row, away_row, config)
        lambda_home, lambda_away = _apply_home_advantage(lambda_home, lambda_away, config)

        vol_home = float(home_row.get("Volatility", np.nan))
        vol_away = float(away_row.get("Volatility", np.nan))

        rows.append(
            {
                **game.to_dict(),
                "lambda_home": float(lambda_home),
                "lambda_away": float(lambda_away),
                "vol_home": vol_home,
                "vol_away": vol_away,
                "has_strengths": True,
            }
        )

    result = pd.DataFrame(rows)

    if verbose:
        qstep(
            f"compute_lambda_for_schedule: {len(result)} games, "
            f"{result['has_strengths'].sum()} with full strengths, "
            f"{len(result) - result['has_strengths'].sum()} missing"
        )
        if missing_home or missing_away:
            qwarn(
                f"Missing strengths for {missing_home} home-team lookups "
                f"and {missing_away} away-team lookups"
            )

    return result
