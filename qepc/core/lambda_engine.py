"""
QEPC Module: lambda_engine.py
=============================

Compute expected scores (lambda) for NBA games.

Key features
------------
- Dynamic home court advantage (HCA) by team.
- Rest-day advantage between teams.
- Back-to-back penalties.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# CONFIGURATION
# =============================================================================

LEAGUE_AVG_POINTS = 114.0  # 2024-25 NBA average estimate

# Base home court advantage multiplier
BASE_HCA = 1.028

# Team-specific HCA adjustments on top of BASE_HCA
TEAM_HCA_BOOST = {
    # Strong home courts (altitude, crowd, travel)
    "Denver Nuggets": 1.02,
    "Utah Jazz": 1.015,
    "Miami Heat": 1.01,

    # Slightly weaker home courts
    "Brooklyn Nets": 0.99,
    "Los Angeles Clippers": 0.99,
}

# Rest advantage
REST_ADVANTAGE_PER_DAY = 1.5   # points per rest-day difference
MAX_REST_ADVANTAGE = 4.0       # cap the advantage (in points)

# Back-to-back penalty
B2B_PENALTY = 0.97  # 3% scoring reduction on second night of B2B


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_team_hca(team: str) -> float:
    """
    Get the home court advantage multiplier for a team.

    Returns a factor >= 0 (1.0 = neutral).
    """
    base_boost = TEAM_HCA_BOOST.get(team, 1.0)
    return BASE_HCA * base_boost


def _calculate_rest_factor(
    rest_days_home: Optional[float],
    rest_days_away: Optional[float],
) -> Tuple[float, float]:
    """
    Calculate rest-based adjustments for both teams.

    Returns
    -------
    (home_factor, away_factor)
        Multiplicative scoring factors for home and away teams.
    """
    if rest_days_home is None or rest_days_away is None:
        return (1.0, 1.0)

    # Rest advantage is relative
    rest_diff = rest_days_home - rest_days_away

    point_swing = np.clip(
        rest_diff * REST_ADVANTAGE_PER_DAY,
        -MAX_REST_ADVANTAGE,
        MAX_REST_ADVANTAGE,
    )

    # Convert point swing into multiplicative factors
    home_factor = 1.0 + (point_swing / LEAGUE_AVG_POINTS / 2.0)
    away_factor = 1.0 - (point_swing / LEAGUE_AVG_POINTS / 2.0)

    return (home_factor, away_factor)


# =============================================================================
# MAIN FUNCTIONS
# =============================================================================

def compute_lambda(
    schedule_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    include_situational: bool = True,
) -> pd.DataFrame:
    """
    Compute expected scores (lambda) for each game.

    Formula (simplified)
    --------------------
    λ_home = LEAGUE_AVG
             × home_off_ratio
             × away_def_ratio
             × pace_factor
             × HCA
             × situational_factors

    λ_away is similar but without HCA.

    Parameters
    ----------
    schedule_df : DataFrame
        Must contain at least:
            'Home Team', 'Away Team'
        Optional situational columns:
            'rest_days_home', 'rest_days_away',
            'home_b2b', 'away_b2b'
    team_stats_df : DataFrame
        Must contain:
            'Team', 'ORtg', 'DRtg', 'Pace', 'Volatility'
    include_situational : bool
        If True, uses rest-day and B2B info if present.

    Returns
    -------
    DataFrame
        schedule_df with extra columns:
            lambda_home, lambda_away, vol_home, vol_away
    """
    if team_stats_df.empty:
        print("[QEPC Lambda] ERROR: Cannot compute without team strengths.")
        return schedule_df

    required_schedule_cols = {"Home Team", "Away Team"}
    if not required_schedule_cols.issubset(schedule_df.columns):
        missing = required_schedule_cols - set(schedule_df.columns)
        raise ValueError(f"schedule_df missing required columns: {missing}")

    # League averages for normalization
    league_avg_ortg = float(team_stats_df["ORtg"].mean())
    league_avg_drtg = float(team_stats_df["DRtg"].mean())
    league_avg_pace = float(team_stats_df["Pace"].mean())

    # Build lookup dict for fast access
    strengths = {}
    for _, row in team_stats_df.iterrows():
        team_name = str(row["Team"])
        strengths[team_name] = {
            "off_ratio": float(row["ORtg"]) / league_avg_ortg,
            "def_ratio": float(row["DRtg"]) / league_avg_drtg,  # >1 = worse defense
            "pace_ratio": float(row["Pace"]) / league_avg_pace,
            "volatility": float(row.get("Volatility", 11.0)),
        }

    df = schedule_df.copy()
    df["lambda_home"] = 0.0
    df["lambda_away"] = 0.0
    df["vol_home"] = 0.0
    df["vol_away"] = 0.0

    missing_teams = set()

    for index, row in df.iterrows():
        home_team = str(row["Home Team"])
        away_team = str(row["Away Team"])

        if home_team not in strengths:
            missing_teams.add(home_team)
            continue
        if away_team not in strengths:
            missing_teams.add(away_team)
            continue

        home = strengths[home_team]
        away = strengths[away_team]

        # ---------------------------------------------------------------------
        # PACE FACTOR
        # ---------------------------------------------------------------------
        game_pace = home["pace_ratio"] * away["pace_ratio"]

        # ---------------------------------------------------------------------
        # HOME COURT ADVANTAGE
        # ---------------------------------------------------------------------
        hca = _get_team_hca(home_team)

        # ---------------------------------------------------------------------
        # SITUATIONAL FACTORS
        # ---------------------------------------------------------------------
        home_situational = 1.0
        away_situational = 1.0

        if include_situational:
            # Rest days (treat NaN as "no data")
            rest_home_raw = row.get("rest_days_home", None)
            rest_away_raw = row.get("rest_days_away", None)

            rest_home = (
                float(rest_home_raw)
                if rest_home_raw is not None and pd.notna(rest_home_raw)
                else None
            )
            rest_away = (
                float(rest_away_raw)
                if rest_away_raw is not None and pd.notna(rest_away_raw)
                else None
            )

            if rest_home is not None and rest_away is not None:
                h_rest, a_rest = _calculate_rest_factor(rest_home, rest_away)
                home_situational *= h_rest
                away_situational *= a_rest

            # Back-to-back flags: treat NaN/None as False
            home_b2b = row.get("home_b2b", False)
            away_b2b = row.get("away_b2b", False)

            if home_b2b is True:
                home_situational *= B2B_PENALTY
            if away_b2b is True:
                away_situational *= B2B_PENALTY

        # ---------------------------------------------------------------------
        # LAMBDA CALCULATION
        # ---------------------------------------------------------------------
        lambda_home = (
            LEAGUE_AVG_POINTS
            * home["off_ratio"]
            * away["def_ratio"]
            * game_pace
            * hca
            * home_situational
        )

        lambda_away = (
            LEAGUE_AVG_POINTS
            * away["off_ratio"]
            * home["def_ratio"]
            * game_pace
            * away_situational
        )

        df.loc[index, "lambda_home"] = lambda_home
        df.loc[index, "lambda_away"] = lambda_away
        df.loc[index, "vol_home"] = home["volatility"]
        df.loc[index, "vol_away"] = away["volatility"]

    if missing_teams:
        print(f"[QEPC Lambda] WARNING: {len(missing_teams)} teams not found in strengths:")
        for t in sorted(missing_teams)[:5]:
            print(f"  - {t}")
        if len(missing_teams) > 5:
            print(f"  ... and {len(missing_teams) - 5} more")

    valid_games = int((df["lambda_home"] > 0).sum())
    print(f"[QEPC Lambda] Computed λ for {valid_games}/{len(df)} games.")

    return df


def compute_lambda_simple(
    schedule_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Simplified version without situational factors.
    """
    return compute_lambda(schedule_df, team_stats_df, include_situational=False)
