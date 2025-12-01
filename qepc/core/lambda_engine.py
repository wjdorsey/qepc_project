"""
QEPC Module: lambda_engine.py
-----------------------------
Core λ-builder for team scoring expectations.

Takes:
  - schedule_df: games with "Home Team", "Away Team" (and optionally rest/travel flags)
  - team_stats_df: team strength table with columns:
      Team, ORtg, DRtg, Pace, Volatility

Produces:
  - lambda_home, lambda_away  (expected points per team)
  - vol_home, vol_away        (volatility proxy)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qepc.core.model_config import (
    BASE_HCA,
    TEAM_HCA_BOOST,
    REST_ADVANTAGE_PER_DAY,
    MAX_REST_ADVANTAGE,
    B2B_PENALTY,
    TRAVEL_PENALTY_PER_1000MI,
    LEAGUE_AVG_POINTS,
    QUANTUM_NOISE_STD,
    LAMBDA_TOTAL_SCALE,  # <-- NEW: global calibration factor
)

# Optional: real distances if you want to expand this later
CITY_DISTANCES = {
    # Example: great-circle approx between cities (miles)
    # "Denver Nuggets": {"Los Angeles Lakers": 831},
}


def _get_team_hca(team: str) -> float:
    """
    Home-court advantage multiplier.

    BASE_HCA is the global baseline, TEAM_HCA_BOOST allows per-team tweaks.
    """
    return BASE_HCA * TEAM_HCA_BOOST.get(team, 1.0)


def _calculate_rest_factor(rest_home: float, rest_away: float) -> tuple[float, float]:
    """
    Translate rest days differential into scoring multipliers for home/away.
    """
    diff = float(rest_home) - float(rest_away)
    advantage = np.clip(
        diff * REST_ADVANTAGE_PER_DAY,
        -MAX_REST_ADVANTAGE,
        MAX_REST_ADVANTAGE,
    )

    # Convert extra points advantage into multiplicative factor around league average
    home_adjust = 1.0 + (advantage / LEAGUE_AVG_POINTS)
    away_adjust = 1.0 - (advantage / LEAGUE_AVG_POINTS)
    return home_adjust, away_adjust


def _get_travel_penalty(home_team: str, away_team: str) -> float:
    """
    Basic travel penalty factor for the away team only.

    Distance table is optional and can be expanded. If not found, returns 1.0.
    """
    distance = CITY_DISTANCES.get(home_team, {}).get(away_team, 0.0)
    penalty = (distance / 1000.0) * TRAVEL_PENALTY_PER_1000MI

    # Never reduce a team below 90% of expected scoring from travel alone
    return max(1.0 - penalty, 0.9)


def compute_lambda(
    schedule_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    include_situational: bool = True,
) -> pd.DataFrame:
    """
    Compute per-game scoring lambdas (expected points) for each matchup.

    Parameters
    ----------
    schedule_df : DataFrame
        Must have columns:
          - "Home Team"
          - "Away Team"
        Can optionally include:
          - "home_rest_days", "away_rest_days"
          - "home_b2b", "away_b2b"
          - anything else you use in the future

    team_stats_df : DataFrame
        Team strengths with columns:
          - "Team" (must match schedule names)
          - "ORtg"  (offensive rating, per-100)
          - "DRtg"  (defensive rating, per-100)
          - "Pace"  (points-per-team-per-game proxy)
          - "Volatility" (std dev of scores; used as sigma)

    include_situational : bool
        If True, apply rest / B2B / travel adjustments.

    Returns
    -------
    DataFrame
        Copy of schedule_df with new columns:
          - lambda_home, lambda_away
          - vol_home, vol_away
    """
    df = schedule_df.copy()

    # Initialize columns
    df["lambda_home"] = 0.0
    df["lambda_away"] = 0.0
    df["vol_home"] = 0.0
    df["vol_away"] = 0.0

    # Small helper for safe team lookup
    team_stats = team_stats_df.set_index("Team")

    count_with_lambdas = 0

    for index, row in df.iterrows():
        home_name = row.get("Home Team")
        away_name = row.get("Away Team")

        if home_name not in team_stats.index or away_name not in team_stats.index:
            # Skip if we don't have strengths for either team
            continue

        home = team_stats.loc[home_name]
        away = team_stats.loc[away_name]

        # ------------------------------------------------------------------
        # Base game pace: approximate points-per-team-per-game
        # ------------------------------------------------------------------
        home_pace = float(home.get("Pace", LEAGUE_AVG_POINTS))
        away_pace = float(away.get("Pace", LEAGUE_AVG_POINTS))
        game_pace = (home_pace + away_pace) / 2.0

        # ------------------------------------------------------------------
        # Extract ratings
        # ------------------------------------------------------------------
        home_ortg = float(home.get("ORtg", LEAGUE_AVG_POINTS))
        away_ortg = float(away.get("ORtg", LEAGUE_AVG_POINTS))
        home_drtg = float(home.get("DRtg", LEAGUE_AVG_POINTS))
        away_drtg = float(away.get("DRtg", LEAGUE_AVG_POINTS))

        # Safety: avoid divide-by-zero
        away_drtg_safe = max(1.0, away_drtg)
        home_drtg_safe = max(1.0, home_drtg)

        # ------------------------------------------------------------------
        # Home Court Advantage
        # ------------------------------------------------------------------
        hca = _get_team_hca(home_name)

        # ------------------------------------------------------------------
        # Situational factors (rest, B2B, travel)
        # ------------------------------------------------------------------
        home_situational = 1.0
        away_situational = 1.0

        if include_situational:
            # Rest days (default to 3 if not provided)
            rest_home = row.get("home_rest_days", 3.0)
            rest_away = row.get("away_rest_days", 3.0)
            h_rest, a_rest = _calculate_rest_factor(rest_home, rest_away)
            home_situational *= h_rest
            away_situational *= a_rest

            # Back-to-back flags (boolean or 0/1)
            home_b2b = bool(row.get("home_b2b", False))
            away_b2b = bool(row.get("away_b2b", False))
            if home_b2b:
                home_situational *= B2B_PENALTY
            if away_b2b:
                away_situational *= B2B_PENALTY

            # Travel penalty for away team
            travel_factor = _get_travel_penalty(home_name, away_name)
            away_situational *= travel_factor

        # ------------------------------------------------------------------
        # Translate ORtg/DRtg into expected points
        #
        # Intuition:
        #   ORtg / DRtg are per-100-poss metrics.
        #   We use game_pace as an effective "points per team per game" scale.
        #
        #   λ_home ≈ game_pace * (home_ORtg / away_DRtg)
        #   λ_away ≈ game_pace * (away_ORtg / home_DRtg)
        #
        # Then apply:
        #   - Home court
        #   - Situational multipliers
        #   - Global calibration factor LAMBDA_TOTAL_SCALE
        # ------------------------------------------------------------------
        base_home = game_pace * (home_ortg / away_drtg_safe)
        base_away = game_pace * (away_ortg / home_drtg_safe)

        lambda_home = base_home * hca * home_situational * LAMBDA_TOTAL_SCALE
        lambda_away = base_away * away_situational * LAMBDA_TOTAL_SCALE

        # Optional clamping to avoid wild outliers
        lambda_home = float(np.clip(lambda_home, 70.0, 150.0))
        lambda_away = float(np.clip(lambda_away, 70.0, 150.0))

        # ------------------------------------------------------------------
        # Quantum noise: small multiplicative jitter
        # ------------------------------------------------------------------
        noise_home = np.random.normal(1.0, QUANTUM_NOISE_STD)
        noise_away = np.random.normal(1.0, QUANTUM_NOISE_STD)

        df.at[index, "lambda_home"] = lambda_home * noise_home
        df.at[index, "lambda_away"] = lambda_away * noise_away

        df.at[index, "vol_home"] = float(home.get("Volatility", 10.0))
        df.at[index, "vol_away"] = float(away.get("Volatility", 10.0))

        count_with_lambdas += 1

    print(f"Computed real lambdas for {count_with_lambdas} games.")
    return df
