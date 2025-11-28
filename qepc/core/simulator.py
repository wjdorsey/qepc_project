"""
QEPC Module: simulator.py
=========================

Monte Carlo game simulator for NBA scores.

Features
--------
- Poisson or Normal-based score generation.
- Optional correlation between home/away scores (pace effect).
- Handles OT by splitting tied games.
- "Fast" vectorized version for large backtests.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

DEFAULT_NUM_TRIALS = 20000

# Correlation between home/away scores (pace effect).
SCORE_CORRELATION = 0.35


def run_qepc_simulation(
    df: pd.DataFrame,
    num_trials: int = DEFAULT_NUM_TRIALS,
    use_normal: bool = False,
    correlated_scores: bool = True,
) -> pd.DataFrame:
    """
    Run QEPC Monte Carlo simulation for each game in the schedule.

    Parameters
    ----------
    df : DataFrame
        Must have columns: lambda_home, lambda_away.
        Optionally: vol_home, vol_away.
    num_trials : int
        Number of Monte Carlo trials per game.
    use_normal : bool
        If True, use Normal distribution for scores (often better for NBA).
        If False, use a Poisson-style approach with random lambda variation.
    correlated_scores : bool
        If True, correlate home/away scores (pace factor affects both teams).

    Returns
    -------
    DataFrame
        Original df plus:
          Home_Win_Prob, Away_Win_Prob, Tie_Prob,
          Expected_Score_Total, Expected_Spread,
          Sim_Home_Score, Sim_Away_Score
    """
    if df.empty or "lambda_home" not in df.columns or "lambda_away" not in df.columns:
        print("[QEPC Simulator] ERROR: Missing required columns.")
        return df

    df = df.copy()

    # Initialize result columns
    df["Home_Win_Prob"] = 0.0
    df["Away_Win_Prob"] = 0.0
    df["Tie_Prob"] = 0.0
    df["Expected_Score_Total"] = 0.0
    df["Expected_Spread"] = 0.0
    df["Sim_Home_Score"] = 0.0
    df["Sim_Away_Score"] = 0.0

    mode = "Normal" if use_normal else "Poisson"
    corr_str = "Correlated" if correlated_scores else "Independent"
    print(f"[QEPC Simulator] Running {num_trials} trials ({mode}, {corr_str})...")

    # Used in Poisson branch to vary lambda
    lambda_scale = 0.1  # 10% lambda variation per 1 std of Z

    for index, row in df.iterrows():
        lambda_home = float(row["lambda_home"])
        lambda_away = float(row["lambda_away"])

        # Default volatility from Poisson theory if not provided
        vol_home = float(row.get("vol_home", np.sqrt(lambda_home)))
        vol_away = float(row.get("vol_away", np.sqrt(lambda_away)))

        # Ensure a minimum volatility (scores shouldn't be perfectly fixed)
        vol_home = max(vol_home, 8.0) if vol_home > 0 else np.sqrt(lambda_home)
        vol_away = max(vol_away, 8.0) if vol_away > 0 else np.sqrt(lambda_away)

        # ---------------------------------------------------------------------
        # SCORE GENERATION
        # ---------------------------------------------------------------------
        if correlated_scores:
            cov_matrix = [[1.0, SCORE_CORRELATION], [SCORE_CORRELATION, 1.0]]
            correlated_z = np.random.multivariate_normal(
                mean=[0.0, 0.0],
                cov=cov_matrix,
                size=num_trials,
            )
            z_home = correlated_z[:, 0]
            z_away = correlated_z[:, 1]
        else:
            z_home = np.random.standard_normal(num_trials)
            z_away = np.random.standard_normal(num_trials)

        if use_normal:
            # Normal scores: mean = lambda, std = volatility
            home_scores = lambda_home + vol_home * z_home
            away_scores = lambda_away + vol_away * z_away

            # Floor scores to a realistic minimum
            home_scores = np.maximum(home_scores, 50.0)
            away_scores = np.maximum(away_scores, 50.0)
        else:
            # Poisson with volatility-adjusted lambda
            if vol_home > 0:
                home_lambdas = lambda_home * (1 + lambda_scale * z_home)
                home_lambdas = np.maximum(home_lambdas, 50.0)
            else:
                home_lambdas = np.full(num_trials, lambda_home)

            if vol_away > 0:
                away_lambdas = lambda_away * (1 + lambda_scale * z_away)
                away_lambdas = np.maximum(away_lambdas, 50.0)
            else:
                away_lambdas = np.full(num_trials, lambda_away)

            home_scores = np.random.poisson(home_lambdas)
            away_scores = np.random.poisson(away_lambdas)

        # ---------------------------------------------------------------------
        # OUTCOME CALCULATION
        # ---------------------------------------------------------------------
        home_scores_int = np.round(home_scores).astype(int)
        away_scores_int = np.round(away_scores).astype(int)

        home_wins = int(np.sum(home_scores_int > away_scores_int))
        away_wins = int(np.sum(away_scores_int > home_scores_int))
        ties = int(np.sum(home_scores_int == away_scores_int))

        # In NBA, ties go to OT. We'll treat OT as 50/50.
        ot_home_wins = ties // 2
        ot_away_wins = ties - ot_home_wins

        home_wins += ot_home_wins
        away_wins += ot_away_wins

        df.loc[index, "Home_Win_Prob"] = home_wins / num_trials
        df.loc[index, "Away_Win_Prob"] = away_wins / num_trials
        df.loc[index, "Tie_Prob"] = 0.0  # No ties in final results
        df.loc[index, "Expected_Score_Total"] = float(
            np.mean(home_scores + away_scores)
        )
        df.loc[index, "Expected_Spread"] = float(
            np.mean(home_scores - away_scores)
        )
        df.loc[index, "Sim_Home_Score"] = float(np.mean(home_scores))
        df.loc[index, "Sim_Away_Score"] = float(np.mean(away_scores))

    print("[QEPC Simulator] Simulation complete.")
    return df


def run_qepc_simulation_fast(
    df: pd.DataFrame,
    num_trials: int = DEFAULT_NUM_TRIALS,
) -> pd.DataFrame:
    """
    Faster vectorized version for large backtests.

    Uses a Normal distribution with correlation for all games.

    Parameters
    ----------
    df : DataFrame
        Must have lambda_home, lambda_away columns.
        Optionally vol_home, vol_away.
    num_trials : int
        Number of simulations per game.

    Returns
    -------
    DataFrame
        df with additional simulation columns.
    """
    if df.empty or "lambda_home" not in df.columns or "lambda_away" not in df.columns:
        return df

    df = df.copy()
    n_games = len(df)

    lambda_home = df["lambda_home"].astype(float).values
    lambda_away = df["lambda_away"].astype(float).values

    vol_home = df.get("vol_home", pd.Series(np.sqrt(lambda_home))).astype(float).values
    vol_away = df.get("vol_away", pd.Series(np.sqrt(lambda_away))).astype(float).values

    vol_home = np.maximum(vol_home, 8.0)
    vol_away = np.maximum(vol_away, 8.0)

    cov = [[1.0, SCORE_CORRELATION], [SCORE_CORRELATION, 1.0]]

    results = {
        "Home_Win_Prob": np.zeros(n_games),
        "Away_Win_Prob": np.zeros(n_games),
        "Expected_Spread": np.zeros(n_games),
        "Expected_Score_Total": np.zeros(n_games),
        "Sim_Home_Score": np.zeros(n_games),
        "Sim_Away_Score": np.zeros(n_games),
    }

    for i in range(n_games):
        z = np.random.multivariate_normal([0.0, 0.0], cov, num_trials)

        home_scores = lambda_home[i] + vol_home[i] * z[:, 0]
        away_scores = lambda_away[i] + vol_away[i] * z[:, 1]

        home_scores = np.maximum(home_scores, 50.0)
        away_scores = np.maximum(away_scores, 50.0)

        home_wins = int(np.sum(home_scores > away_scores))
        ties = int(np.sum(np.abs(home_scores - away_scores) < 0.5))
        home_wins += ties // 2

        results["Home_Win_Prob"][i] = home_wins / num_trials
        results["Away_Win_Prob"][i] = 1.0 - results["Home_Win_Prob"][i]
        results["Expected_Spread"][i] = float(np.mean(home_scores - away_scores))
        results["Expected_Score_Total"][i] = float(np.mean(home_scores + away_scores))
        results["Sim_Home_Score"][i] = float(np.mean(home_scores))
        results["Sim_Away_Score"][i] = float(np.mean(away_scores))

    for col, vals in results.items():
        df[col] = vals

    df["Tie_Prob"] = 0.0
    return df
