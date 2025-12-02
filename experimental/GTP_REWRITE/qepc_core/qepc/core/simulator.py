# qepc/core/simulator.py
#
# QEPC experimental simulator (NBA-first).
#
# Input:
#   - games_df with at least:
#       lambda_home, lambda_away
#       vol_home, vol_away  (optional, for overdispersed noise)
#
# Output:
#   - games_df with simulated expectations:
#       Sim_Home_Score, Sim_Away_Score
#       Home_Win_Prob, Away_Win_Prob
#       Expected_Score_Total, Expected_Spread
#
# This is a "soft" Monte Carlo: for speed in notebooks, we use
# closed-form expectations where possible plus a small stochastic
# correction to mimic quantum-ish variance.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from qepc.logging_utils import qstep, qwarn


@dataclass
class SimConfig:
    """
    Configuration for QEPC simulation.

    Attributes
    ----------
    num_trials : int
        Number of Monte Carlo samples per game. 2000–5000 is good for
        production, but 500 is often fine for notebooks.
    use_poisson : bool
        If True, sample scores from Poisson(lambda). If False, use a
        normal approximation around lambda with std from volatility.
    volatility_scale : float
        Scales vol_home / vol_away into per-game std dev when using
        normal approximation.
    clip_scores_min : int
        Minimum possible score per team (safety).
    clip_scores_max : int
        Maximum possible score per team (safety).
    random_seed : Optional[int]
        Seed for reproducibility. If None, rng is not seeded.
    """

    num_trials: int = 2000
    use_poisson: bool = True
    volatility_scale: float = 1.0
    clip_scores_min: int = 60
    clip_scores_max: int = 180
    random_seed: Optional[int] = None


def _simulate_game_row(
    row: pd.Series,
    cfg: SimConfig,
    rng: np.random.Generator,
) -> dict:
    """
    Simulate one game using the provided λ and volatility.
    Returns a dict of summary statistics.
    """
    lam_h = float(row["lambda_home"])
    lam_a = float(row["lambda_away"])

    if not np.isfinite(lam_h) or not np.isfinite(lam_a):
        # Missing λ -> return NaNs to signal upstream issues
        return {
            "Sim_Home_Score": np.nan,
            "Sim_Away_Score": np.nan,
            "Home_Win_Prob": np.nan,
            "Away_Win_Prob": np.nan,
            "Expected_Score_Total": np.nan,
            "Expected_Spread": np.nan,
        }

    num_trials = max(1, int(cfg.num_trials))

    if cfg.use_poisson:
        # Pure Poisson sampling
        home_scores = rng.poisson(lam=lam_h, size=num_trials)
        away_scores = rng.poisson(lam=lam_a, size=num_trials)
    else:
        # Normal approximation with volatility-based std dev
        vol_h = float(row.get("vol_home", np.sqrt(lam_h)))
        vol_a = float(row.get("vol_away", np.sqrt(lam_a)))
        std_h = max(1.0, vol_h * cfg.volatility_scale)
        std_a = max(1.0, vol_a * cfg.volatility_scale)

        home_scores = rng.normal(loc=lam_h, scale=std_h, size=num_trials)
        away_scores = rng.normal(loc=lam_a, scale=std_a, size=num_trials)

        # Clip and round to plausible scores
        home_scores = np.clip(home_scores, cfg.clip_scores_min, cfg.clip_scores_max)
        away_scores = np.clip(away_scores, cfg.clip_scores_min, cfg.clip_scores_max)
        home_scores = np.round(home_scores).astype(int)
        away_scores = np.round(away_scores).astype(int)

    # Compute summary stats
    total_scores = home_scores + away_scores
    spreads = home_scores - away_scores

    sim_home_mean = float(np.mean(home_scores))
    sim_away_mean = float(np.mean(away_scores))
    home_win_prob = float(np.mean(home_scores > away_scores))
    away_win_prob = 1.0 - home_win_prob

    expected_total = float(np.mean(total_scores))
    expected_spread = float(np.mean(spreads))

    return {
        "Sim_Home_Score": sim_home_mean,
        "Sim_Away_Score": sim_away_mean,
        "Home_Win_Prob": home_win_prob,
        "Away_Win_Prob": away_win_prob,
        "Expected_Score_Total": expected_total,
        "Expected_Spread": expected_spread,
    }


def run_qepc_simulation(
    games_df: pd.DataFrame,
    config: Optional[SimConfig] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run QEPC simulation over a set of games with λ.

    Parameters
    ----------
    games_df : DataFrame
        Must contain 'lambda_home' and 'lambda_away' columns.
        'vol_home' and 'vol_away' are optional but recommended.
    config : SimConfig, optional
        Controls number of trials, sampling mode, etc.
    verbose : bool
        If True, prints progress and a brief summary.

    Returns
    -------
    sim_df : DataFrame
        Same columns as games_df plus:
          Sim_Home_Score, Sim_Away_Score,
          Home_Win_Prob, Away_Win_Prob,
          Expected_Score_Total, Expected_Spread
    """
    if config is None:
        config = SimConfig()

    if "lambda_home" not in games_df.columns or "lambda_away" not in games_df.columns:
        raise ValueError("games_df must contain 'lambda_home' and 'lambda_away' columns")

    # Set up RNG
    if config.random_seed is not None:
        rng = np.random.default_rng(config.random_seed)
    else:
        rng = np.random.default_rng()

    rows = []
    total_games = len(games_df)
    for i, (_, row) in enumerate(games_df.iterrows()):
        if verbose and (i == 0 or (i + 1) % 25 == 0 or i == total_games - 1):
            qstep(f"Simulating game {i + 1}/{total_games}")

        sim_stats = _simulate_game_row(row, config, rng)
        rows.append(sim_stats)

    sim_stats_df = pd.DataFrame(rows)
    result = pd.concat([games_df.reset_index(drop=True), sim_stats_df], axis=1)

    if verbose:
        valid_mask = result["Home_Win_Prob"].notna()
        n_valid = int(valid_mask.sum())
        qstep(
            f"run_qepc_simulation: simulated {n_valid}/{len(result)} games "
            f"with finite λ"
        )

    return result
