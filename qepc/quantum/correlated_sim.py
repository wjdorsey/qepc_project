"""
QEPC Quantum: Correlated (entangled) multi-player simulations for SGPs.

This module provides tools to simulate multiple players' points together
in the *same* universe, using shared multiplicative shocks.

Idea:
    - Each simulated universe has a team-level scoring factor F_team
      that scales all players' λ on that team.
    - Each player also has an individual factor F_player.
    - Points are sampled as Poisson( λ_base * F_team * F_player ).

This creates positive correlation between teammates' outcomes because
they share F_team. You can think of F_team as a "team hot/cold state"
or "script shock" for that universe.

Future extensions can:
    - Introduce correlated F_player using entanglement matrices.
    - Extend to rebounds/assists and cross-stat SGPs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class PlayerPropConfig:
    """
    Configuration for a single player points prop in the entangled sim.

    Attributes
    ----------
    player_id : int
        Unique player identifier.
    player_name : str
        Human-readable name (for outputs).
    lambda_pts : float
        Baseline expected points (QEPC λ).
    line_pts : float
        Betting line for points (e.g., 22.5, 27.5).
    """

    player_id: int
    player_name: str
    lambda_pts: float
    line_pts: float


def _lognormal_mean_one(mu: float, sigma: float) -> float:
    """
    Helper: mean of LogNormal(mu, sigma) is exp(mu + 0.5 sigma^2).
    We often want mean ~ 1, so we set mu = -0.5 sigma^2.
    """
    return float(np.exp(mu + 0.5 * sigma * sigma))


def simulate_entangled_points(
    players: List[PlayerPropConfig],
    n_sims: int = 100_000,
    random_state: Optional[int] = None,
    team_sigma: float = 0.25,
    player_sigma: float = 0.15,
) -> Dict[str, Any]:
    """
    Simulate correlated player points using shared team-level and individual shocks.

    Parameters
    ----------
    players : list of PlayerPropConfig
        Players and their baseline λ + lines.
    n_sims : int
        Number of simulated universes.
    random_state : int or None
        Seed for reproducibility.
    team_sigma : float
        Std dev of lognormal team shock factor F_team.
        Larger values => more swingy team scoring / stronger co-movement.
    player_sigma : float
        Std dev of lognormal player shock factor F_player.
        Larger values => more player-specific volatility.

    Returns
    -------
    dict with keys:
        - 'players': list of PlayerPropConfig (input)
        - 'samples_df': DataFrame with shape (n_sims, n_players),
              columns = player_name, values = simulated points.
        - 'hit_matrix': DataFrame with True/False for each player hitting their line.
        - 'marginals': DataFrame with per-player:
              lambda_pts, line_pts, prob_over, prob_under, mean_sim, std_sim
        - 'joint': dict with:
              'n_sims', 'prob_all_over', 'prob_any_over', 'prob_none_over',
              'naive_product_all_over'
        - 'pairwise_corr': DataFrame with empirical correlations between players' points.
    """
    if len(players) == 0:
        raise ValueError("players list must not be empty.")

    rng = np.random.default_rng(random_state)

    # Team-level shock: LogNormal with mean ~ 1
    # If X ~ N(mu, sigma^2), then exp(X) has mean exp(mu + 0.5 sigma^2).
    # To get mean 1, set mu = -0.5 sigma^2.
    mu_team = -0.5 * team_sigma * team_sigma
    F_team = rng.lognormal(mean=mu_team, sigma=team_sigma, size=n_sims)

    # Player-specific shocks: independent for now (future: correlate)
    mu_player = -0.5 * player_sigma * player_sigma
    F_players = rng.lognormal(
        mean=mu_player,
        sigma=player_sigma,
        size=(n_sims, len(players)),
    )

    # Baseline lambdas and lines
    lambdas = np.array([p.lambda_pts for p in players], dtype=float)
    lines = np.array([p.line_pts for p in players], dtype=float)
    names = [p.player_name for p in players]

    # Broadcast team shock over players: shape (n_sims, n_players)
    F_team_matrix = F_team[:, None]

    # Simulated lambdas
    lam_sim = lambdas[None, :] * F_team_matrix * F_players

    # Points ~ Poisson(lam_sim)
    samples = rng.poisson(lam_sim)

    # Build DataFrame of samples
    samples_df = pd.DataFrame(samples, columns=names)

    # Hit matrix: did each player go over their line in each sim?
    # For half-lines, equality is effectively impossible; we use >
    # For integer lines, you might want >=; we keep it simple as >
    hits = samples > lines[None, :]
    hit_df = pd.DataFrame(hits, columns=names)

    # Marginal stats per player
    marg_rows = []
    for idx, p in enumerate(players):
        col_samples = samples[:, idx]
        col_hits = hits[:, idx]

        prob_over = float(col_hits.mean())
        prob_under = float(1.0 - prob_over)  # ignoring exact-push detail here
        mean_sim = float(col_samples.mean())
        std_sim = float(col_samples.std(ddof=0))

        marg_rows.append(
            {
                "player_name": p.player_name,
                "lambda_pts": float(p.lambda_pts),
                "line_pts": float(p.line_pts),
                "prob_over": prob_over,
                "prob_under": prob_under,
                "mean_sim": mean_sim,
                "std_sim": std_sim,
            }
        )

    marginals_df = pd.DataFrame(marg_rows)

    # Joint probabilities
    all_over_mask = hits.all(axis=1)
    any_over_mask = hits.any(axis=1)
    none_over_mask = ~any_over_mask

    prob_all_over = float(all_over_mask.mean())
    prob_any_over = float(any_over_mask.mean())
    prob_none_over = float(none_over_mask.mean())

    # Naive independence product (product of marginals P(Over))
    naive_product_all_over = float(np.prod(marginals_df["prob_over"].values))

    joint_info = {
        "n_sims": int(n_sims),
        "prob_all_over": prob_all_over,
        "prob_any_over": prob_any_over,
        "prob_none_over": prob_none_over,
        "naive_product_all_over": naive_product_all_over,
    }

    # Empirical pairwise correlations of simulated points
    pairwise_corr = samples_df.corr()
    pairwise_corr = pairwise_corr.reset_index().rename(columns={"index": "player_name"})

    return {
        "players": players,
        "samples_df": samples_df,
        "hit_matrix": hit_df,
        "marginals": marginals_df,
        "joint": joint_info,
        "pairwise_corr": pairwise_corr,
    }
