# qepc/core/simulator.py

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd

from qepc.core.scripts import SCRIPTS, default_script_probs, GameScript
from qepc.logging_utils import qstep


def simulate_games_multiverse(
    lambda_df: pd.DataFrame,
    num_universes: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    lambda_df:
        Home_Team, Away_Team, lambda_home, lambda_away, vol_home, vol_away
    Returns:
        DataFrame with:
            Home_Team, Away_Team,
            Sim_Home_Score, Sim_Away_Score,
            Home_Win_Prob, Total_Mean, Spread_Mean
    """
    rng = np.random.default_rng(seed)
    probs = default_script_probs()
    script_names = list(SCRIPTS.keys())
    script_probs = np.array([probs[n] for n in script_names])
    script_probs /= script_probs.sum()

    out_rows = []
    for _, row in lambda_df.iterrows():
        lam_h = row["lambda_home"]
        lam_a = row["lambda_away"]
        vol_h = row["vol_home"]
        vol_a = row["vol_away"]

        # sample scripts for each universe
        scripts_idx = rng.choice(len(script_names), size=num_universes, p=script_probs)
        scores_h = np.zeros(num_universes)
        scores_a = np.zeros(num_universes)

        for i, idx in enumerate(scripts_idx):
            script: GameScript = SCRIPTS[script_names[idx]]
            # apply script modifiers
            lam_h_script = max(70.0, (lam_h + script.offense_boost) * script.pace_factor)
            lam_a_script = max(70.0, (lam_a + script.offense_boost) * script.pace_factor)

            # approximate scoring as normal around λ with variance from volatility
            # (you can swap to Poisson later; this is smoother & easier to debug)
            std_h = script.volatility_factor * vol_h
            std_a = script.volatility_factor * vol_a

            scores_h[i] = rng.normal(lam_h_script, std_h)
            scores_a[i] = rng.normal(lam_a_script, std_a)

        scores_h = np.clip(scores_h, 60, 160)
        scores_a = np.clip(scores_a, 60, 160)

        home_win_prob = float((scores_h > scores_a).mean())
        home_avg = float(scores_h.mean())
        away_avg = float(scores_a.mean())

        out_rows.append(
            {
                "Home_Team": row["Home_Team"],
                "Away_Team": row["Away_Team"],
                "Sim_Home_Score": home_avg,
                "Sim_Away_Score": away_avg,
                "Home_Win_Prob": home_win_prob,
                "Total_Mean": home_avg + away_avg,
                "Spread_Mean": home_avg - away_avg,
            }
        )

    out_df = pd.DataFrame(out_rows)
    qstep(f"Simulated multiverse for {len(out_df)} games")
    return out_df
