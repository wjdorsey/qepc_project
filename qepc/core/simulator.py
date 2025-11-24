"""
QEPC Module: simulator.py
Runs the Quantum Entangled Poisson Cascade (QEPC) Monte Carlo simulation.
Includes 'Chaos Factor' (Volatility) logic to model team consistency.
"""
import pandas as pd
import numpy as np
from typing import Optional, Union

DEFAULT_NUM_TRIALS = 20000

def run_qepc_simulation(
    df: pd.DataFrame, 
    num_trials: int = DEFAULT_NUM_TRIALS
) -> pd.DataFrame:
    """
    Runs the QEPC Monte Carlo simulation for all games in the schedule dataframe.
    Uses Team Volatility to adjust the width of the probability distribution.
    """
    if df.empty or 'lambda_home' not in df.columns or 'lambda_away' not in df.columns:
        print("[QEPC Simulator] ERROR: Input DataFrame is empty or missing lambda values.")
        return df

    # Prepare columns for results
    df['Home_Win_Prob'] = 0.0
    df['Away_Win_Prob'] = 0.0
    df['Tie_Prob'] = 0.0
    df['Expected_Score_Total'] = 0.0
    df['Expected_Spread'] = 0.0
    df['Sim_Home_Score'] = 0.0
    df['Sim_Away_Score'] = 0.0
    
    print(f"[QEPC Simulator] Running {num_trials} trials for {len(df)} games (Chaos Engine Active)...")

    # Iterate through each game
    for index, row in df.iterrows():
        lambda_home = row['lambda_home']
        lambda_away = row['lambda_away']
        
        # CHAOS FACTOR: Retrieve Volatility (Standard Deviation)
        # If missing (legacy data), default to 0.0 (Standard Poisson)
        vol_home = row.get('vol_home', 0.0)
        vol_away = row.get('vol_away', 0.0)

        # --- STEP 1: THE SUPERPOSITION (Determine "Game Form") ---
        # We adjust the lambda for this specific trial based on team volatility.
        # A high volatility team has a wider range of potential "Game Forms".
        
        if vol_home > 0:
            # Sample a specific lambda for each trial from a Normal Distribution
            # We use size=num_trials to generate a unique "Form" for every simulated universe
            home_lambdas = np.random.normal(lambda_home, vol_home * 0.5, num_trials)
            # Lambdas cannot be negative, clamp at 0.1
            home_lambdas = np.maximum(home_lambdas, 0.1)
        else:
            # Standard consistent performance
            home_lambdas = lambda_home

        if vol_away > 0:
            away_lambdas = np.random.normal(lambda_away, vol_away * 0.5, num_trials)
            away_lambdas = np.maximum(away_lambdas, 0.1)
        else:
            away_lambdas = lambda_away

        # --- STEP 2: THE COLLAPSE (Generate Scores) ---
        # We sample the final score from the Poisson distribution using the specific form lambda
        home_scores = np.random.poisson(home_lambdas, num_trials) if vol_home > 0 else np.random.poisson(lambda_home, num_trials)
        away_scores = np.random.poisson(away_lambdas, num_trials) if vol_away > 0 else np.random.poisson(lambda_away, num_trials)

        # --- STEP 3: DETERMINE WINNERS ---
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)
        ties = np.sum(home_scores == away_scores)

        # Calculate probabilities
        total_trials = num_trials
        
        # Store results
        df.loc[index, 'Home_Win_Prob'] = home_wins / total_trials
        df.loc[index, 'Away_Win_Prob'] = away_wins / total_trials
        df.loc[index, 'Tie_Prob'] = ties / total_trials
        df.loc[index, 'Expected_Score_Total'] = np.mean(home_scores + away_scores)
        df.loc[index, 'Expected_Spread'] = np.mean(home_scores - away_scores)
        df.loc[index, 'Sim_Home_Score'] = np.mean(home_scores)
        df.loc[index, 'Sim_Away_Score'] = np.mean(away_scores)

    print("[QEPC Simulator] Simulation complete.")
    return df