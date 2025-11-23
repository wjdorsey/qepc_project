"""
QEPC Module: simulator.py
Runs the Quantum Entangled Poisson Cascade (QEPC) Monte Carlo simulation.
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
    
    print(f"[QEPC Simulator] Running {num_trials} trials for {len(df)} games...")

    # Iterate through each game and run the simulation
    for index, row in df.iterrows():
        lambda_home = row['lambda_home']
        lambda_away = row['lambda_away']

        # 1. Simulate the scores using numpy's Poisson sampling
        home_scores = np.random.poisson(lambda_home, num_trials)
        away_scores = np.random.poisson(lambda_away, num_trials)

        # 2. Determine winners in each trial
        home_wins = np.sum(home_scores > away_scores)
        away_wins = np.sum(away_scores > home_scores)
        ties = np.sum(home_scores == away_scores)

        # 3. Calculate probabilities
        total_trials = num_trials
        expected_total = np.mean(home_scores + away_scores)
        
        # 4. Store results back into the DataFrame
        df.loc[index, 'Home_Win_Prob'] = home_wins / total_trials
        df.loc[index, 'Away_Win_Prob'] = away_wins / total_trials
        df.loc[index, 'Tie_Prob'] = ties / total_trials
        df.loc[index, 'Expected_Score_Total'] = expected_total
        df.loc[index, 'Expected_Spread'] = np.mean(home_scores - away_scores)
        df.loc[index, 'Sim_Home_Score'] = np.mean(home_scores)
        df.loc[index, 'Sim_Away_Score'] = np.mean(away_scores)

    print("[QEPC Simulator] Simulation complete. Prediction metrics added.")
    return df