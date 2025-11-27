"""
QEPC Module: simulator.py - IMPROVED
=====================================

Key improvements over original:
1. Score correlation (pace affects both teams together)
2. Option to use Normal distribution instead of Poisson
3. Better handling of ties (rare in NBA due to OT)
4. Vectorized operations for speed

"""
import pandas as pd
import numpy as np
from typing import Optional

DEFAULT_NUM_TRIALS = 20000

# Correlation between home/away scores (pace effect)
# Positive correlation: fast games = both teams score more
SCORE_CORRELATION = 0.35


def run_qepc_simulation(
    df: pd.DataFrame, 
    num_trials: int = DEFAULT_NUM_TRIALS,
    use_normal: bool = False,
    correlated_scores: bool = True
) -> pd.DataFrame:
    """
    Run QEPC Monte Carlo simulation with optional improvements.
    
    Parameters
    ----------
    df : DataFrame
        Schedule with lambda_home, lambda_away, vol_home, vol_away columns
    num_trials : int
        Number of Monte Carlo simulations per game
    use_normal : bool
        If True, use Normal distribution instead of Poisson.
        Normal can be more accurate for basketball scores.
    correlated_scores : bool
        If True, correlate home/away scores (pace effect).
        Fast-paced games boost both teams' scores.
    
    Returns
    -------
    DataFrame with win probabilities, expected scores, spreads
    """
    if df.empty or 'lambda_home' not in df.columns or 'lambda_away' not in df.columns:
        print("[QEPC Simulator] ERROR: Missing required columns.")
        return df

    # Prepare output columns
    df = df.copy()
    df['Home_Win_Prob'] = 0.0
    df['Away_Win_Prob'] = 0.0
    df['Tie_Prob'] = 0.0
    df['Expected_Score_Total'] = 0.0
    df['Expected_Spread'] = 0.0
    df['Sim_Home_Score'] = 0.0
    df['Sim_Away_Score'] = 0.0
    
    mode = "Normal" if use_normal else "Poisson"
    corr_str = "Correlated" if correlated_scores else "Independent"
    print(f"[QEPC Simulator] Running {num_trials} trials ({mode}, {corr_str})...")

    for index, row in df.iterrows():
        lambda_home = row['lambda_home']
        lambda_away = row['lambda_away']
        
        # Get volatility (default to sqrt(lambda) for Poisson-like variance)
        vol_home = row.get('vol_home', np.sqrt(lambda_home))
        vol_away = row.get('vol_away', np.sqrt(lambda_away))
        
        # Ensure minimum volatility
        vol_home = max(vol_home, 8.0) if vol_home > 0 else np.sqrt(lambda_home)
        vol_away = max(vol_away, 8.0) if vol_away > 0 else np.sqrt(lambda_away)

        # =================================================================
        # SCORE GENERATION
        # =================================================================
        
        if correlated_scores:
            # Generate correlated random factors
            # This creates a "pace factor" that affects both teams
            cov_matrix = [
                [1.0, SCORE_CORRELATION],
                [SCORE_CORRELATION, 1.0]
            ]
            correlated_z = np.random.multivariate_normal(
                mean=[0, 0], 
                cov=cov_matrix, 
                size=num_trials
            )
            z_home = correlated_z[:, 0]
            z_away = correlated_z[:, 1]
        else:
            z_home = np.random.standard_normal(num_trials)
            z_away = np.random.standard_normal(num_trials)
        
        if use_normal:
            # Normal distribution: mean = lambda, std = volatility
            home_scores = lambda_home + vol_home * z_home
            away_scores = lambda_away + vol_away * z_away
            
            # Scores can't be negative (floor at ~50 for realistic NBA)
            home_scores = np.maximum(home_scores, 50)
            away_scores = np.maximum(away_scores, 50)
            
        else:
            # Poisson with volatility-adjusted lambda (original QEPC approach)
            # First, adjust lambda for each trial based on volatility
            if vol_home > 0:
                # Scale z to create lambda variation
                lambda_scale = 0.1  # 10% lambda variation per 1 std
                home_lambdas = lambda_home * (1 + lambda_scale * z_home)
                home_lambdas = np.maximum(home_lambdas, 50)  # Floor
            else:
                home_lambdas = np.full(num_trials, lambda_home)
            
            if vol_away > 0:
                away_lambdas = lambda_away * (1 + lambda_scale * z_away)
                away_lambdas = np.maximum(away_lambdas, 50)
            else:
                away_lambdas = np.full(num_trials, lambda_away)
            
            # Generate Poisson scores
            home_scores = np.random.poisson(home_lambdas)
            away_scores = np.random.poisson(away_lambdas)

        # =================================================================
        # CALCULATE RESULTS
        # =================================================================
        
        # Round to integers for final scores
        home_scores_int = np.round(home_scores).astype(int)
        away_scores_int = np.round(away_scores).astype(int)
        
        # Count outcomes
        home_wins = np.sum(home_scores_int > away_scores_int)
        away_wins = np.sum(away_scores_int > home_scores_int)
        ties = np.sum(home_scores_int == away_scores_int)
        
        # In NBA, ties go to OT - simulate as 50/50 for tied games
        # (In reality, home team has slight OT advantage, could adjust)
        ot_home_wins = ties // 2
        ot_away_wins = ties - ot_home_wins
        
        home_wins += ot_home_wins
        away_wins += ot_away_wins
        
        # Store results
        df.loc[index, 'Home_Win_Prob'] = home_wins / num_trials
        df.loc[index, 'Away_Win_Prob'] = away_wins / num_trials
        df.loc[index, 'Tie_Prob'] = 0.0  # NBA has no ties (OT resolves)
        df.loc[index, 'Expected_Score_Total'] = np.mean(home_scores + away_scores)
        df.loc[index, 'Expected_Spread'] = np.mean(home_scores - away_scores)
        df.loc[index, 'Sim_Home_Score'] = np.mean(home_scores)
        df.loc[index, 'Sim_Away_Score'] = np.mean(away_scores)

    print("[QEPC Simulator] Simulation complete.")
    return df


def run_qepc_simulation_fast(
    df: pd.DataFrame, 
    num_trials: int = DEFAULT_NUM_TRIALS
) -> pd.DataFrame:
    """
    Faster vectorized version for large backtests.
    Uses Normal distribution with correlation.
    """
    if df.empty or 'lambda_home' not in df.columns:
        return df
    
    df = df.copy()
    n_games = len(df)
    
    # Extract arrays
    lambda_home = df['lambda_home'].values
    lambda_away = df['lambda_away'].values
    vol_home = df.get('vol_home', pd.Series(np.sqrt(lambda_home))).values
    vol_away = df.get('vol_away', pd.Series(np.sqrt(lambda_away))).values
    
    # Set minimum volatility
    vol_home = np.maximum(vol_home, 8.0)
    vol_away = np.maximum(vol_away, 8.0)
    
    # Generate all random numbers at once (much faster)
    # Shape: (n_games, num_trials, 2) for correlated home/away
    cov = [[1, SCORE_CORRELATION], [SCORE_CORRELATION, 1]]
    
    results = {
        'Home_Win_Prob': np.zeros(n_games),
        'Away_Win_Prob': np.zeros(n_games),
        'Expected_Spread': np.zeros(n_games),
        'Expected_Score_Total': np.zeros(n_games),
        'Sim_Home_Score': np.zeros(n_games),
        'Sim_Away_Score': np.zeros(n_games),
    }
    
    for i in range(n_games):
        z = np.random.multivariate_normal([0, 0], cov, num_trials)
        
        home_scores = lambda_home[i] + vol_home[i] * z[:, 0]
        away_scores = lambda_away[i] + vol_away[i] * z[:, 1]
        
        home_scores = np.maximum(home_scores, 50)
        away_scores = np.maximum(away_scores, 50)
        
        home_wins = np.sum(home_scores > away_scores)
        # Handle ties (split them)
        ties = np.sum(np.abs(home_scores - away_scores) < 0.5)
        home_wins += ties // 2
        
        results['Home_Win_Prob'][i] = home_wins / num_trials
        results['Away_Win_Prob'][i] = 1 - results['Home_Win_Prob'][i]
        results['Expected_Spread'][i] = np.mean(home_scores - away_scores)
        results['Expected_Score_Total'][i] = np.mean(home_scores + away_scores)
        results['Sim_Home_Score'][i] = np.mean(home_scores)
        results['Sim_Away_Score'][i] = np.mean(away_scores)
    
    for col, vals in results.items():
        df[col] = vals
    
    df['Tie_Prob'] = 0.0
    
    return df
