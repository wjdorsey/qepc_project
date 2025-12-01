"""
QEPC Module: simulator.py
Simulates with real vol from API.
"""

import numpy as np
import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats

DEFAULT_NUM_TRIALS = 20000
SCORE_CORRELATION = 0.35
OT_POINTS_AVG = 10.0

def run_qepc_simulation(df: pd.DataFrame, num_trials: int = DEFAULT_NUM_TRIALS) -> pd.DataFrame:
    if df.empty:
        print("No data.")
        return df
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season='2025-26')
        team_stats = stats.get_data_frames()[0]
        
        # Check if PTS_STD column exists
        if 'PTS_STD' not in team_stats.columns:
            print("⚠️  Warning: PTS_STD column not in API response, using defaults")
            df['vol_home'] = 10.0
            df['vol_away'] = 10.0
        else:
            # Merge home team volatility
            df = pd.merge(df, team_stats[['TEAM_NAME', 'PTS_STD']], 
                          left_on='Home Team', right_on='TEAM_NAME', how='left')
            df['vol_home'] = df['PTS_STD'].fillna(10.0)
            df = df.drop(['PTS_STD', 'TEAM_NAME'], axis=1)
            
            # Merge away team volatility
            df = pd.merge(df, team_stats[['TEAM_NAME', 'PTS_STD']], 
                          left_on='Away Team', right_on='TEAM_NAME', how='left')
            df['vol_away'] = df['PTS_STD'].fillna(10.0)
            df = df.drop(['PTS_STD', 'TEAM_NAME'], axis=1)
        
    except Exception as e:
        print(f"⚠️  Could not fetch live volatility stats: {type(e).__name__}: {e}")
        print("    Using default volatility values (10.0 for all teams)")
        df['vol_home'] = 10.0
        df['vol_away'] = 10.0
    
    # Continue with simulation...
        
        # Merge away team volatility
        df = pd.merge(df, team_stats[['TEAM_NAME', 'PTS_STD']], 
                      left_on='Away Team', right_on='TEAM_NAME', how='left')
        df['vol_away'] = df['PTS_STD'].fillna(10.0)
        df = df.drop(['PTS_STD', 'TEAM_NAME'], axis=1)
        
    except (ImportError, AttributeError, IndexError) as e:
        print(f"⚠️  Could not fetch live volatility stats: {type(e).__name__}: {e}")
        df['vol_home'] = 10.0
        df['vol_away'] = 10.0
    
    lambda_home = df["lambda_home"].values
    lambda_away = df["lambda_away"].values
    vol_home = df["vol_home"].values
    vol_away = df["vol_away"].values
    
    for i in range(len(df)):
        z = np.random.multivariate_normal([0.0, 0.0], [[1.0, SCORE_CORRELATION], [SCORE_CORRELATION, 1.0]], num_trials)
        home_scores = lambda_home[i] + vol_home[i] * z[:, 0]
        away_scores = lambda_away[i] + vol_away[i] * z[:, 1]
        home_scores = np.maximum(home_scores, 50.0)
        away_scores = np.maximum(away_scores, 50.0)
        
        ties = np.abs(home_scores - away_scores) < 0.5
        if np.any(ties):
            ot_extra = np.random.normal(OT_POINTS_AVG, 2, sum(ties))
            home_scores[ties] += ot_extra / 2
            away_scores[ties] += ot_extra / 2
        
        home_wins = np.sum(home_scores > away_scores) / num_trials
        df.at[i, "Home_Win_Prob"] = home_wins
        df.at[i, "Away_Win_Prob"] = 1 - home_wins
        df.at[i, "Expected_Score_Total"] = np.mean(home_scores + away_scores)
        df.at[i, "Expected_Spread"] = np.mean(home_scores - away_scores)
        df.at[i, "Sim_Home_Score"] = np.mean(home_scores)
        df.at[i, "Sim_Away_Score"] = np.mean(away_scores)
    
    return df