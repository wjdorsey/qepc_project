"""
QEPC Module: lambda_engine.py
Real lambda with live adjustments.
"""

import numpy as np
import pandas as pd
from qepc.core.model_config import *  # Settings

CITY_DISTANCES = {"Denver Nuggets": {"Los Angeles Lakers": 831}}  # Real distances; add more

def _get_team_hca(team: str) -> float:
    return BASE_HCA * TEAM_HCA_BOOST.get(team, 1.0)

def _calculate_rest_factor(rest_home, rest_away):
    diff = rest_home - rest_away
    advantage = np.clip(diff * REST_ADVANTAGE_PER_DAY, -MAX_REST_ADVANTAGE, MAX_REST_ADVANTAGE)
    home_adjust = 1.0 + (advantage / LEAGUE_AVG_POINTS)
    away_adjust = 1.0 - (advantage / LEAGUE_AVG_POINTS)
    return home_adjust, away_adjust

def _get_travel_penalty(home_team, away_team):
    distance = CITY_DISTANCES.get(home_team, {}).get(away_team, 0)
    penalty = distance / 1000 * TRAVEL_PENALTY_PER_1000MI
    return max(1.0 - penalty, 0.9)

def compute_lambda(schedule_df: pd.DataFrame, team_stats_df: pd.DataFrame, include_situational: bool = True) -> pd.DataFrame:
    df = schedule_df.copy()
    df["lambda_home"] = 0.0
    df["lambda_away"] = 0.0
    df["vol_home"] = 0.0
    df["vol_away"] = 0.0

    for index, row in df.iterrows():
        home = team_stats_df[team_stats_df['Team'] == row['Home Team']]
        away = team_stats_df[team_stats_df['Team'] == row['Away Team']]
        
        if home.empty or away.empty:
            continue
        
        home = home.iloc[0]
        away = away.iloc[0]
        
        game_pace = (home['Pace'] + away['Pace']) / 2
        hca = _get_team_hca(row['Home Team'])
        home_situational = 1.0
        away_situational = 1.0
        
        if include_situational:
            rest_home = row.get("home_rest_days", 3.0)
            rest_away = row.get("away_rest_days", 3.0)
            h_rest, a_rest = _calculate_rest_factor(rest_home, rest_away)
            home_situational *= h_rest
            away_situational *= a_rest
            
            home_b2b = row.get("home_b2b", False)
            away_b2b = row.get("away_b2b", False)
            if home_b2b:
                home_situational *= B2B_PENALTY
            if away_b2b:
                away_situational *= B2B_PENALTY
            
            travel_factor = _get_travel_penalty(row['Home Team'], row['Away Team'])
            away_situational *= travel_factor
        
        lambda_home = LEAGUE_AVG_POINTS * (home['ORtg'] / 100) * (away['DRtg'] / 100) * game_pace * hca * home_situational
        lambda_away = LEAGUE_AVG_POINTS * (away['ORtg'] / 100) * (home['DRtg'] / 100) * game_pace * away_situational
        
        noise_home = np.random.normal(1.0, QUANTUM_NOISE_STD)
        noise_away = np.random.normal(1.0, QUANTUM_NOISE_STD)
        df.at[index, "lambda_home"] = lambda_home * noise_home
        df.at[index, "lambda_away"] = lambda_away * noise_away
        df.at[index, "vol_home"] = home.get('Volatility', 10.0)
        df.at[index, "vol_away"] = away.get('Volatility', 10.0)
    
    print(f"Computed real lambdas for {len(df)} games.")
    return df