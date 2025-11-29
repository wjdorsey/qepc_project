"""
QEPC Module: scripts.py
Helpers with real data.
"""

import pandas as pd
from qepc.core.lambda_engine import compute_lambda
from qepc.core.simulator import run_qepc_simulation

def run_quick_prediction(home_team, away_team, season='2025-26'):
    team_stats = pd.read_csv('data/raw/Team_Stats.csv')  # Real file
    schedule = pd.DataFrame({'Home Team': [home_team], 'Away Team': [away_team]})
    lambdas = compute_lambda(schedule, team_stats)
    sim = run_qepc_simulation(lambdas)
    return sim