"""
QEPC Module: backtest_engine.py
Orchestrates time-travel backtesting. 
Optimized for long-range simulations (Quiet Mode).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional

from qepc.autoload import paths
from qepc.utils.data_cleaning import standardize_team_name
from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths
from qepc.core.lambda_engine import compute_lambda
from qepc.core.simulator import run_qepc_simulation

# --- Configuration ---
ACTUAL_RESULTS_FILE = "TeamStatistics.csv" 

def run_daily_backtest(target_date: str, num_trials: int = 5000, verbose: bool = True) -> pd.DataFrame:
    """
    Runs a full QEPC prediction cycle for a specific past date.
    """
    # 1. Load the "Answer Key"
    results_path = paths.get_data_dir() / "raw" / ACTUAL_RESULTS_FILE
    if not results_path.exists():
        return pd.DataFrame()
        
    df_actuals = pd.read_csv(results_path)
    df_actuals['date_str'] = df_actuals['gameDate'].astype(str).str.slice(0, 10)
    daily_games = df_actuals[df_actuals['date_str'] == target_date].copy()
    
    if daily_games.empty:
        return pd.DataFrame()
    
    # 2. Construct Schedule
    daily_schedule = daily_games[daily_games['home'] == 1].copy()
    daily_schedule['Home Team'] = daily_schedule['teamName'].apply(standardize_team_name)
    daily_schedule['Away Team'] = daily_schedule['opponentTeamName'].apply(standardize_team_name)
    
    sim_schedule = daily_schedule[['Home Team', 'Away Team']].copy()
    sim_schedule['Actual_Home_Score'] = daily_schedule['teamScore']
    sim_schedule['Actual_Away_Score'] = daily_schedule['opponentScore']
    
    # 3. Time Travel: Calculate Strengths
    # Run silently
    strengths = calculate_advanced_strengths(cutoff_date=target_date, verbose=False)
    
    if strengths.empty:
        return pd.DataFrame()

    # 4. Run Prediction 
    schedule_with_lambda = compute_lambda(sim_schedule, strengths)
    predictions = run_qepc_simulation(schedule_with_lambda, num_trials=num_trials)
    
    # 5. Score Results
    predictions['Actual_Spread'] = predictions['Actual_Home_Score'] - predictions['Actual_Away_Score']
    predictions['Spread_Error'] = predictions['Expected_Spread'] - predictions['Actual_Spread']
    
    predictions['Predicted_Winner_Home'] = predictions['Home_Win_Prob'] > 0.5
    predictions['Actual_Winner_Home'] = predictions['Actual_Home_Score'] > predictions['Actual_Away_Score']
    predictions['Correct_Pick'] = predictions['Predicted_Winner_Home'] == predictions['Actual_Winner_Home']
    predictions['Date'] = target_date
    
    if verbose:
        acc = predictions['Correct_Pick'].mean()
        print(f"✅ {target_date}: {len(predictions)} Games | Accuracy: {acc:.0%}")

    cols = [
        'Date', 'Away Team', 'Home Team', 
        'Away_Win_Prob', 'Home_Win_Prob', 
        'Expected_Spread', 'Actual_Spread', 'Spread_Error', 
        'Correct_Pick', 
        'Sim_Home_Score', 'Actual_Home_Score', 
        'Sim_Away_Score', 'Actual_Away_Score'
    ]
    return predictions[cols]

def run_season_backtest(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Iterates through a date range silently.
    Returns a cumulative performance report.
    """
    print(f"🚀 STARTING LONG-RANGE BACKTEST ({start_date} to {end_date})")
    print("Processing... (This will update in place)")
    
    date_range = pd.date_range(start=start_date, end=end_date)
    all_results = []
    total_days = len(date_range)
    
    for i, dt in enumerate(date_range):
        date_str = dt.strftime("%Y-%m-%d")
        
        # Update progress bar
        print(f"⏳ Processing Day {i+1}/{total_days}: {date_str}", end="\r")
        
        # Run daily test SILENTLY (verbose=False)
        daily_df = run_daily_backtest(date_str, num_trials=1000, verbose=False)
        
        if not daily_df.empty:
            all_results.append(daily_df)
            
    if not all_results:
        print("\n❌ No games found in this date range.")
        return pd.DataFrame()
        
    full_season_df = pd.concat(all_results)
    
    # --- FINAL REPORT ---
    total_games = len(full_season_df)
    total_accuracy = full_season_df['Correct_Pick'].mean() * 100
    avg_spread_error = full_season_df['Spread_Error'].abs().mean()
    
    print("\n" + "="*40)
    print(f"🏆 CUMULATIVE BACKTEST COMPLETE")
    print("="*40)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"🏀 Games Simulated: {total_games}")
    print(f"✅ Overall Accuracy: {total_accuracy:.2f}%")
    print(f"🎯 Avg Spread Error: {avg_spread_error:.2f} points")
    print("="*40)
    
    return full_season_df