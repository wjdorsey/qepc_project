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
    
    Parameters
    ----------
    target_date : str
        Date to backtest in format 'YYYY-MM-DD'
    num_trials : int, default 5000
        Number of Monte Carlo simulation trials
    verbose : bool, default True
        Print progress messages
        
    Returns
    -------
    pd.DataFrame
        Backtest results with predictions vs actuals
    """
    # 1. Load the "Answer Key"
    results_path = paths.get_data_dir() / "raw" / ACTUAL_RESULTS_FILE
    if not results_path.exists():
        if verbose:
            print(f"❌ Cannot find results file at {results_path}")
        return pd.DataFrame()
        
    try:
        df_actuals = pd.read_csv(results_path)
    except Exception as e:
        if verbose:
            print(f"❌ Error reading {results_path}: {e}")
        return pd.DataFrame()
    
    # 2a. Validate required columns exist
    required_cols = ['gameDate', 'home', 'teamName', 'opponentTeamName', 'teamScore', 'opponentScore']
    missing_cols = [col for col in required_cols if col not in df_actuals.columns]
    if missing_cols:
        if verbose:
            print(f"❌ Missing required columns in game data: {missing_cols}")
            print(f"   Available columns: {list(df_actuals.columns)}")
        return pd.DataFrame()
    
    # 2b. Parse dates - handle ISO 8601 format with timezone
    if verbose:
        print(f"Parsing dates (ISO 8601 format with timezone)...")
    
    try:
        # Parse with utc=True to handle timezone info properly
        df_actuals['gameDate'] = pd.to_datetime(df_actuals['gameDate'], utc=True, errors='coerce')
        
        # Remove timezone for consistent datetime operations
        df_actuals['gameDate'] = df_actuals['gameDate'].dt.tz_localize(None)
        
        valid_dates = df_actuals['gameDate'].notna()
        parsed_count = valid_dates.sum()
        
        if verbose:
            print(f"✅ Successfully parsed {parsed_count} dates")
        
        if parsed_count == 0:
            if verbose:
                print(f"❌ Could not parse any dates from gameDate column")
                print(f"   Sample values: {df_actuals['gameDate'].head(3).tolist()}")
            return pd.DataFrame()
        
        # Keep only valid dates
        invalid_count = (~valid_dates).sum()
        if invalid_count > 0 and verbose:
            print(f"⚠️  Dropped {invalid_count} rows with invalid dates")
        
        df_actuals = df_actuals[valid_dates].copy()
        
    except Exception as e:
        if verbose:
            print(f"❌ Error parsing dates: {e}")
            print(f"   Sample gameDate values: {df_actuals['gameDate'].head(3).tolist()}")
        return pd.DataFrame()
    
    if df_actuals.empty:
        if verbose:
            print(f"❌ No valid dates found after parsing")
        return pd.DataFrame()
    
    # 2c. Create date string for filtering
    df_actuals['date_str'] = df_actuals['gameDate'].dt.strftime('%Y-%m-%d')
    daily_games = df_actuals[df_actuals['date_str'] == target_date].copy()
    
    if daily_games.empty:
        if verbose:
            print(f"ℹ️  No games found for {target_date}")
            if not df_actuals.empty:
                print(f"   Data range: {df_actuals['gameDate'].min().date()} to {df_actuals['gameDate'].max().date()}")
        return pd.DataFrame()
    
    # 3. Construct Schedule
    # Filter for home games (home == 1)
    daily_schedule = daily_games[daily_games['home'] == 1].copy()
    if daily_schedule.empty:
        if verbose:
            print(f"ℹ️  No home games found for {target_date}")
        return pd.DataFrame()
    
    daily_schedule['Home Team'] = daily_schedule['teamName'].apply(standardize_team_name)
    daily_schedule['Away Team'] = daily_schedule['opponentTeamName'].apply(standardize_team_name)
    
    sim_schedule = daily_schedule[['Home Team', 'Away Team']].copy()
    sim_schedule['Actual_Home_Score'] = daily_schedule['teamScore'].values
    sim_schedule['Actual_Away_Score'] = daily_schedule['opponentScore'].values
    
    if verbose:
        print(f"Found {len(sim_schedule)} games for {target_date}")
    
    # 4. Time Travel: Calculate Strengths (as if we're on that date)
    try:
        strengths = calculate_advanced_strengths(cutoff_date=target_date, verbose=False)
    except Exception as e:
        if verbose:
            print(f"❌ Error calculating strengths for {target_date}: {e}")
        return pd.DataFrame()
    
    if strengths is None or strengths.empty:
        if verbose:
            print(f"⚠️  No strength data available for {target_date}")
        return pd.DataFrame()

    # 5. Run Prediction 
    try:
        schedule_with_lambda = compute_lambda(sim_schedule, strengths)
        predictions = run_qepc_simulation(schedule_with_lambda, num_trials=num_trials)
    except Exception as e:
        if verbose:
            print(f"❌ Error running simulation for {target_date}: {e}")
        return pd.DataFrame()
    
    if predictions is None or predictions.empty:
        if verbose:
            print(f"❌ Simulation produced no results for {target_date}")
        return pd.DataFrame()

    # 6. Score Results
    predictions['Actual_Spread'] = predictions['Actual_Home_Score'] - predictions['Actual_Away_Score']
    
    # Handle Expected_Spread safely (might not exist)
    if 'Expected_Spread' in predictions.columns:
        predictions['Spread_Error'] = (predictions['Expected_Spread'] - predictions['Actual_Spread']).abs()
    else:
        predictions['Spread_Error'] = 0.0
    
    predictions['Predicted_Winner_Home'] = predictions['Home_Win_Prob'] > 0.5
    predictions['Actual_Winner_Home'] = predictions['Actual_Home_Score'] > predictions['Actual_Away_Score']
    predictions['Correct_Pick'] = predictions['Predicted_Winner_Home'] == predictions['Actual_Winner_Home']
    predictions['Date'] = target_date
    
    if verbose:
        acc = predictions['Correct_Pick'].mean() * 100
        mae = predictions['Spread_Error'].mean()
        print(f"✅ {target_date}: {len(predictions)} Games | Accuracy: {acc:.1f}% | MAE: {mae:.2f}")

    # 7. Select and return relevant columns
    cols = [
        'Date', 'Home Team', 'Away Team', 
        'Home_Win_Prob', 'Away_Win_Prob', 
        'Expected_Spread', 'Actual_Spread', 'Spread_Error', 
        'Correct_Pick', 
        'Sim_Home_Score', 'Actual_Home_Score', 
        'Sim_Away_Score', 'Actual_Away_Score'
    ]
    
    # Only return columns that exist
    available_cols = [col for col in cols if col in predictions.columns]
    return predictions[available_cols]

def run_season_backtest(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Iterates through a date range silently.
    Returns a cumulative performance report.
    
    Parameters
    ----------
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns
    -------
    pd.DataFrame
        Combined results from all dates in range
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
        try:
            daily_df = run_daily_backtest(date_str, num_trials=1000, verbose=False)
            
            if daily_df is not None and not daily_df.empty:
                all_results.append(daily_df)
        except Exception as e:
            # Continue to next date even if one fails
            continue
            
    if not all_results:
        print("\n❌ No games found in this date range.")
        return pd.DataFrame()
        
    full_season_df = pd.concat(all_results, ignore_index=True)
    
    # --- FINAL REPORT ---
    total_games = len(full_season_df)
    total_accuracy = full_season_df['Correct_Pick'].mean() * 100
    avg_spread_error = full_season_df['Spread_Error'].mean()
    
    # Handle Expected_Score_Total safely
    if 'Expected_Score_Total' in full_season_df.columns:
        actual_totals = full_season_df['Actual_Home_Score'] + full_season_df['Actual_Away_Score']
        avg_total_error = (full_season_df['Expected_Score_Total'] - actual_totals).abs().mean()
    else:
        avg_total_error = 0.0
    
    print("\n" + "="*50)
    print(f"🏆 CUMULATIVE BACKTEST COMPLETE")
    print("="*50)
    print(f"📅 Date Range: {start_date} to {end_date}")
    print(f"🏀 Games Simulated: {total_games}")
    print(f"✅ Overall Accuracy: {total_accuracy:.2f}%")
    print(f"🎯 Avg Spread Error: {avg_spread_error:.2f} points")
    if avg_total_error > 0:
        print(f"📊 Avg Total Error: {avg_total_error:.2f} points")
    print("="*50)
    
    return full_season_df