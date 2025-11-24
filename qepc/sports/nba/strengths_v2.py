"""
QEPC Module: strengths_v2.py
Calculates robust, time-decayed team strength ratings (ORtg/DRtg/Pace/Volatility).
Now supports Silent Mode for backtesting.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
from qepc.sports.nba.player_data import load_raw_player_data
from qepc.sports.nba.opponent_data import load_and_process_opponent_data 
from qepc.utils.data_cleaning import standardize_team_name 

# --- Configuration ---
FTA_MULTIPLIER = 0.44 
MIN_GAMES_PLAYED = 5 
DAYS_HISTORY_WINDOW = 730 
DECAY_LAMBDA = 0.002 
RATING_FLOOR = 92.0
RATING_CEILING = 122.0

def calculate_advanced_strengths(cutoff_date: Optional[str] = None, verbose: bool = True) -> pd.DataFrame:
    if verbose:
        print(f"[QEPC Strength V2] Starting Advanced Calculation (Cutoff: {cutoff_date if cutoff_date else 'Now'})...") 

    try:
        # Load data WITHOUT printing success messages (we rely on verbose flag in loader if needed)
        # Ideally, we would silence player_data too, but let's just silence this function first.
        raw_df = load_raw_player_data(file_name="PlayerStatistics.csv")
    except Exception as e:
        if verbose: print(f"[QEPC Strength V2] ERROR loading player data: {e}")
        return pd.DataFrame()
    
    if raw_df.empty:
        return pd.DataFrame()

    raw_df['Team'] = raw_df['Team'].apply(standardize_team_name)
    raw_df['gameDate'] = pd.to_datetime(raw_df['gameDate'], utc=True, errors='coerce')

    # --- TIME TRAVEL LOGIC ---
    if cutoff_date:
        target_dt = pd.to_datetime(cutoff_date, utc=True)
        raw_df = raw_df[raw_df['gameDate'] < target_dt].copy()
        
        if raw_df.empty:
            if verbose: print(f"[QEPC Strength V2] Error: No data available before {cutoff_date}.")
            return pd.DataFrame()
            
        current_time = target_dt
    else:
        current_time = pd.Timestamp.now(tz='UTC')

    # ... (Standard Logic - Remains the same) ...
    
    window_start = current_time - timedelta(days=DAYS_HISTORY_WINDOW)
    raw_df = raw_df[raw_df['gameDate'] >= window_start].copy()

    # 3. Aggregate Players -> Team-Game Level
    game_stats = raw_df.groupby(['Team', 'gameId', 'gameDate']).agg(
        G_Points=('PTS', 'sum'),
        G_FGA=('FGA', 'sum'),
        G_OR=('REB', 'sum'), 
        G_TO=('TOV', 'sum'),
        G_FTA=('FTA', 'sum'),
    ).reset_index()

    # 4. Calculate Per-Game Stats
    game_stats['G_Poss'] = (
        game_stats['G_FGA'] - game_stats['G_OR'] + game_stats['G_TO'] + (FTA_MULTIPLIER * game_stats['G_FTA'])
    )
    game_stats['G_Poss'] = game_stats['G_Poss'].replace(0, np.nan)
    
    # 5. Apply Time Decay
    game_stats['DaysAgo'] = (current_time - game_stats['gameDate']).dt.days
    game_stats['Weight'] = np.exp(-DECAY_LAMBDA * game_stats['DaysAgo'])

    # 6. Calculate Weighted Averages
    game_stats['Wt_Points'] = game_stats['G_Points'] * game_stats['Weight']
    game_stats['Wt_Poss'] = game_stats['G_Poss'] * game_stats['Weight']

    team_stats = game_stats.groupby('Team').agg(
        Sum_Wt_Points=('Wt_Points', 'sum'),
        Sum_Wt_Poss=('Wt_Poss', 'sum'),
        GamesPlayed=('gameId', 'count'),
        Volatility=('G_Points', 'std') 
    ).reset_index()

    # Reduce min games for backtesting early season
    min_games = MIN_GAMES_PLAYED if not cutoff_date else 1 
    team_stats = team_stats[team_stats['GamesPlayed'] >= min_games].copy()
    
    avg_vol = team_stats['Volatility'].mean()
    team_stats['Volatility'] = team_stats['Volatility'].fillna(avg_vol)

    # 7. Final Ratings
    team_stats['ORtg'] = (team_stats['Sum_Wt_Points'] / team_stats['Sum_Wt_Poss']) * 100
    simple_pace = game_stats.groupby('Team')['G_Poss'].mean().reset_index()
    team_stats = pd.merge(team_stats, simple_pace, on='Team')
    team_stats.rename(columns={'G_Poss': 'Pace'}, inplace=True)
    
    df_ortg = team_stats[['Team', 'ORtg', 'Pace', 'Volatility']].copy()

    # --- PART 2: MERGE WITH DEFENSE ---
    df_drtg = load_and_process_opponent_data()

    if df_drtg.empty:
        LA_ORtg = df_ortg['ORtg'].mean()
        df_ortg['DRtg'] = LA_ORtg
        return df_ortg[['Team', 'ORtg', 'DRtg', 'Pace', 'Volatility']]

    final_strengths = pd.merge(df_ortg, df_drtg, on='Team', how='inner')
    
    final_strengths['ORtg'] = final_strengths['ORtg'].clip(lower=RATING_FLOOR, upper=RATING_CEILING)
    final_strengths['DRtg'] = final_strengths['DRtg'].clip(lower=RATING_FLOOR, upper=RATING_CEILING)
    
    if verbose:
        print(f"[QEPC Strength V2] Calculated Time-Travel Strengths for {len(final_strengths)} teams.")
    return final_strengths[['Team', 'ORtg', 'DRtg', 'Pace', 'Volatility']]