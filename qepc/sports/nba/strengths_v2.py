"""
QEPC Module: strengths_v2.py
Calculates robust, time-decayed team strength ratings (ORtg/DRtg/Pace/Volatility).
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

def calculate_advanced_strengths() -> pd.DataFrame:
    print("[QEPC Strength V2] Starting Advanced Strength Calculation (with Volatility)...") 

    try:
        raw_df = load_raw_player_data(file_name="PlayerStatistics.csv")
    except Exception as e:
        print(f"[QEPC Strength V2] ERROR loading player data: {e}")
        return pd.DataFrame()
    
    if raw_df.empty:
        return pd.DataFrame()

    # 1. Pre-Processing
    raw_df['Team'] = raw_df['Team'].apply(standardize_team_name)
    raw_df['gameDate'] = pd.to_datetime(raw_df['gameDate'], utc=True, errors='coerce')

    # 2. Filter by Date Window
    cutoff_date = pd.Timestamp.now(tz='UTC') - timedelta(days=DAYS_HISTORY_WINDOW)
    raw_df = raw_df[raw_df['gameDate'] >= cutoff_date].copy()
    
    if raw_df.empty:
        return pd.DataFrame()

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
    current_time = pd.Timestamp.now(tz='UTC')
    game_stats['DaysAgo'] = (current_time - game_stats['gameDate']).dt.days
    game_stats['Weight'] = np.exp(-DECAY_LAMBDA * game_stats['DaysAgo'])

    # 6. Calculate Weighted Averages AND Volatility
    game_stats['Wt_Points'] = game_stats['G_Points'] * game_stats['Weight']
    game_stats['Wt_Poss'] = game_stats['G_Poss'] * game_stats['Weight']

    team_stats = game_stats.groupby('Team').agg(
        Sum_Wt_Points=('Wt_Points', 'sum'),
        Sum_Wt_Poss=('Wt_Poss', 'sum'),
        GamesPlayed=('gameId', 'count'),
        # CHAOS FACTOR: Calculate Standard Deviation of raw points scored per game
        Volatility=('G_Points', 'std') 
    ).reset_index()

    # Filter noise
    team_stats = team_stats[team_stats['GamesPlayed'] >= MIN_GAMES_PLAYED].copy()
    
    # Fill NaN volatility (for teams with 1 game) with league average
    avg_vol = team_stats['Volatility'].mean()
    team_stats['Volatility'] = team_stats['Volatility'].fillna(avg_vol)

    # 7. Final Ratings
    team_stats['ORtg'] = (team_stats['Sum_Wt_Points'] / team_stats['Sum_Wt_Poss']) * 100
    # Estimate Pace (Possessions / Weight Sum isn't direct, so we average the raw G_Poss)
    # Re-calculating simple average pace for stability
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
    
    # Clamping
    final_strengths['ORtg'] = final_strengths['ORtg'].clip(lower=RATING_FLOOR, upper=RATING_CEILING)
    final_strengths['DRtg'] = final_strengths['DRtg'].clip(lower=RATING_FLOOR, upper=RATING_CEILING)
    
    print(f"[QEPC Strength V2] Strengths + Volatility calculated for {len(final_strengths)} teams.")
    return final_strengths[['Team', 'ORtg', 'DRtg', 'Pace', 'Volatility']]
