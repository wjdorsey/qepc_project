"""
QEPC Module: sim.py
Live schedules from API.
"""

import pandas as pd
from datetime import date
from nba_api.stats.endpoints import leaguegamelog

def load_nba_schedule(season='2025-26'):
    try:
        log = leaguegamelog.LeagueGameLog(season=season)
        df = log.get_data_frames()[0]
        df['gameDate'] = pd.to_datetime(df['GAME_DATE'])
    except:
        df = pd.read_csv('data/Games.csv')
        df['gameDate'] = pd.to_datetime(df['Date'])
    
    return df

def get_today_games():
    schedule = load_nba_schedule()
    today = date.today()
    return schedule[schedule['gameDate'].dt.date == today]

# ... (keep top code)

from nba_api.stats.endpoints import leaguegamelog, scoreboard

def load_nba_schedule(season='2025-26'):
    try:
        # Live today's schedule with game IDs
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        data = []
        for game in games:
            data.append({'gameDate': game['gameTimeUTC'], 'gameId': game['gameId'], 'Home Team': game['homeTeam']['teamName'], 'Away Team': game['awayTeam']['teamName']})
        df = pd.DataFrame(data)
        df['gameDate'] = pd.to_datetime(df['gameDate'])
        print("Fetched live schedule from nba_api.")
    except:
        # Backup full season
        log = leaguegamelog.LeagueGameLog(season=season)
        df = log.get_data_frames()[0]
        df['gameDate'] = pd.to_datetime(df['GAME_DATE'])
    
    return df

def get_today_games(with_lineups=False):
    schedule = load_nba_schedule()
    today = date.today()
    today_games = schedule[schedule['gameDate'].dt.date == today]
    
    if with_lineups:
        for index, row in today_games.iterrows():
            home_lineup = get_lineup(row['Home Team'], row['gameId'])
            away_lineup = get_lineup(row['Away Team'], row['gameId'])
            today_games.at[index, 'Home Lineup'] = ', '.join(home_lineup)
            today_games.at[index, 'Away Lineup'] = ', '.join(away_lineup)
    
    return today_games