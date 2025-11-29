"""
QEPC NBA Data Source
Live from API.
"""

import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats
from datetime import datetime, timedelta

def load_team_stats(season="2025-26", local_file="data/raw/Team_Stats.csv"):
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df = stats.get_data_frames()[0]
        print("Live API data fetched.")
    except:
        df = pd.read_csv(local_file)
        print("Used local file.")
    
    # Fresh check
    # Add date col if needed
    if 'LAST_GAME_DATE' in df:  # Assume or add
        max_date = pd.to_datetime(df['LAST_GAME_DATE']).max()
        if (datetime.now() - max_date) > timedelta(days=1):
            print("Data old!")
    
    df.rename(columns={'TEAM_NAME': 'Team', 'OFF_RATING': 'ORtg', 'DEF_RATING': 'DRtg', 'PACE': 'Pace'}, inplace=True)
    return df[['Team', 'ORtg', 'DRtg', 'Pace']]

# ... (keep your existing code at top)

import requests  # For Balldontlie

BALLDONTLIE_API_KEY = "c5ae7df3-682e-450c-b47e-f7e91396379e"  # Your key

def load_live_injuries(local_file="data/Injury_Overrides.csv"):
    """
    Fetches live injuries from API.
    - Tries nba_api Rotowire first (live updates).
    - Backup: Balldontlie API.
    - Fallback: Local CSV.
    """
    try:
        # nba_api Rotowire (live)
        from nba_api.live.nba.endpoints import scoreboard
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        # Extract injuries (simplified; Rotowire has full)
        injuries = []  # Loop games for player status
        for game in games:
            for team in [game['homeTeam'], game['awayTeam']]:
                for player in team['players']:
                    if player.get('status') != 'ACTIVE':
                        injuries.append({'PlayerName': player['name'], 'Team': team['teamTricode'], 'Status': player.get('status', 'Unknown')})
        df = pd.DataFrame(injuries)
        if not df.empty:
            print(f"Fetched {len(df)} live injuries from Rotowire/nba_api.")
            return df
    except Exception as e:
        print(f"nba_api error: {e}. Trying Balldontlie.")
    
    try:
        # Balldontlie backup (live with key)
        url = "https://api.balldontlie.io/v1/injuries"
        headers = {"Authorization": BALLDONTLIE_API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()['data']
            df = pd.DataFrame(data)
            df = df[['player', 'team', 'status']]  # Adjust cols
            df.rename(columns={'player': 'PlayerName', 'team': 'Team', 'status': 'Status'}, inplace=True)
            print(f"Fetched {len(df)} live injuries from Balldontlie.")
            return df
    except Exception as e:
        print(f"Balldontlie error: {e}. Using local file.")
    
    # Fallback local
    df = pd.read_csv(local_file)
    print(f"Used local {len(df)} injuries.")
    return df


# ... (keep top code)

import requests  # For Balldontlie

BALLDONTLIE_API_KEY = "c5ae7df3-682e-450c-b47e-f7e91396379e"  # Your key

def load_live_lineups(team_name=None, game_id=None, local_file="data/raw/Lineups.csv"):
    """
    Fetches live player lineups.
    - team_name: For team roster (starters).
    - game_id: For specific game lineups (live).
    - Tries nba_api first.
    - Backup: Balldontlie (players only).
    - Fallback: Local CSV.
    """
    try:
        from nba_api.stats.static import teams
        from nba_api.stats.endpoints import commonteamroster, boxscoretraditionalv2
        
        if team_name:
            # Live team roster (starters)
            team = [t for t in teams.get_teams() if t['full_name'] == team_name]
            if not team:
                raise ValueError("Team not found.")
            team_id = team[0]['id']
            roster = commonteamroster.CommonTeamRoster(season='2025-26', team_id=team_id)
            df = roster.get_data_frames()[0]
            df = df[['PLAYER', 'POSITION']]  # Starters
            print(f"Fetched live roster for {team_name} from nba_api.")
            return df
        
        if game_id:
            # Live game lineups
            box = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
            df = box.get_data_frames()[1]  # Player stats with starters
            starters = df[df['START_POSITION'] != '']  # Filter starters
            print(f"Fetched live lineups for game {game_id} from nba_api.")
            return starters[['PLAYER_NAME', 'START_POSITION']]
    
    except Exception as e:
        print(f"nba_api error: {e}. Trying Balldontlie.")
    
    try:
        # Balldontlie backup (players only, no lineups—use for rosters)
        url = f"https://api.balldontlie.io/v1/players?team={team_name.lower() if team_name else ''}"
        headers = {"Authorization": BALLDONTLIE_API_KEY}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()['data']
            df = pd.DataFrame(data)
            df = df[['full_name', 'position']]  # Approx roster
            print(f"Fetched {len(df)} players from Balldontlie.")
            return df
    except Exception as e:
        print(f"Balldontlie error: {e}. Using local.")
    
    # Fallback local
    df = pd.read_csv(local_file)
    print(f"Used local {len(df)} lineups.")
    return df