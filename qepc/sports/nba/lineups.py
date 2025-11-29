"""
QEPC Module: lineups.py
Fetches live lineups from API.
Beginner: Calls data_source for easy use.
"""

from qepc.sports.nba.data_source import load_live_lineups

def get_lineup(team_name, game_id=None):
    """
    Get live lineup for team or game.
    - team_name: e.g., "Los Angeles Lakers"
    - game_id: Optional for specific game (get from schedule).
    """
    df = load_live_lineups(team_name=team_name, game_id=game_id)
    if df.empty:
        print("No lineup found.")
        return []
    starters = df[df['POSITION'] != 'Bench'] if 'POSITION' in df else df  # Filter starters
    return starters['PLAYER'].tolist()  # List of names