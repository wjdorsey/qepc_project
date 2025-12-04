"""
Live NBA ScoreBoard Fetcher
Gets today's games with real-time data
"""

from nba_api.live.nba.endpoints import scoreboard
import json
from pathlib import Path
from datetime import datetime

def fetch_todays_scoreboard():
    """
    Fetch today's NBA scoreboard
    
    Returns:
        dict: Today's games with scores, leaders, status
    """
    try:
        # Get scoreboard
        board = scoreboard.ScoreBoard()
        
        # Extract data
        data = {
            'fetch_time': datetime.now().isoformat(),
            'game_date': board.score_board_date,
            'games': board.games.get_dict()
        }
        
        # Save to file
        output_dir = Path(__file__).parent.parent / 'data' / 'live'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'scoreboard.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Fetched {len(data['games'])} games")
        print(f"💾 Saved to: {output_file}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching scoreboard: {e}")
        return None


def get_todays_player_leaders():
    """
    Extract top performers from today's games
    """
    data = fetch_todays_scoreboard()
    
    if not data:
        return []
    
    leaders = []
    
    for game in data['games']:
        if game['gameStatus'] == 3:  # Final
            # Home leader
            home_leader = game['gameLeaders']['homeLeaders']
            leaders.append({
                'player': home_leader['name'],
                'team': game['homeTeam']['teamName'],
                'points': home_leader['points'],
                'rebounds': home_leader['rebounds'],
                'assists': home_leader['assists']
            })
            
            # Away leader
            away_leader = game['gameLeaders']['awayLeaders']
            leaders.append({
                'player': away_leader['name'],
                'team': game['awayTeam']['teamName'],
                'points': away_leader['points'],
                'rebounds': away_leader['rebounds'],
                'assists': away_leader['assists']
            })
    
    return leaders


if __name__ == "__main__":
    print("🏀 Fetching today's NBA scoreboard...")
    data = fetch_todays_scoreboard()
    
    if data:
        print(f"\n📅 Game Date: {data['game_date']}")
        print(f"🎮 Games: {len(data['games'])}")
        
        # Show leaders
        leaders = get_todays_player_leaders()
        if leaders:
            print(f"\n🌟 Top Performers:")
            for leader in leaders[:5]:
                print(f"   {leader['player']} ({leader['team']}): "
                      f"{leader['points']} PTS, {leader['rebounds']} REB, {leader['assists']} AST")
