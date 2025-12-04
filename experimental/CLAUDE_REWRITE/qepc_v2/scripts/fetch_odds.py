"""
NBA Betting Odds Fetcher
Gets current betting lines for props comparison
"""

from nba_api.live.nba.endpoints import odds
import json
from pathlib import Path
from datetime import datetime

def fetch_todays_odds():
    """
    Fetch today's betting odds
    
    Returns:
        dict: Betting lines (spreads, totals, moneylines)
    """
    try:
        # Get odds
        odds_data = odds.Odds()
        
        # Extract data
        data = {
            'fetch_time': datetime.now().isoformat(),
            'games': odds_data.games.get_dict()
        }
        
        # Save to file
        output_dir = Path(__file__).parent.parent / 'data' / 'live'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / 'odds.json'
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✅ Fetched odds for {len(data['games'])} games")
        print(f"💾 Saved to: {output_file}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error fetching odds: {e}")
        return None


def extract_spreads_and_totals():
    """
    Extract clean betting lines for QEPC comparison
    """
    data = fetch_todays_odds()
    
    if not data:
        return []
    
    lines = []
    
    for game in data['games']:
        game_lines = {
            'game_id': game['gameId'],
            'home_team_id': game['homeTeamId'],
            'away_team_id': game['awayTeamId'],
            'spreads': [],
            'totals': [],
            'moneylines': []
        }
        
        # Extract from all books
        for market in game.get('markets', []):
            market_name = market['name']
            
            for book in market.get('books', []):
                book_name = book['name']
                
                if market_name == 'spread':
                    for outcome in book['outcomes']:
                        game_lines['spreads'].append({
                            'book': book_name,
                            'team': outcome['type'],
                            'line': outcome.get('spread'),
                            'odds': outcome['odds']
                        })
                
                elif market_name == 'total':
                    for outcome in book['outcomes']:
                        game_lines['totals'].append({
                            'book': book_name,
                            'type': outcome['type'],  # over/under
                            'line': outcome.get('spread'),
                            'odds': outcome['odds']
                        })
                
                elif market_name == '2way':
                    for outcome in book['outcomes']:
                        game_lines['moneylines'].append({
                            'book': book_name,
                            'team': outcome['type'],
                            'odds': outcome['odds']
                        })
        
        lines.append(game_lines)
    
    return lines


def get_consensus_lines():
    """
    Calculate consensus lines across all books
    """
    lines = extract_spreads_and_totals()
    
    consensus = []
    
    for game in lines:
        # Average spread
        home_spreads = [s['line'] for s in game['spreads'] 
                       if s['team'] == 'home' and s['line']]
        avg_spread = sum(home_spreads) / len(home_spreads) if home_spreads else None
        
        # Average total
        over_totals = [t['line'] for t in game['totals'] 
                      if t['type'] == 'over' and t['line']]
        avg_total = sum(over_totals) / len(over_totals) if over_totals else None
        
        consensus.append({
            'game_id': game['game_id'],
            'consensus_spread': avg_spread,
            'consensus_total': avg_total,
            'num_books': len(set(s['book'] for s in game['spreads']))
        })
    
    return consensus


if __name__ == "__main__":
    print("💰 Fetching today's betting odds...")
    data = fetch_todays_odds()
    
    if data:
        print(f"\n🎲 Games with odds: {len(data['games'])}")
        
        # Show consensus
        consensus = get_consensus_lines()
        print(f"\n📊 Consensus Lines:")
        for line in consensus[:3]:
            if line['consensus_spread']:
                print(f"   Game {line['game_id']}: "
                      f"Spread {line['consensus_spread']:.1f}, "
                      f"Total {line['consensus_total']:.1f}")
