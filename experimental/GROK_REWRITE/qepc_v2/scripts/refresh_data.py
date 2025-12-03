"""
QEPC Data Refresh
=================
Fetches fresh data from NBA APIs and saves to clean CSV structure.

APIs Used:
- NBA Stats API (via nba_api package) - Team ratings, game logs
- NBA Live API (direct CDN) - Scoreboard, odds

Run this script on your LOCAL machine to refresh data before predictions.

Usage:
    python refresh_data.py           # Refresh all data
    python refresh_data.py --today   # Just today's games + odds
    python refresh_data.py --full    # Full historical refresh
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import time
import json
import requests
from typing import Optional, Dict, List

# ============================================================================
# CONFIGURATION
# ============================================================================

# Auto-detect project root
def find_project_root() -> Path:
    """Find project root by looking for data folder."""
    current = Path.cwd()
    for p in [current] + list(current.parents)[:5]:
        if (p / "data").exists():
            return p
    return current

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"

# API Settings
NBA_CDN_BASE = "https://cdn.nba.com/static/json/liveData"
REQUEST_DELAY = 0.6  # Seconds between API calls (be nice to servers)

# Season
def get_current_season() -> str:
    """Get current NBA season string (e.g., '2024-25')."""
    now = datetime.now()
    year = now.year if now.month >= 10 else now.year - 1
    return f"{year}-{str(year + 1)[2:]}"

CURRENT_SEASON = get_current_season()

print(f"📁 Project Root: {PROJECT_ROOT}")
print(f"🏀 Season: {CURRENT_SEASON}")


# ============================================================================
# NBA LIVE API (Direct CDN - No package needed)
# ============================================================================

class NBALiveAPI:
    """Direct access to NBA's live data CDN."""
    
    def __init__(self):
        self.base_url = NBA_CDN_BASE
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.nba.com/',
        }
    
    def _get(self, endpoint: str) -> Optional[Dict]:
        """Make GET request to NBA CDN."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                print(f"⚠️  HTTP {response.status_code} for {endpoint}")
                return None
        except Exception as e:
            print(f"❌ Error fetching {endpoint}: {e}")
            return None
    
    def get_scoreboard(self) -> Optional[Dict]:
        """Get today's scoreboard (games, scores, status)."""
        return self._get("scoreboard/todaysScoreboard_00.json")
    
    def get_odds(self) -> Optional[Dict]:
        """Get betting odds for today's games."""
        return self._get("odds/odds_todaysGames.json")
    
    def get_boxscore(self, game_id: str) -> Optional[Dict]:
        """Get boxscore for a specific game."""
        return self._get(f"boxscore/boxscore_{game_id}.json")


# ============================================================================
# DATA PROCESSORS
# ============================================================================

def process_scoreboard(data: Dict) -> pd.DataFrame:
    """Process scoreboard JSON into clean DataFrame."""
    if not data or 'scoreboard' not in data:
        return pd.DataFrame()
    
    games = data['scoreboard'].get('games', [])
    game_date = data['scoreboard'].get('gameDate', datetime.now().strftime('%Y-%m-%d'))
    
    rows = []
    for game in games:
        home = game.get('homeTeam', {})
        away = game.get('awayTeam', {})
        
        rows.append({
            'game_id': game.get('gameId'),
            'game_date': game_date,
            'game_status': game.get('gameStatus'),  # 1=scheduled, 2=in progress, 3=final
            'game_status_text': game.get('gameStatusText'),
            'home_team_id': home.get('teamId'),
            'home_team': f"{home.get('teamCity', '')} {home.get('teamName', '')}".strip(),
            'home_tricode': home.get('teamTricode'),
            'home_score': home.get('score', 0),
            'home_wins': home.get('wins', 0),
            'home_losses': home.get('losses', 0),
            'away_team_id': away.get('teamId'),
            'away_team': f"{away.get('teamCity', '')} {away.get('teamName', '')}".strip(),
            'away_tricode': away.get('teamTricode'),
            'away_score': away.get('score', 0),
            'away_wins': away.get('wins', 0),
            'away_losses': away.get('losses', 0),
        })
    
    return pd.DataFrame(rows)


def process_odds(data: Dict) -> pd.DataFrame:
    """Process odds JSON into clean DataFrame."""
    if not data or 'games' not in data:
        return pd.DataFrame()
    
    rows = []
    for game in data['games']:
        game_id = game.get('gameId')
        home_team_id = game.get('homeTeamId')
        away_team_id = game.get('awayTeamId')
        
        # Extract spread and moneyline from markets
        spread_home = None
        spread_away = None
        ml_home = None
        ml_away = None
        
        for market in game.get('markets', []):
            market_name = market.get('name')
            
            # Get first US book or any book
            books = market.get('books', [])
            us_books = [b for b in books if b.get('countryCode') == 'US']
            book = us_books[0] if us_books else (books[0] if books else None)
            
            if book:
                for outcome in book.get('outcomes', []):
                    outcome_type = outcome.get('type')
                    
                    if market_name == '2way':  # Moneyline
                        odds = float(outcome.get('odds', 0))
                        if outcome_type == 'home':
                            ml_home = odds
                        elif outcome_type == 'away':
                            ml_away = odds
                    
                    elif market_name == 'spread':
                        spread = outcome.get('spread')
                        if spread is not None:
                            spread = float(spread)
                            if outcome_type == 'home':
                                spread_home = spread
                            elif outcome_type == 'away':
                                spread_away = spread
        
        # Convert moneyline to implied probability
        def ml_to_prob(ml):
            if ml is None or ml <= 0:
                return None
            return 1 / ml
        
        rows.append({
            'game_id': game_id,
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'vegas_spread_home': spread_home,
            'vegas_spread_away': spread_away,
            'vegas_ml_home': ml_home,
            'vegas_ml_away': ml_away,
            'vegas_implied_home_prob': ml_to_prob(ml_home),
            'vegas_implied_away_prob': ml_to_prob(ml_away),
        })
    
    return pd.DataFrame(rows)


# ============================================================================
# NBA STATS API (requires nba_api package)
# ============================================================================

def refresh_team_ratings() -> Optional[pd.DataFrame]:
    """Fetch current team ratings from NBA Stats API."""
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        
        print("📊 Fetching team ratings...")
        time.sleep(REQUEST_DELAY)
        
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=CURRENT_SEASON,
            measure_type_detailed_defense='Advanced',
            per_mode_detailed='PerGame'
        )
        
        df = stats.get_data_frames()[0]
        
        # Select and rename columns
        result = df[['TEAM_ID', 'TEAM_NAME', 'W', 'L', 'GP',
                    'OFF_RATING', 'DEF_RATING', 'NET_RATING', 'PACE']].copy()
        
        result = result.rename(columns={
            'TEAM_ID': 'team_id',
            'TEAM_NAME': 'Team',
            'W': 'Wins',
            'L': 'Losses', 
            'GP': 'GamesPlayed',
            'OFF_RATING': 'ORtg',
            'DEF_RATING': 'DRtg',
            'NET_RATING': 'NetRtg',
            'PACE': 'Pace'
        })
        
        result['Season'] = CURRENT_SEASON
        result['updated_at'] = datetime.now().isoformat()
        
        print(f"✅ Got ratings for {len(result)} teams")
        return result
        
    except ImportError:
        print("❌ nba_api not installed. Run: pip install nba_api")
        return None
    except Exception as e:
        print(f"❌ Error fetching team ratings: {e}")
        return None


def refresh_team_game_logs(last_n_days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch recent team game logs for volatility calculations."""
    try:
        from nba_api.stats.endpoints import teamgamelogs
        
        print(f"📊 Fetching team game logs (last {last_n_days} days)...")
        time.sleep(REQUEST_DELAY)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=last_n_days)
        
        logs = teamgamelogs.TeamGameLogs(
            season_nullable=CURRENT_SEASON,
            date_from_nullable=start_date.strftime('%m/%d/%Y'),
            date_to_nullable=end_date.strftime('%m/%d/%Y')
        )
        
        df = logs.get_data_frames()[0]
        
        if df.empty:
            print("⚠️  No game logs found")
            return None
        
        # Clean up
        df['gameDate'] = pd.to_datetime(df['GAME_DATE'])
        df['Team'] = df['TEAM_NAME']
        df['teamScore'] = df['PTS']
        
        # Determine home/away and opponent score
        df['home'] = df['MATCHUP'].str.contains(' vs. ').astype(int)
        
        print(f"✅ Got {len(df)} game logs")
        return df
        
    except ImportError:
        print("❌ nba_api not installed. Run: pip install nba_api")
        return None
    except Exception as e:
        print(f"❌ Error fetching game logs: {e}")
        return None


# ============================================================================
# MAIN REFRESH FUNCTIONS
# ============================================================================

def refresh_today() -> Dict[str, pd.DataFrame]:
    """Refresh just today's data (fast)."""
    print("\n🔄 Refreshing today's data...")
    
    api = NBALiveAPI()
    results = {}
    
    # Today's games
    print("📅 Fetching today's scoreboard...")
    scoreboard_data = api.get_scoreboard()
    if scoreboard_data:
        results['todays_games'] = process_scoreboard(scoreboard_data)
        print(f"   Found {len(results['todays_games'])} games")
    
    time.sleep(REQUEST_DELAY)
    
    # Today's odds
    print("💰 Fetching today's odds...")
    odds_data = api.get_odds()
    if odds_data:
        results['todays_odds'] = process_odds(odds_data)
        print(f"   Found odds for {len(results['todays_odds'])} games")
    
    return results


def refresh_ratings() -> Dict[str, pd.DataFrame]:
    """Refresh team ratings."""
    print("\n🔄 Refreshing team ratings...")
    
    results = {}
    
    ratings = refresh_team_ratings()
    if ratings is not None:
        results['team_ratings'] = ratings
    
    return results


def refresh_full() -> Dict[str, pd.DataFrame]:
    """Full data refresh (slower)."""
    print("\n🔄 Full data refresh...")
    
    results = {}
    
    # Today's data
    today_data = refresh_today()
    results.update(today_data)
    
    time.sleep(REQUEST_DELAY)
    
    # Team ratings
    ratings_data = refresh_ratings()
    results.update(ratings_data)
    
    time.sleep(REQUEST_DELAY)
    
    # Game logs (for volatility)
    game_logs = refresh_team_game_logs(last_n_days=45)
    if game_logs is not None:
        results['team_game_logs'] = game_logs
    
    return results


# ============================================================================
# SAVE FUNCTIONS
# ============================================================================

def save_data(data: Dict[str, pd.DataFrame], data_dir: Path = None):
    """Save DataFrames to clean CSV structure."""
    if data_dir is None:
        data_dir = DATA_DIR
    
    # Create clean folder structure
    live_dir = data_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving data...")
    
    # Save each DataFrame
    file_map = {
        'todays_games': live_dir / 'todays_games.csv',
        'todays_odds': live_dir / 'todays_odds.csv',
        'team_ratings': live_dir / 'team_ratings.csv',
        'team_game_logs': data_dir / 'raw' / 'team_game_logs_recent.csv',
    }
    
    for name, df in data.items():
        if df is not None and not df.empty:
            filepath = file_map.get(name)
            if filepath:
                filepath.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(filepath, index=False)
                print(f"   ✅ {filepath.name}: {len(df)} rows")


def create_clean_structure():
    """Create the clean data folder structure."""
    print("\n📁 Creating clean folder structure...")
    
    folders = [
        DATA_DIR / "live",           # Daily refreshed data
        DATA_DIR / "raw",            # Historical data
        DATA_DIR / "injuries",       # Injury data
        DATA_DIR / "results" / "predictions",
        DATA_DIR / "results" / "backtests",
    ]
    
    for folder in folders:
        folder.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {folder.relative_to(DATA_DIR)}")


# ============================================================================
# CLI
# ============================================================================

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QEPC Data Refresh')
    parser.add_argument('--today', action='store_true', help='Refresh only today\'s data')
    parser.add_argument('--ratings', action='store_true', help='Refresh only team ratings')
    parser.add_argument('--full', action='store_true', help='Full refresh (default)')
    parser.add_argument('--setup', action='store_true', help='Create clean folder structure')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🔮 QEPC Data Refresh")
    print("=" * 60)
    
    if args.setup:
        create_clean_structure()
        return
    
    if args.today:
        data = refresh_today()
    elif args.ratings:
        data = refresh_ratings()
    else:
        data = refresh_full()
    
    if data:
        save_data(data)
        print("\n✅ Data refresh complete!")
    else:
        print("\n⚠️  No data was fetched")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
