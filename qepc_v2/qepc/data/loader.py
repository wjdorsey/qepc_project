"""
QEPC Data Loader
================
Handles loading and processing data from CSV files.

Clean Data Structure:
    data/
    ├── live/                    # Refreshed daily
    │   ├── todays_games.csv     # Today's schedule
    │   ├── todays_odds.csv      # Vegas lines
    │   └── team_ratings.csv     # Current ORtg/DRtg/Pace
    ├── raw/                     # Historical
    │   ├── TeamStatistics.csv   # Game-by-game stats
    │   └── team_game_logs_recent.csv
    ├── injuries/
    │   └── current_injuries.csv
    └── results/
        ├── predictions/
        └── backtests/
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """
    Central data loading class for QEPC.
    
    Handles both old file structure (for compatibility) and new clean structure.
    """
    
    def __init__(self, project_root: Path = None):
        """
        Initialize the data loader.
        
        Parameters
        ----------
        project_root : Path, optional
            Path to project root. If None, auto-detects.
        """
        self.project_root = project_root or self._find_project_root()
        self.data_dir = self.project_root / "data"
        
        # Cache for loaded data
        self._cache = {}
        
        print(f"📁 QEPC Data Loader initialized")
        print(f"   Project root: {self.project_root}")
    
    def _find_project_root(self) -> Path:
        """Auto-detect project root by looking for data folder."""
        current = Path.cwd()
        
        for p in [current] + list(current.parents)[:5]:
            if (p / "data").exists():
                return p
        
        return current
    
    def _make_tz_naive(self, series: pd.Series) -> pd.Series:
        """Convert datetime series to timezone-naive."""
        if series is None:
            return series
        if hasattr(series, 'dt') and series.dt.tz is not None:
            return series.dt.tz_convert('UTC').dt.tz_localize(None)
        return series
    
    # =========================================================================
    # TEAM RATINGS (Primary source for predictions)
    # =========================================================================
    
    def load_team_ratings(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load current team ratings (ORtg, DRtg, Pace).
        
        NEW: data/live/team_ratings.csv
        OLD: data/live/team_stats_live_nba_api.csv
        """
        if 'team_ratings' in self._cache and not refresh:
            return self._cache['team_ratings']
        
        paths = [
            self.data_dir / "live" / "team_ratings.csv",
            self.data_dir / "live" / "team_stats_live_nba_api.csv",
        ]
        
        df = self._load_first_available(paths, "Team Ratings")
        
        if df is not None:
            # Standardize column names
            rename_map = {
                'ORtg_live': 'ORtg',
                'DRtg_live': 'DRtg',
                'Pace_live': 'Pace',
                'NetRtg_live': 'NetRtg',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            self._cache['team_ratings'] = df
        
        return df
    
    # Alias for compatibility
    def load_live_ratings(self, refresh: bool = False) -> pd.DataFrame:
        return self.load_team_ratings(refresh)
    
    # =========================================================================
    # TODAY'S GAMES
    # =========================================================================
    
    def load_today_games(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load today's game schedule.
        
        NEW: data/live/todays_games.csv
        OLD: data/live/espn_scoreboard_today.csv
        """
        if 'today_games' in self._cache and not refresh:
            return self._cache['today_games']
        
        paths = [
            self.data_dir / "live" / "todays_games.csv",
            self.data_dir / "live" / "espn_scoreboard_today.csv",
            self.data_dir / "live" / "games_today_nba_api.csv",
        ]
        
        df = self._load_first_available(paths, "Today's Games")
        
        if df is not None:
            df = self._parse_dates(df)
            
            # Standardize column names
            rename_map = {
                'home_team': 'Home Team',
                'away_team': 'Away Team',
                'HOME_TEAM_NAME': 'Home Team',
                'AWAY_TEAM_NAME': 'Away Team',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            self._cache['today_games'] = df
        
        return df
    
    # =========================================================================
    # VEGAS ODDS (NEW!)
    # =========================================================================
    
    def load_vegas_odds(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load today's Vegas betting lines.
        
        Source: data/live/todays_odds.csv
        
        Returns DataFrame with:
        - game_id, home_team_id, away_team_id
        - vegas_spread_home, vegas_spread_away
        - vegas_ml_home, vegas_ml_away
        - vegas_implied_home_prob, vegas_implied_away_prob
        """
        if 'vegas_odds' in self._cache and not refresh:
            return self._cache['vegas_odds']
        
        paths = [
            self.data_dir / "live" / "todays_odds.csv",
        ]
        
        df = self._load_first_available(paths, "Vegas Odds")
        
        if df is not None:
            self._cache['vegas_odds'] = df
        
        return df
    
    def get_vegas_line(self, game_id: str) -> Optional[Dict]:
        """
        Get Vegas line for a specific game.
        
        Returns dict with spread, moneyline, implied probability.
        """
        odds = self.load_vegas_odds()
        if odds is None or odds.empty:
            return None
        
        game = odds[odds['game_id'] == game_id]
        if game.empty:
            return None
        
        row = game.iloc[0]
        return {
            'spread_home': row.get('vegas_spread_home'),
            'spread_away': row.get('vegas_spread_away'),
            'ml_home': row.get('vegas_ml_home'),
            'ml_away': row.get('vegas_ml_away'),
            'implied_home_prob': row.get('vegas_implied_home_prob'),
            'implied_away_prob': row.get('vegas_implied_away_prob'),
        }
    
    # =========================================================================
    # TEAM STATISTICS (Game-by-game for volatility)
    # =========================================================================
    
    def load_team_stats(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load game-by-game team statistics.
        
        Used for calculating volatility and recency-weighted stats.
        """
        if 'team_stats' in self._cache and not refresh:
            return self._cache['team_stats']
        
        paths = [
            self.data_dir / "raw" / "team_game_logs_recent.csv",
            self.data_dir / "raw" / "TeamStatistics.csv",
            self.data_dir / "TeamStatistics.csv",
        ]
        
        df = self._load_first_available(paths, "Team Stats")
        
        if df is not None:
            df = self._parse_dates(df)
            
            # Create Team column if needed
            if 'Team' not in df.columns:
                if 'teamCity' in df.columns and 'teamName' in df.columns:
                    df['Team'] = (df['teamCity'].fillna('') + ' ' + df['teamName'].fillna('')).str.strip()
                elif 'TEAM_NAME' in df.columns:
                    df['Team'] = df['TEAM_NAME']
            
            self._cache['team_stats'] = df
        
        return df
    
    def load_game_results(self, refresh: bool = False) -> pd.DataFrame:
        """Load game results for backtesting."""
        if 'game_results' in self._cache and not refresh:
            return self._cache['game_results']
        
        paths = [
            self.data_dir / "GameResults_2025.csv",
            self.data_dir / "raw" / "GameResults_2025.csv",
        ]
        
        df = self._load_first_available(paths, "Game Results")
        
        if df is not None:
            df = self._parse_dates(df, date_col='Date')
            self._cache['game_results'] = df
        
        return df
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    
    def load_injuries(self, refresh: bool = False) -> pd.DataFrame:
        """Load current injury report."""
        if 'injuries' in self._cache and not refresh:
            return self._cache['injuries']
        
        paths = [
            self.data_dir / "injuries" / "current_injuries.csv",
            self.data_dir / "Injury_Overrides_live_espn.csv",
            self.data_dir / "injuries" / "Injury_Overrides_MASTER.csv",
        ]
        
        df = self._load_first_available(paths, "Injuries")
        
        if df is not None:
            self._cache['injuries'] = df
        
        return df
    
    # =========================================================================
    # SCHEDULE & CONTEXT
    # =========================================================================
    
    def load_schedule_with_rest(self, refresh: bool = False) -> pd.DataFrame:
        """Load schedule with rest day information."""
        if 'schedule_rest' in self._cache and not refresh:
            return self._cache['schedule_rest']
        
        paths = [
            self.data_dir / "Schedule_with_Rest.csv",
        ]
        
        df = self._load_first_available(paths, "Schedule with Rest")
        
        if df is not None:
            df = self._parse_dates(df)
            self._cache['schedule_rest'] = df
        
        return df
    
    def load_team_form(self, refresh: bool = False) -> pd.DataFrame:
        """Load recent team form (last N games)."""
        if 'team_form' in self._cache and not refresh:
            return self._cache['team_form']
        
        paths = [
            self.data_dir / "TeamForm.csv",
        ]
        
        df = self._load_first_available(paths, "Team Form")
        
        if df is not None:
            self._cache['team_form'] = df
        
        return df
    
    # =========================================================================
    # PLAYER DATA (for props)
    # =========================================================================
    
    def load_player_averages(self, refresh: bool = False) -> pd.DataFrame:
        """Load player season averages."""
        if 'player_averages' in self._cache and not refresh:
            return self._cache['player_averages']
        
        paths = [
            self.data_dir / "props" / "Player_Season_Averages.csv",
            self.data_dir / "raw" / "Player_Props_Averages.csv",
        ]
        
        df = self._load_first_available(paths, "Player Averages")
        
        if df is not None:
            self._cache['player_averages'] = df
        
        return df
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _load_first_available(self, paths: List[Path], name: str) -> Optional[pd.DataFrame]:
        """Try to load from multiple paths, return first that works."""
        for path in paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    
                    # Check for Git LFS placeholders
                    if len(df.columns) == 1 and 'git-lfs' in str(df.columns[0]).lower():
                        print(f"⚠️  {name}: Git LFS placeholder (run `git lfs pull`)")
                        continue
                    
                    print(f"✅ Loaded {name}: {len(df):,} rows from {path.name}")
                    return df
                except Exception as e:
                    print(f"⚠️  Error loading {path.name}: {e}")
        
        print(f"❌ {name}: Not found")
        return None
    
    def _parse_dates(self, df: pd.DataFrame, date_col: str = None) -> pd.DataFrame:
        """Parse date columns to datetime and make timezone-naive."""
        if date_col is None:
            for col in ['gameDate', 'game_date', 'Date', 'date', 'GAME_DATE']:
                if col in df.columns:
                    date_col = col
                    break
        
        if date_col and date_col in df.columns:
            df['gameDate'] = pd.to_datetime(df[date_col], errors='coerce')
            df['gameDate'] = self._make_tz_naive(df['gameDate'])
            
            invalid = df['gameDate'].isna()
            if invalid.sum() > 0:
                df = df[~invalid].copy()
        
        return df
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cache = {}
        print("🗑️  Cache cleared")
    
    def get_data_summary(self) -> Dict:
        """Get summary of available data."""
        summary = {}
        
        # Team ratings
        ratings = self.load_team_ratings()
        if ratings is not None:
            summary['team_ratings'] = {'teams': len(ratings)}
        
        # Today's games
        games = self.load_today_games()
        if games is not None:
            summary['today_games'] = {'games': len(games)}
        
        # Vegas odds
        odds = self.load_vegas_odds()
        if odds is not None:
            summary['vegas_odds'] = {'games': len(odds)}
        
        # Team stats
        stats = self.load_team_stats()
        if stats is not None:
            summary['team_stats'] = {
                'rows': len(stats),
                'date_range': f"{stats['gameDate'].min().date()} to {stats['gameDate'].max().date()}" if 'gameDate' in stats.columns else 'N/A'
            }
        
        return summary


# Convenience function
def load_data(project_root: Path = None) -> DataLoader:
    """Create and return a DataLoader instance."""
    return DataLoader(project_root)
