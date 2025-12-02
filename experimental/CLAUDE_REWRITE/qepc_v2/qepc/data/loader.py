"""
QEPC Data Loader
================
Handles loading and processing data from CSV files.

Designed to work with your actual data structure:
- data/raw/TeamStatistics.csv - Game-by-game team stats
- data/live/team_stats_live_nba_api.csv - Current team ratings
- data/GameResults_2025.csv - Game results
- data/Schedule_with_Rest.csv - Rest day info
- data/TeamForm.csv - Recent form
- data/Injury_Overrides.csv - Injury impacts
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
    
    Automatically finds your project root and loads data from CSVs.
    Works on both local JupyterLab and cloud environments.
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
        
        # Check current and parent directories
        for p in [current] + list(current.parents)[:5]:
            if (p / "data").exists():
                return p
        
        # Fallback to current directory
        return current
    
    def _make_tz_naive(self, series: pd.Series) -> pd.Series:
        """Convert datetime series to timezone-naive."""
        if series is None:
            return series
        if hasattr(series, 'dt') and series.dt.tz is not None:
            return series.dt.tz_convert('UTC').dt.tz_localize(None)
        return series
    
    # =========================================================================
    # TEAM STATISTICS (Game-by-game data)
    # =========================================================================
    
    def load_team_stats(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load game-by-game team statistics.
        
        Source: data/raw/TeamStatistics.csv
        
        Returns DataFrame with columns:
        - gameDate, teamCity, teamName, opponentTeamName
        - teamScore, opponentScore
        - Various box score stats
        - home, win
        """
        if 'team_stats' in self._cache and not refresh:
            return self._cache['team_stats']
        
        # Try multiple possible locations
        paths = [
            self.data_dir / "raw" / "TeamStatistics.csv",
            self.data_dir / "TeamStatistics.csv",
            self.data_dir / "raw" / "Team_Stats.csv",
        ]
        
        df = self._load_first_available(paths, "TeamStatistics")
        
        if df is not None:
            # Parse dates
            df = self._parse_dates(df)
            
            # Create full team names
            if 'teamCity' in df.columns and 'teamName' in df.columns:
                df['Team'] = (df['teamCity'].fillna('') + ' ' + df['teamName'].fillna('')).str.strip()
                df['Opponent'] = (df['opponentTeamCity'].fillna('') + ' ' + df['opponentTeamName'].fillna('')).str.strip()
            
            self._cache['team_stats'] = df
        
        return df
    
    def load_game_results(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load game results (simplified format).
        
        Source: data/GameResults_2025.csv
        """
        if 'game_results' in self._cache and not refresh:
            return self._cache['game_results']
        
        paths = [
            self.data_dir / "GameResults_2025.csv",
            self.data_dir / "raw" / "GameResults_2025.csv",
        ]
        
        df = self._load_first_available(paths, "GameResults")
        
        if df is not None:
            df = self._parse_dates(df, date_col='Date')
            self._cache['game_results'] = df
        
        return df
    
    # =========================================================================
    # LIVE TEAM RATINGS
    # =========================================================================
    
    def load_live_ratings(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load current team ratings (ORtg, DRtg, Pace).
        
        Source: data/live/team_stats_live_nba_api.csv
        
        This is the primary source for team strength!
        """
        if 'live_ratings' in self._cache and not refresh:
            return self._cache['live_ratings']
        
        paths = [
            self.data_dir / "live" / "team_stats_live_nba_api.csv",
            self.data_dir / "team_stats_live_nba_api.csv",
        ]
        
        df = self._load_first_available(paths, "Live Ratings")
        
        if df is not None:
            # Standardize column names
            rename_map = {
                'ORtg_live': 'ORtg',
                'DRtg_live': 'DRtg', 
                'Pace_live': 'Pace',
                'NetRtg_live': 'NetRtg',
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            
            self._cache['live_ratings'] = df
        
        return df
    
    # =========================================================================
    # SCHEDULE & CONTEXT
    # =========================================================================
    
    def load_schedule_with_rest(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load schedule with rest day information.
        
        Source: data/Schedule_with_Rest.csv
        """
        if 'schedule_rest' in self._cache and not refresh:
            return self._cache['schedule_rest']
        
        paths = [
            self.data_dir / "Schedule_with_Rest.csv",
            self.data_dir / "raw" / "Schedule_with_Rest.csv",
        ]
        
        df = self._load_first_available(paths, "Schedule with Rest")
        
        if df is not None:
            df = self._parse_dates(df)
            self._cache['schedule_rest'] = df
        
        return df
    
    def load_team_form(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load recent team form (last N games performance).
        
        Source: data/TeamForm.csv
        """
        if 'team_form' in self._cache and not refresh:
            return self._cache['team_form']
        
        paths = [
            self.data_dir / "TeamForm.csv",
            self.data_dir / "raw" / "TeamForm.csv",
        ]
        
        df = self._load_first_available(paths, "Team Form")
        
        if df is not None:
            self._cache['team_form'] = df
        
        return df
    
    def load_today_games(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load today's games schedule.
        
        Source: data/live/espn_scoreboard_today.csv
        """
        if 'today_games' in self._cache and not refresh:
            return self._cache['today_games']
        
        paths = [
            self.data_dir / "live" / "espn_scoreboard_today.csv",
            self.data_dir / "live" / "games_today_nba_api.csv",
        ]
        
        df = self._load_first_available(paths, "Today's Games")
        
        if df is not None:
            df = self._parse_dates(df)
            self._cache['today_games'] = df
        
        return df
    
    # =========================================================================
    # INJURIES
    # =========================================================================
    
    def load_injuries(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load current injury report.
        
        Source: data/Injury_Overrides.csv or data/injuries/Injury_Overrides_MASTER.csv
        """
        if 'injuries' in self._cache and not refresh:
            return self._cache['injuries']
        
        paths = [
            self.data_dir / "Injury_Overrides_live_espn.csv",
            self.data_dir / "injuries" / "Injury_Overrides_MASTER.csv",
            self.data_dir / "Injury_Overrides.csv",
        ]
        
        df = self._load_first_available(paths, "Injuries")
        
        if df is not None:
            self._cache['injuries'] = df
        
        return df
    
    # =========================================================================
    # PLAYER DATA (for props)
    # =========================================================================
    
    def load_player_averages(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load player season averages with statistics.
        
        Source: data/props/Player_Season_Averages.csv
        """
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
    
    def load_player_recent_form(self, lookback: int = 5, refresh: bool = False) -> pd.DataFrame:
        """
        Load player recent form.
        
        Parameters
        ----------
        lookback : int
            Number of games to look back (5, 10, or 15)
        """
        cache_key = f'player_form_{lookback}'
        if cache_key in self._cache and not refresh:
            return self._cache[cache_key]
        
        paths = [
            self.data_dir / "props" / f"Player_Recent_Form_L{lookback}.csv",
            self.data_dir / f"Player_Recent_Form_L{lookback}.csv",
        ]
        
        df = self._load_first_available(paths, f"Player Form (L{lookback})")
        
        if df is not None:
            self._cache[cache_key] = df
        
        return df
    
    def load_player_home_away(self, refresh: bool = False) -> pd.DataFrame:
        """
        Load player home/away splits.
        
        Source: data/props/Player_Home_Away_Splits.csv
        """
        if 'player_home_away' in self._cache and not refresh:
            return self._cache['player_home_away']
        
        paths = [
            self.data_dir / "props" / "Player_Home_Away_Splits.csv",
        ]
        
        df = self._load_first_available(paths, "Player Home/Away")
        
        if df is not None:
            self._cache['player_home_away'] = df
        
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
        # Find date column
        if date_col is None:
            for col in ['gameDate', 'Date', 'date', 'GAME_DATE', 'Last_Game_Date']:
                if col in df.columns:
                    date_col = col
                    break
        
        if date_col and date_col in df.columns:
            # Parse to datetime
            df['gameDate'] = pd.to_datetime(df[date_col], errors='coerce')
            
            # Make timezone-naive
            df['gameDate'] = self._make_tz_naive(df['gameDate'])
            
            # Drop rows with invalid dates
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
        
        # Team stats
        ts = self.load_team_stats()
        if ts is not None:
            summary['team_stats'] = {
                'rows': len(ts),
                'date_range': f"{ts['gameDate'].min().date()} to {ts['gameDate'].max().date()}" if 'gameDate' in ts.columns else 'N/A',
                'teams': ts['Team'].nunique() if 'Team' in ts.columns else 'N/A',
            }
        
        # Live ratings
        lr = self.load_live_ratings()
        if lr is not None:
            summary['live_ratings'] = {
                'rows': len(lr),
                'teams': len(lr),
            }
        
        # Injuries
        inj = self.load_injuries()
        if inj is not None:
            summary['injuries'] = {
                'rows': len(inj),
                'out': len(inj[inj['Status'] == 'Out']) if 'Status' in inj.columns else 'N/A',
            }
        
        return summary


# Convenience function
def load_data(project_root: Path = None) -> DataLoader:
    """Create and return a DataLoader instance."""
    return DataLoader(project_root)
