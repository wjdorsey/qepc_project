"""
QEPC NFL Team Strengths Calculator
===================================

Calculates team strength ratings from historical game data.

Features:
- Drive-based efficiency metrics
- Recency weighting
- Volatility/consistency measurement
- Momentum calculation
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class StrengthConfig:
    """Configuration for strength calculations."""
    
    # Recency weighting
    HALF_LIFE_WEEKS = 6          # 6 weeks for 50% decay
    MIN_GAMES = 3                 # Minimum games for reliable rating
    
    # League averages (2023-24 NFL)
    LEAGUE_AVG_POINTS = 21.5      # Points per game
    LEAGUE_AVG_YARDS = 330.0      # Yards per game
    LEAGUE_AVG_DRIVES = 11.5      # Drives per game
    LEAGUE_AVG_PPD = 1.87         # Points per drive
    LEAGUE_TURNOVER_RATE = 0.12   # Turnovers per drive
    
    # Momentum
    MOMENTUM_LOOKBACK = 4         # Games for momentum calc
    MOMENTUM_DECAY = 0.85         # Week-over-week decay


# =============================================================================
# STRENGTH CALCULATOR
# =============================================================================

class NFLStrengthCalculator:
    """
    Calculates team strength ratings from game data.
    
    Usage:
        calc = NFLStrengthCalculator()
        calc.load_games("path/to/games.csv")
        strengths = calc.calculate_all_strengths()
    """
    
    def __init__(self, config: StrengthConfig = None):
        self.config = config or StrengthConfig()
        self.games: Optional[pd.DataFrame] = None
        self.strengths: Dict[str, dict] = {}
    
    def load_games(self, filepath: str, cutoff_date: str = None):
        """
        Load game data.
        
        Expected columns:
        - game_date
        - home_team, away_team
        - home_score, away_score
        - (optional) home_yards, away_yards
        - (optional) home_turnovers, away_turnovers
        """
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Parse dates
        date_cols = ['game_date', 'gamedate', 'date']
        for col in date_cols:
            if col in df.columns:
                df['game_date'] = pd.to_datetime(df[col], errors='coerce')
                break
        
        # Apply cutoff
        if cutoff_date:
            cutoff = pd.Timestamp(cutoff_date)
            df = df[df['game_date'] < cutoff].copy()
        
        # Standardize team columns
        team_mappings = {
            'home_team': ['home_team', 'hometeam', 'home'],
            'away_team': ['away_team', 'awayteam', 'away'],
            'home_score': ['home_score', 'homescore', 'home_pts'],
            'away_score': ['away_score', 'awayscore', 'away_pts'],
        }
        
        for target, candidates in team_mappings.items():
            for col in candidates:
                if col in df.columns:
                    df[target] = df[col]
                    break
        
        self.games = df
        print(f"[Strengths] Loaded {len(df)} games")
    
    def _calculate_weights(self, dates: pd.Series) -> pd.Series:
        """Calculate recency weights."""
        if dates.empty:
            return pd.Series(dtype=float)
        
        max_date = dates.max()
        weeks_ago = (max_date - dates).dt.days / 7
        
        weights = 0.5 ** (weeks_ago / self.config.HALF_LIFE_WEEKS)
        return weights
    
    def _weighted_mean(self, values: pd.Series, weights: pd.Series) -> float:
        """Calculate weighted mean."""
        mask = values.notna() & weights.notna()
        if not mask.any():
            return np.nan
        return np.average(values[mask], weights=weights[mask])
    
    def _weighted_std(self, values: pd.Series, weights: pd.Series) -> float:
        """Calculate weighted standard deviation."""
        mask = values.notna() & weights.notna()
        if mask.sum() < 2:
            return np.nan
        
        v = values[mask].values
        w = weights[mask].values
        
        mean = np.average(v, weights=w)
        variance = np.average((v - mean) ** 2, weights=w)
        return np.sqrt(variance)
    
    def calculate_team_strength(self, team: str) -> dict:
        """Calculate strength ratings for a single team."""
        if self.games is None:
            raise ValueError("No games loaded!")
        
        # Get all games for this team
        home_games = self.games[self.games['home_team'] == team].copy()
        away_games = self.games[self.games['away_team'] == team].copy()
        
        # Create unified view
        home_games['is_home'] = True
        home_games['team_score'] = home_games['home_score']
        home_games['opp_score'] = home_games['away_score']
        home_games['opponent'] = home_games['away_team']
        
        away_games['is_home'] = False
        away_games['team_score'] = away_games['away_score']
        away_games['opp_score'] = away_games['home_score']
        away_games['opponent'] = away_games['home_team']
        
        all_games = pd.concat([home_games, away_games], ignore_index=True)
        all_games = all_games.sort_values('game_date', ascending=False)
        
        n_games = len(all_games)
        
        if n_games < self.config.MIN_GAMES:
            return None
        
        # Calculate weights
        weights = self._calculate_weights(all_games['game_date'])
        
        # === OFFENSIVE METRICS ===
        avg_points_scored = self._weighted_mean(all_games['team_score'], weights)
        off_efficiency = avg_points_scored / self.config.LEAGUE_AVG_POINTS
        
        # Explosiveness (games with 28+ points)
        high_scoring = (all_games['team_score'] >= 28).astype(float)
        explosiveness = self._weighted_mean(high_scoring, weights) / 0.25  # Normalize to ~1.0
        explosiveness = max(0.5, min(2.0, explosiveness))  # Clamp
        
        # === DEFENSIVE METRICS ===
        avg_points_allowed = self._weighted_mean(all_games['opp_score'], weights)
        def_efficiency = avg_points_allowed / self.config.LEAGUE_AVG_POINTS
        
        # Defensive explosiveness (games allowing 28+)
        high_allowed = (all_games['opp_score'] >= 28).astype(float)
        def_explosiveness = self._weighted_mean(high_allowed, weights) / 0.25
        def_explosiveness = max(0.5, min(2.0, def_explosiveness))
        
        # === CONSISTENCY/VOLATILITY ===
        off_volatility = self._weighted_std(all_games['team_score'], weights)
        def_volatility = self._weighted_std(all_games['opp_score'], weights)
        
        if np.isnan(off_volatility):
            off_volatility = 7.0  # Default
        if np.isnan(def_volatility):
            def_volatility = 7.0
        
        # Normalize volatility to 0-1 scale (7 pts std = 0.5)
        off_consistency = 1 - min(1.0, off_volatility / 14.0)
        def_consistency = 1 - min(1.0, def_volatility / 14.0)
        
        # === TURNOVER METRICS ===
        if 'home_turnovers' in all_games.columns:
            home_to = all_games[all_games['is_home']]['home_turnovers']
            away_to = all_games[~all_games['is_home']]['away_turnovers']
            turnovers = pd.concat([home_to, away_to])
            turnover_rate = turnovers.mean() / self.config.LEAGUE_AVG_DRIVES
        else:
            # Estimate from point differential
            turnover_rate = 0.12  # Default
        
        takeaway_rate = 0.12  # Default (would need opponent turnover data)
        
        # === HOME/AWAY SPLITS ===
        home_avg = home_games['home_score'].mean() if len(home_games) > 0 else avg_points_scored
        away_avg = away_games['away_score'].mean() if len(away_games) > 0 else avg_points_scored
        home_boost = (home_avg - away_avg) / self.config.LEAGUE_AVG_POINTS / 2
        home_boost = max(-0.05, min(0.10, home_boost))  # Clamp
        
        # === MOMENTUM ===
        recent_games = all_games.head(self.config.MOMENTUM_LOOKBACK)
        if len(recent_games) >= 2:
            recent_margin = (recent_games['team_score'] - recent_games['opp_score']).mean()
            season_margin = (all_games['team_score'] - all_games['opp_score']).mean()
            
            # Momentum = how much better/worse than season average
            momentum = (recent_margin - season_margin) / 10.0  # Normalize
            momentum = max(-1.0, min(1.0, momentum))
        else:
            momentum = 0.0
        
        return {
            'team': team,
            'games_played': n_games,
            'off_efficiency': round(off_efficiency, 3),
            'def_efficiency': round(def_efficiency, 3),
            'off_explosiveness': round(explosiveness, 3),
            'def_explosiveness': round(def_explosiveness, 3),
            'off_consistency': round(off_consistency, 3),
            'def_consistency': round(def_consistency, 3),
            'volatility': round((1 - off_consistency + 1 - def_consistency) / 2, 3),
            'turnover_rate': round(turnover_rate, 3),
            'takeaway_rate': round(takeaway_rate, 3),
            'home_boost': round(home_boost, 4),
            'momentum': round(momentum, 3),
            'avg_points_scored': round(avg_points_scored, 1),
            'avg_points_allowed': round(avg_points_allowed, 1),
            'point_differential': round(avg_points_scored - avg_points_allowed, 1),
        }
    
    def calculate_all_strengths(self) -> pd.DataFrame:
        """Calculate strengths for all teams."""
        if self.games is None:
            raise ValueError("No games loaded!")
        
        # Get all unique teams
        teams = set(self.games['home_team'].unique()) | set(self.games['away_team'].unique())
        
        results = []
        for team in teams:
            strength = self.calculate_team_strength(team)
            if strength:
                results.append(strength)
                self.strengths[team] = strength
        
        df = pd.DataFrame(results)
        df = df.sort_values('off_efficiency', ascending=False)
        
        print(f"[Strengths] Calculated ratings for {len(df)} teams")
        return df
    
    def get_strength(self, team: str) -> Optional[dict]:
        """Get strength for a specific team."""
        return self.strengths.get(team)
    
    def save_strengths(self, filepath: str):
        """Save strengths to CSV."""
        df = pd.DataFrame(list(self.strengths.values()))
        df.to_csv(filepath, index=False)
        print(f"[Strengths] Saved to {filepath}")
    
    def power_rankings(self) -> pd.DataFrame:
        """Generate power rankings."""
        if not self.strengths:
            raise ValueError("Calculate strengths first!")
        
        df = pd.DataFrame(list(self.strengths.values()))
        
        # Calculate overall rating
        df['overall_rating'] = (
            df['off_efficiency'] / df['def_efficiency'] 
            * (1 + df['momentum'] * 0.1)
        )
        
        df = df.sort_values('overall_rating', ascending=False)
        df['rank'] = range(1, len(df) + 1)
        
        return df[['rank', 'team', 'overall_rating', 'off_efficiency', 
                   'def_efficiency', 'momentum', 'point_differential']]


# =============================================================================
# STRENGTH FROM SUMMARY DATA
# =============================================================================

def calculate_strengths_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate strengths from summary team stats.
    
    Expected columns:
    - team
    - points_per_game (or ppg)
    - points_allowed_per_game (or papg)
    - yards_per_game (optional)
    - turnovers (optional)
    """
    df = df.copy()
    df.columns = df.columns.str.lower().str.strip()
    
    # Standardize column names
    if 'ppg' in df.columns:
        df['points_per_game'] = df['ppg']
    if 'papg' in df.columns:
        df['points_allowed_per_game'] = df['papg']
    
    # League averages
    league_ppg = df['points_per_game'].mean()
    league_papg = df['points_allowed_per_game'].mean()
    
    # Calculate efficiencies
    df['off_efficiency'] = df['points_per_game'] / league_ppg
    df['def_efficiency'] = df['points_allowed_per_game'] / league_papg
    
    # Default other values
    df['off_explosiveness'] = 1.0
    df['def_explosiveness'] = 1.0
    df['volatility'] = 0.5
    df['turnover_rate'] = 0.12
    df['takeaway_rate'] = 0.12
    df['home_boost'] = 0.03
    df['momentum'] = 0.0
    
    return df


# =============================================================================
# GENERATE SAMPLE DATA
# =============================================================================

def generate_sample_nfl_games(n_weeks: int = 10) -> pd.DataFrame:
    """Generate sample NFL game data for testing."""
    
    teams = [
        "Kansas City Chiefs", "Buffalo Bills", "San Francisco 49ers",
        "Detroit Lions", "Philadelphia Eagles", "Dallas Cowboys",
        "Baltimore Ravens", "Miami Dolphins", "Cleveland Browns",
        "Cincinnati Bengals", "Jacksonville Jaguars", "Houston Texans",
        "Los Angeles Rams", "Seattle Seahawks", "Green Bay Packers",
        "Minnesota Vikings", "Chicago Bears", "New York Jets",
        "New England Patriots", "Las Vegas Raiders", "Denver Broncos",
        "Los Angeles Chargers", "Pittsburgh Steelers", "Tennessee Titans",
        "Indianapolis Colts", "Atlanta Falcons", "New Orleans Saints",
        "Tampa Bay Buccaneers", "Carolina Panthers", "Arizona Cardinals",
        "New York Giants", "Washington Commanders"
    ]
    
    # Team strength tiers
    strength_tiers = {
        0: teams[:8],    # Elite
        1: teams[8:16],  # Good
        2: teams[16:24], # Average
        3: teams[24:],   # Below average
    }
    
    def get_tier(team):
        for tier, team_list in strength_tiers.items():
            if team in team_list:
                return tier
        return 2
    
    games = []
    base_date = datetime(2024, 9, 8)  # Start of season
    
    for week in range(n_weeks):
        week_date = base_date + timedelta(weeks=week)
        
        # Shuffle teams for matchups
        week_teams = teams.copy()
        np.random.shuffle(week_teams)
        
        for i in range(0, len(week_teams), 2):
            home = week_teams[i]
            away = week_teams[i + 1]
            
            # Generate scores based on tiers
            home_tier = get_tier(home)
            away_tier = get_tier(away)
            
            home_base = 24 - home_tier * 2
            away_base = 24 - away_tier * 2
            
            # Add randomness
            home_score = max(0, int(np.random.normal(home_base + 3, 8)))  # Home advantage
            away_score = max(0, int(np.random.normal(away_base, 8)))
            
            games.append({
                'game_date': week_date,
                'week': week + 1,
                'home_team': home,
                'away_team': away,
                'home_score': home_score,
                'away_score': away_score,
            })
    
    return pd.DataFrame(games)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample NFL games...")
    games = generate_sample_nfl_games(10)
    games.to_csv("sample_nfl_games.csv", index=False)
    print(f"Created sample_nfl_games.csv with {len(games)} games")
    
    # Calculate strengths
    calc = NFLStrengthCalculator()
    calc.load_games("sample_nfl_games.csv")
    
    strengths = calc.calculate_all_strengths()
    print("\nTeam Strengths:")
    print(strengths.head(10).to_string())
    
    print("\nPower Rankings:")
    rankings = calc.power_rankings()
    print(rankings.to_string())
