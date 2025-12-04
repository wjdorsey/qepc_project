"""
QEPC Matchup Analysis
=====================
Analyzes how opponents defend specific stat categories.

Features:
- Team defense vs position (PG, SG, SF, PF, C)
- Team defense vs stat category (PTS, REB, AST, FG3M)
- Pace adjustments
- Usage opportunity modeling

Quantum Interpretation:
- Matchup effects are "interference patterns"
- Good defender = destructive interference (reduces production)
- Bad defender = constructive interference (boosts production)
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class MatchupProfile:
    """Defensive matchup profile for a team."""
    team: str
    
    # Overall defense (1.0 = league average)
    # < 1.0 = good defense (allows less)
    # > 1.0 = bad defense (allows more)
    pts_allowed_factor: float = 1.0
    reb_allowed_factor: float = 1.0
    ast_allowed_factor: float = 1.0
    fg3m_allowed_factor: float = 1.0
    
    # Pace factor (affects volume)
    pace_factor: float = 1.0
    
    # Position-specific factors
    vs_guards_factor: float = 1.0
    vs_wings_factor: float = 1.0
    vs_bigs_factor: float = 1.0


class MatchupAnalyzer:
    """
    Analyzes matchups for prop adjustments.
    
    Uses opponent defensive tendencies to adjust projections.
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or self._find_data_dir()
        
        self._team_defense: Dict[str, MatchupProfile] = {}
        self._pace_data: Dict[str, float] = {}
        self._league_averages: Dict[str, float] = {}
    
    def _find_data_dir(self) -> Path:
        """Find the data directory."""
        current = Path.cwd()
        for p in [current] + list(current.parents)[:5]:
            if (p / "data").exists():
                return p / "data"
        return current / "data"
    
    def load_data(self, verbose: bool = True):
        """Load all matchup-related data."""
        if verbose:
            print("🛡️ Loading matchup data...")
        
        self._load_team_defense(verbose)
        self._load_pace_data(verbose)
        self._calculate_league_averages(verbose)
        
        if verbose:
            print(f"   Loaded defense profiles for {len(self._team_defense)} teams")
    
    def _load_team_defense(self, verbose: bool = True):
        """Load team defensive statistics."""
        paths = [
            self.data_dir / "props" / "Team_Defense_Stats.csv",
            self.data_dir / "live" / "team_ratings.csv",
        ]
        
        for path in paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    self._build_defense_profiles(df)
                    if verbose:
                        print(f"   ✅ Loaded team defense from {path.name}")
                    return
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Error loading {path.name}: {e}")
        
        # Use defaults if no data
        self._use_default_profiles(verbose)
    
    def _build_defense_profiles(self, df: pd.DataFrame):
        """Build defense profiles from data."""
        # Find team name column
        team_col = None
        for col in ['TEAM_NAME', 'Team', 'team_name', 'TEAM_ABBREVIATION']:
            if col in df.columns:
                team_col = col
                break
        
        if team_col is None:
            return
        
        # Calculate league averages for normalization
        pts_avg = df['OPP_PTS'].mean() if 'OPP_PTS' in df.columns else 115
        reb_avg = df['OPP_REB'].mean() if 'OPP_REB' in df.columns else 44
        ast_avg = df['OPP_AST'].mean() if 'OPP_AST' in df.columns else 25
        fg3m_avg = df['OPP_FG3M'].mean() if 'OPP_FG3M' in df.columns else 13
        pace_avg = df['PACE'].mean() if 'PACE' in df.columns else 100
        
        for _, row in df.iterrows():
            team = row[team_col]
            
            profile = MatchupProfile(
                team=team,
                pts_allowed_factor=row.get('OPP_PTS', pts_avg) / pts_avg if pts_avg > 0 else 1.0,
                reb_allowed_factor=row.get('OPP_REB', reb_avg) / reb_avg if reb_avg > 0 else 1.0,
                ast_allowed_factor=row.get('OPP_AST', ast_avg) / ast_avg if ast_avg > 0 else 1.0,
                fg3m_allowed_factor=row.get('OPP_FG3M', fg3m_avg) / fg3m_avg if fg3m_avg > 0 else 1.0,
                pace_factor=row.get('PACE', pace_avg) / pace_avg if pace_avg > 0 else 1.0,
            )
            
            self._team_defense[team.lower()] = profile
    
    def _use_default_profiles(self, verbose: bool = True):
        """Create default neutral profiles."""
        if verbose:
            print("   ⚠️  Using default defense profiles (neutral)")
        
        # NBA teams
        teams = [
            "Atlanta Hawks", "Boston Celtics", "Brooklyn Nets", "Charlotte Hornets",
            "Chicago Bulls", "Cleveland Cavaliers", "Dallas Mavericks", "Denver Nuggets",
            "Detroit Pistons", "Golden State Warriors", "Houston Rockets", "Indiana Pacers",
            "LA Clippers", "Los Angeles Lakers", "Memphis Grizzlies", "Miami Heat",
            "Milwaukee Bucks", "Minnesota Timberwolves", "New Orleans Pelicans", 
            "New York Knicks", "Oklahoma City Thunder", "Orlando Magic", "Philadelphia 76ers",
            "Phoenix Suns", "Portland Trail Blazers", "Sacramento Kings", "San Antonio Spurs",
            "Toronto Raptors", "Utah Jazz", "Washington Wizards"
        ]
        
        for team in teams:
            self._team_defense[team.lower()] = MatchupProfile(team=team)
    
    def _load_pace_data(self, verbose: bool = True):
        """Load team pace data."""
        paths = [
            self.data_dir / "live" / "team_ratings.csv",
        ]
        
        for path in paths:
            if path.exists():
                try:
                    df = pd.read_csv(path)
                    
                    team_col = None
                    for col in ['Team', 'TEAM_NAME', 'team_name']:
                        if col in df.columns:
                            team_col = col
                            break
                    
                    if team_col and 'Pace' in df.columns:
                        avg_pace = df['Pace'].mean()
                        for _, row in df.iterrows():
                            self._pace_data[row[team_col].lower()] = row['Pace'] / avg_pace
                        
                        if verbose:
                            print(f"   ✅ Loaded pace data from {path.name}")
                        return
                except Exception as e:
                    pass
    
    def _calculate_league_averages(self, verbose: bool = True):
        """Calculate league average stats for reference."""
        # 2024-25 approximate league averages
        self._league_averages = {
            'pts': 115.0,
            'reb': 44.0,
            'ast': 26.0,
            'fg3m': 13.5,
            'pace': 100.0,
        }
    
    def get_matchup_modifier(
        self,
        opponent: str,
        stat_type: str,
        position: str = None,
    ) -> float:
        """
        Get the matchup modifier for a stat vs opponent.
        
        Parameters
        ----------
        opponent : str
            Opponent team name
        stat_type : str
            Stat category ('pts', 'reb', 'ast', 'fg3m')
        position : str, optional
            Player position for position-specific adjustments
        
        Returns
        -------
        float : Modifier (>1 = good matchup, <1 = bad matchup)
        """
        profile = self._team_defense.get(opponent.lower())
        
        if profile is None:
            return 1.0
        
        # Base factor for stat type
        factor_map = {
            'pts': profile.pts_allowed_factor,
            'reb': profile.reb_allowed_factor,
            'ast': profile.ast_allowed_factor,
            'fg3m': profile.fg3m_allowed_factor,
            'pra': (profile.pts_allowed_factor + profile.reb_allowed_factor + profile.ast_allowed_factor) / 3,
        }
        
        base_factor = factor_map.get(stat_type, 1.0)
        
        # Position adjustment
        if position:
            pos_lower = position.lower()
            if pos_lower in ['pg', 'sg', 'g']:
                base_factor *= profile.vs_guards_factor
            elif pos_lower in ['sf', 'pf', 'f']:
                base_factor *= profile.vs_wings_factor
            elif pos_lower in ['c', 'pf-c']:
                base_factor *= profile.vs_bigs_factor
        
        return base_factor
    
    def get_pace_modifier(self, team: str, opponent: str) -> float:
        """
        Get pace modifier for a game.
        
        Fast pace = more possessions = more stats.
        
        Parameters
        ----------
        team : str
            Player's team
        opponent : str
            Opponent team
        
        Returns
        -------
        float : Pace modifier (>1 = faster than average)
        """
        team_pace = self._pace_data.get(team.lower(), 1.0)
        opp_pace = self._pace_data.get(opponent.lower(), 1.0)
        
        # Game pace is average of both teams
        return (team_pace + opp_pace) / 2
    
    def get_full_adjustment(
        self,
        opponent: str,
        stat_type: str,
        player_team: str = None,
        position: str = None,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Get full adjustment factor with breakdown.
        
        Returns
        -------
        Tuple of (total_factor, breakdown_dict)
        """
        matchup = self.get_matchup_modifier(opponent, stat_type, position)
        pace = self.get_pace_modifier(player_team or "", opponent)
        
        total = matchup * pace
        
        breakdown = {
            'matchup': matchup,
            'pace': pace,
            'total': total,
        }
        
        return total, breakdown
    
    def get_best_matchups(
        self,
        stat_type: str = 'pts',
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Get teams that allow the most of a stat category.
        
        Returns sorted list of best matchups for that stat.
        """
        matchups = []
        
        for team_name, profile in self._team_defense.items():
            factor_map = {
                'pts': profile.pts_allowed_factor,
                'reb': profile.reb_allowed_factor,
                'ast': profile.ast_allowed_factor,
                'fg3m': profile.fg3m_allowed_factor,
            }
            
            matchups.append({
                'Team': profile.team,
                'Factor': factor_map.get(stat_type, 1.0),
                'Pace': profile.pace_factor,
            })
        
        df = pd.DataFrame(matchups)
        df = df.sort_values('Factor', ascending=False).head(n)
        
        return df
    
    def get_worst_matchups(
        self,
        stat_type: str = 'pts',
        n: int = 10,
    ) -> pd.DataFrame:
        """
        Get teams that allow the least of a stat category.
        
        Returns sorted list of worst matchups for that stat.
        """
        matchups = []
        
        for team_name, profile in self._team_defense.items():
            factor_map = {
                'pts': profile.pts_allowed_factor,
                'reb': profile.reb_allowed_factor,
                'ast': profile.ast_allowed_factor,
                'fg3m': profile.fg3m_allowed_factor,
            }
            
            matchups.append({
                'Team': profile.team,
                'Factor': factor_map.get(stat_type, 1.0),
                'Pace': profile.pace_factor,
            })
        
        df = pd.DataFrame(matchups)
        df = df.sort_values('Factor', ascending=True).head(n)
        
        return df


# =============================================================================
# ENTANGLEMENT: Teammate Usage Correlation
# =============================================================================

@dataclass
class UsageEntanglement:
    """
    Models how one player's usage affects teammates.
    
    Quantum interpretation: Players are "entangled" - when one
    player's usage collapses to high, teammates collapse to lower.
    """
    player_name: str
    team: str
    
    # How much this player's presence affects teammates
    # Negative = takes away from teammates (high usage star)
    # Positive = creates for teammates (playmaker)
    pts_impact: float = 0.0  # Impact on teammates' scoring
    ast_impact: float = 0.0  # Impact on teammates' assists
    reb_impact: float = 0.0  # Impact on teammates' rebounds
    usage_share: float = 0.0  # Player's share of team usage


def calculate_teammate_adjustment(
    target_usage: float,
    injured_players: list = None,
    resting_players: list = None,
) -> float:
    """
    Calculate usage boost when key players are out.
    
    When a high-usage player is out, remaining players
    get more opportunities.
    
    Parameters
    ----------
    target_usage : float
        Target player's normal usage rate
    injured_players : list
        List of injured player UsageEntanglement objects
    resting_players : list
        List of resting player UsageEntanglement objects
    
    Returns
    -------
    float : Usage multiplier (>1 = more opportunities)
    """
    if not injured_players and not resting_players:
        return 1.0
    
    # Calculate total usage freed up
    freed_usage = 0.0
    
    for player in (injured_players or []):
        freed_usage += player.usage_share
    
    for player in (resting_players or []):
        freed_usage += player.usage_share
    
    # Distribute proportionally based on target player's usage
    if freed_usage > 0:
        # Target player gets share proportional to their usage
        # But cap at reasonable levels
        boost = 1 + (freed_usage * 0.5 * target_usage)
        return min(1.3, boost)  # Cap at 30% boost
    
    return 1.0
