"""
QEPC Configuration
==================
All tunable parameters in one place.

Modify these to calibrate the model based on backtest results.
"""

from dataclasses import dataclass
from typing import Dict
from datetime import datetime


@dataclass
class LeagueConfig:
    """League-wide settings for NBA 2024-25 season."""
    
    # Scoring environment
    league_avg_points: float = 114.5  # 2024-25 average
    league_avg_pace: float = 100.0    # Possessions per 48 min
    league_avg_ortg: float = 114.5    # Offensive rating
    league_avg_drtg: float = 114.5    # Defensive rating
    
    # Score bounds
    min_realistic_score: int = 70
    max_realistic_score: int = 175
    
    # Default volatility if not calculated
    default_team_std: float = 11.0  # Points


@dataclass
class HomeCourtConfig:
    """Home court advantage settings."""
    
    # Base advantage in points
    base_advantage: float = 3.2
    
    # Team-specific modifiers (multiply by base)
    team_multipliers: Dict[str, float] = None
    
    def __post_init__(self):
        if self.team_multipliers is None:
            self.team_multipliers = {
                # Altitude advantage
                "Denver Nuggets": 1.25,
                "Utah Jazz": 1.15,
                
                # Strong home environments
                "Boston Celtics": 1.12,
                "Golden State Warriors": 1.10,
                "Miami Heat": 1.10,
                "Phoenix Suns": 1.08,
                "Oklahoma City Thunder": 1.10,
                "Memphis Grizzlies": 1.08,
                "Cleveland Cavaliers": 1.10,
                "New York Knicks": 1.08,
                "Philadelphia 76ers": 1.08,
                
                # Average home courts
                "Los Angeles Lakers": 1.02,
                "Milwaukee Bucks": 1.05,
                "Dallas Mavericks": 1.05,
                "Minnesota Timberwolves": 1.05,
                
                # Weaker home courts
                "Brooklyn Nets": 0.92,
                "Los Angeles Clippers": 0.95,
                "Washington Wizards": 0.95,
            }
    
    def get_advantage(self, team: str) -> float:
        """Get home court advantage in points for a team."""
        multiplier = self.team_multipliers.get(team, 1.0)
        return self.base_advantage * multiplier


@dataclass
class RestConfig:
    """Rest and fatigue settings."""
    
    # Points advantage per day of rest difference
    rest_advantage_per_day: float = 1.2
    
    # Maximum rest advantage (caps at 4 days difference)
    max_rest_advantage: float = 4.5
    
    # Back-to-back penalty in points
    b2b_penalty: float = 2.8
    
    # Extended rest bonus (4+ days)
    extended_rest_bonus: float = 1.5
    
    # Travel penalty per 1000 miles (estimated)
    travel_penalty_per_1000mi: float = 0.8


@dataclass
class QuantumConfig:
    """Quantum mechanics parameters."""
    
    # Entanglement: Correlation between team performances
    # Higher = when one team exceeds expectations, other tends to underperform
    entanglement_strength: float = 0.35
    
    # Tunneling: Upset probability floor
    # Even huge favorites can lose
    tunneling_base_rate: float = 0.08
    tunneling_min_prob: float = 0.02  # 2% minimum for any team
    
    # Interference: Matchup amplification
    # How much do good offense vs bad defense compound?
    interference_factor: float = 0.15
    
    # Variance adjustments
    primetime_variance_boost: float = 0.08  # More variance in big games
    playoff_variance_reduction: float = 0.10  # Less variance in playoffs (tighter D)


@dataclass
class SimulationConfig:
    """Monte Carlo simulation settings."""
    
    # Number of simulations per prediction
    default_simulations: int = 20000
    
    # Quick mode for testing
    quick_simulations: int = 5000
    
    # High precision mode
    precision_simulations: int = 50000
    
    # Score correlation (pace entanglement)
    score_correlation: float = 0.35
    
    # Overtime settings
    avg_ot_points: float = 10.0
    ot_std: float = 3.0
    home_ot_edge: float = 0.52  # Home team wins 52% of OT


@dataclass 
class RecencyConfig:
    """Recency weighting for team strength calculations."""
    
    # Half-life in days (after this many days, weight is 50%)
    half_life_days: int = 21
    
    # Minimum games required for stable rating
    min_games: int = 5
    
    # Maximum lookback (games older than this get 0 weight)
    max_lookback_days: int = 90
    
    # Form lookback for momentum calculation
    form_games: int = 5


@dataclass
class PropConfig:
    """Player props prediction settings."""
    
    # Minimum games for reliable projection
    min_games: int = 5
    
    # Recency weighting half-life (games)
    half_life_games: int = 10
    
    # Home/away adjustment factor
    home_away_factor: float = 0.6  # Apply 60% of historical home/away split
    
    # Opponent adjustment factor
    opponent_adjustment: float = 0.5  # Apply 50% of opponent strength
    
    # Minutes correlation
    minutes_pts_correlation: float = 0.85
    
    # Confidence thresholds
    high_confidence_threshold: float = 0.65
    low_confidence_threshold: float = 0.55


class QEPCConfig:
    """Master configuration class combining all settings."""
    
    def __init__(self):
        self.league = LeagueConfig()
        self.home_court = HomeCourtConfig()
        self.rest = RestConfig()
        self.quantum = QuantumConfig()
        self.simulation = SimulationConfig()
        self.recency = RecencyConfig()
        self.props = PropConfig()
        
        # Version tracking
        self.version = "2.0.0"
        self.created = datetime.now().isoformat()
    
    def get_season(self) -> str:
        """Get current NBA season string."""
        now = datetime.now()
        year = now.year
        if now.month >= 10:  # Season starts in October
            return f"{year}-{str(year+1)[2:]}"
        return f"{year-1}-{str(year)[2:]}"


# Global default config
DEFAULT_CONFIG = QEPCConfig()


# Quick access functions
def get_config() -> QEPCConfig:
    """Get the default configuration."""
    return DEFAULT_CONFIG


def get_hca(team: str) -> float:
    """Get home court advantage for a team."""
    return DEFAULT_CONFIG.home_court.get_advantage(team)


def get_league_avg() -> float:
    """Get league average points per game."""
    return DEFAULT_CONFIG.league.league_avg_points
