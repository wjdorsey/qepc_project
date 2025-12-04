"""
QEPC Player State Modeling
==========================
Quantum-inspired state modeling for individual player performance.

A player exists in a superposition of performance states:
- CEILING: 95th percentile game (everything clicks)
- ELEVATED: 75th percentile (above average night)
- BASELINE: 50th percentile (typical performance)
- FLOOR: 25th percentile (off night)
- BUST: 5th percentile (disaster game)

The probability distribution shifts based on:
- Recent form (hot/cold streaks)
- Matchup quality (defensive opponent)
- Home/away
- Rest days
- Minutes projection
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from enum import Enum


class PerformanceState(Enum):
    """Player performance states with multipliers."""
    CEILING = ("ceiling", 1.35, 0.05)   # 95th percentile, 5% base probability
    ELEVATED = ("elevated", 1.15, 0.20)  # 75th percentile, 20% probability
    BASELINE = ("baseline", 1.00, 0.50)  # 50th percentile, 50% probability
    FLOOR = ("floor", 0.85, 0.20)        # 25th percentile, 20% probability
    BUST = ("bust", 0.50, 0.05)          # 5th percentile, 5% probability
    
    def __init__(self, label: str, multiplier: float, base_prob: float):
        self.label = label
        self.multiplier = multiplier
        self.base_prob = base_prob


@dataclass
class PlayerProfile:
    """Complete statistical profile for a player."""
    player_id: str
    player_name: str
    team: str
    position: str
    
    # Season averages
    games_played: int = 0
    minutes_avg: float = 0.0
    
    # Per-game averages
    pts_avg: float = 0.0
    reb_avg: float = 0.0
    ast_avg: float = 0.0
    stl_avg: float = 0.0
    blk_avg: float = 0.0
    tov_avg: float = 0.0
    fg3m_avg: float = 0.0  # 3-pointers made
    
    # Standard deviations (volatility)
    pts_std: float = 0.0
    reb_std: float = 0.0
    ast_std: float = 0.0
    fg3m_std: float = 0.0
    
    # Consistency metrics (coefficient of variation)
    pts_cv: float = 0.0  # std/mean - lower = more consistent
    reb_cv: float = 0.0
    ast_cv: float = 0.0
    
    # Usage and role
    usage_rate: float = 0.0
    
    # Recent form (last 5 games vs season)
    pts_l5: float = 0.0
    reb_l5: float = 0.0
    ast_l5: float = 0.0
    
    # Home/away splits
    pts_home: float = 0.0
    pts_away: float = 0.0
    
    # Per-minute rates (for minutes-adjusted projections)
    pts_per_min: float = 0.0
    reb_per_min: float = 0.0
    ast_per_min: float = 0.0
    
    @property
    def pra_avg(self) -> float:
        """Points + Rebounds + Assists average."""
        return self.pts_avg + self.reb_avg + self.ast_avg
    
    @property
    def fantasy_avg(self) -> float:
        """DraftKings fantasy points average."""
        return (self.pts_avg * 1.0 + 
                self.reb_avg * 1.25 + 
                self.ast_avg * 1.5 + 
                self.stl_avg * 2.0 + 
                self.blk_avg * 2.0 - 
                self.tov_avg * 0.5)
    
    @property
    def momentum(self) -> float:
        """Recent form vs season average (-1 to 1)."""
        if self.pts_avg == 0:
            return 0.0
        return np.clip((self.pts_l5 - self.pts_avg) / self.pts_avg, -1, 1)


@dataclass 
class PlayerQuantumState:
    """
    Quantum state representation for a player's performance distribution.
    
    The player exists in a superposition of performance states,
    with probabilities that shift based on context.
    """
    player: PlayerProfile
    
    # State probabilities (must sum to 1)
    amplitudes: Dict[PerformanceState, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with base probabilities."""
        if not self.amplitudes:
            self.amplitudes = {
                state: state.base_prob for state in PerformanceState
            }
        self._normalize()
    
    def _normalize(self):
        """Ensure probabilities sum to 1."""
        total = sum(self.amplitudes.values())
        if total > 0:
            self.amplitudes = {k: v/total for k, v in self.amplitudes.items()}
    
    def shift_toward(self, target_state: PerformanceState, strength: float = 0.1):
        """
        Shift probability mass toward a target state.
        
        Used for situational adjustments (hot streak, good matchup, etc.)
        """
        # Add probability to target
        self.amplitudes[target_state] += strength
        
        # Remove equally from others
        others = [s for s in PerformanceState if s != target_state]
        reduction = strength / len(others)
        for state in others:
            self.amplitudes[state] = max(0.01, self.amplitudes[state] - reduction)
        
        self._normalize()
    
    def increase_variance(self, factor: float = 0.1):
        """
        Increase probability of extreme states (ceiling/bust).
        
        Used for volatile players or high-variance situations.
        """
        self.amplitudes[PerformanceState.CEILING] += factor
        self.amplitudes[PerformanceState.BUST] += factor
        self.amplitudes[PerformanceState.BASELINE] -= factor * 2
        self.amplitudes[PerformanceState.BASELINE] = max(0.1, self.amplitudes[PerformanceState.BASELINE])
        self._normalize()
    
    def decrease_variance(self, factor: float = 0.1):
        """
        Decrease probability of extreme states.
        
        Used for consistent players or stable matchups.
        """
        self.amplitudes[PerformanceState.BASELINE] += factor * 2
        self.amplitudes[PerformanceState.CEILING] -= factor
        self.amplitudes[PerformanceState.BUST] -= factor
        
        # Ensure minimums
        for state in PerformanceState:
            self.amplitudes[state] = max(0.02, self.amplitudes[state])
        self._normalize()
    
    def collapse(self) -> PerformanceState:
        """
        Collapse the quantum state to a definite state.
        
        This is the "measurement" - returns a concrete performance level
        based on the probability distribution.
        """
        states = list(self.amplitudes.keys())
        probs = list(self.amplitudes.values())
        return np.random.choice(states, p=probs)
    
    def get_expected_multiplier(self) -> float:
        """Get expected performance multiplier based on state distribution."""
        return sum(
            state.multiplier * prob 
            for state, prob in self.amplitudes.items()
        )
    
    def get_stat_distribution(self, stat_avg: float, stat_std: float) -> Tuple[float, float]:
        """
        Get expected value and std for a stat based on quantum state.
        
        Returns (expected_value, uncertainty)
        """
        expected = stat_avg * self.get_expected_multiplier()
        
        # Uncertainty increases with extreme state probabilities
        extreme_prob = (self.amplitudes[PerformanceState.CEILING] + 
                       self.amplitudes[PerformanceState.BUST])
        uncertainty = stat_std * (1 + extreme_prob)
        
        return expected, uncertainty


def create_player_state(
    player: PlayerProfile,
    is_home: bool = True,
    momentum: float = 0.0,
    matchup_modifier: float = 1.0,
    minutes_factor: float = 1.0,
) -> PlayerQuantumState:
    """
    Create a context-adjusted quantum state for a player.
    
    Parameters
    ----------
    player : PlayerProfile
        The player's statistical profile
    is_home : bool
        Whether playing at home
    momentum : float
        Recent form (-1 to 1, positive = hot)
    matchup_modifier : float
        Opponent defense quality (>1 = bad D, <1 = good D)
    minutes_factor : float
        Expected minutes vs average (>1 = more minutes)
    
    Returns
    -------
    PlayerQuantumState with adjusted probabilities
    """
    state = PlayerQuantumState(player)
    
    # Adjust for home court
    if is_home:
        state.shift_toward(PerformanceState.ELEVATED, 0.05)
    else:
        state.shift_toward(PerformanceState.FLOOR, 0.03)
    
    # Adjust for momentum (hot/cold streaks)
    if momentum > 0.1:
        state.shift_toward(PerformanceState.ELEVATED, momentum * 0.15)
    elif momentum < -0.1:
        state.shift_toward(PerformanceState.FLOOR, abs(momentum) * 0.15)
    
    # Adjust for matchup
    if matchup_modifier > 1.1:  # Bad defense
        state.shift_toward(PerformanceState.ELEVATED, 0.08)
    elif matchup_modifier < 0.9:  # Good defense
        state.shift_toward(PerformanceState.FLOOR, 0.08)
    
    # Adjust for minutes
    if minutes_factor > 1.1:  # More minutes than usual
        state.shift_toward(PerformanceState.ELEVATED, 0.05)
    elif minutes_factor < 0.9:  # Fewer minutes
        state.shift_toward(PerformanceState.FLOOR, 0.05)
    
    # Adjust variance based on player consistency
    if player.pts_cv > 0.4:  # High variance player
        state.increase_variance(0.08)
    elif player.pts_cv < 0.25:  # Consistent player
        state.decrease_variance(0.08)
    
    return state


def simulate_player_game(
    state: PlayerQuantumState,
    stat: str = 'pts',
    n_sims: int = 10000,
) -> Dict:
    """
    Simulate a player's performance using quantum state.
    
    Parameters
    ----------
    state : PlayerQuantumState
        The player's quantum state
    stat : str
        Which stat to simulate ('pts', 'reb', 'ast', 'pra', 'fg3m')
    n_sims : int
        Number of Monte Carlo simulations
    
    Returns
    -------
    Dict with mean, std, percentiles, over/under probabilities
    """
    player = state.player
    
    # Get base stats
    stat_map = {
        'pts': (player.pts_avg, player.pts_std),
        'reb': (player.reb_avg, player.reb_std),
        'ast': (player.ast_avg, player.ast_std),
        'fg3m': (player.fg3m_avg, player.fg3m_std),
        'pra': (player.pra_avg, np.sqrt(player.pts_std**2 + player.reb_std**2 + player.ast_std**2)),
    }
    
    base_avg, base_std = stat_map.get(stat, (player.pts_avg, player.pts_std))
    
    # Run simulations
    results = []
    for _ in range(n_sims):
        # Collapse quantum state
        perf_state = state.collapse()
        multiplier = perf_state.multiplier
        
        # Generate stat with noise
        mean = base_avg * multiplier
        std = base_std * (0.8 + 0.4 * np.random.random())  # Some variance in variance
        
        value = np.random.normal(mean, std)
        value = max(0, value)  # Can't have negative stats
        
        results.append(value)
    
    results = np.array(results)
    
    return {
        'mean': np.mean(results),
        'std': np.std(results),
        'median': np.median(results),
        'p10': np.percentile(results, 10),
        'p25': np.percentile(results, 25),
        'p75': np.percentile(results, 75),
        'p90': np.percentile(results, 90),
        'min': np.min(results),
        'max': np.max(results),
        'distribution': results,
    }
