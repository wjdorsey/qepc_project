"""
QEPC Quantum Core
=================
The heart of the Quantum Enhanced Prediction Calculator.

This module implements quantum-inspired mechanics for sports prediction:
- SUPERPOSITION: Teams exist in multiple performance states simultaneously
- ENTANGLEMENT: Team performances are correlated (pace, game flow)
- INTERFERENCE: Matchup effects amplify or cancel
- TUNNELING: Upset probability floor (any given Sunday)
- DECOHERENCE: Environmental factors reduce quantum effects

These aren't just labels - they're mathematical models that capture
the inherent uncertainty in sports better than traditional approaches.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum


class PerformanceState(Enum):
    """
    Quantum-inspired performance states.
    A team exists in superposition of these until game time.
    """
    DOMINANT = "dominant"      # 90th percentile performance
    ELEVATED = "elevated"      # 75th percentile
    BASELINE = "baseline"      # 50th percentile (expected)
    DIMINISHED = "diminished"  # 25th percentile
    STRUGGLING = "struggling"  # 10th percentile


@dataclass
class QuantumState:
    """
    Represents a team's quantum state - a superposition of possible performances.
    
    The amplitudes determine probability of collapsing into each state.
    |amplitude|² = probability (Born rule from quantum mechanics)
    """
    amplitudes: Dict[PerformanceState, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize with default amplitudes if not provided."""
        if not self.amplitudes:
            # Default: peaked at baseline with tails
            self.amplitudes = {
                PerformanceState.DOMINANT: 0.10,
                PerformanceState.ELEVATED: 0.20,
                PerformanceState.BASELINE: 0.40,
                PerformanceState.DIMINISHED: 0.20,
                PerformanceState.STRUGGLING: 0.10,
            }
        self._normalize()
    
    def _normalize(self):
        """Ensure amplitudes sum to 1 (valid probability distribution)."""
        total = sum(self.amplitudes.values())
        if total > 0:
            self.amplitudes = {k: v/total for k, v in self.amplitudes.items()}
    
    def collapse(self) -> PerformanceState:
        """
        Collapse the superposition into a definite state.
        This is called at simulation time - the "measurement" moment.
        """
        states = list(self.amplitudes.keys())
        probs = list(self.amplitudes.values())
        return np.random.choice(states, p=probs)
    
    def get_multiplier(self, state: PerformanceState) -> float:
        """Get the performance multiplier for a given state."""
        multipliers = {
            PerformanceState.DOMINANT: 1.12,    # +12% performance
            PerformanceState.ELEVATED: 1.05,    # +5%
            PerformanceState.BASELINE: 1.00,    # Expected
            PerformanceState.DIMINISHED: 0.95,  # -5%
            PerformanceState.STRUGGLING: 0.88,  # -12%
        }
        return multipliers[state]
    
    def shift_toward(self, target_state: PerformanceState, strength: float = 0.2):
        """
        Shift probability mass toward a target state.
        Used for situational adjustments (home court, rest, etc.)
        """
        current = self.amplitudes[target_state]
        boost = strength * (1 - current)  # Diminishing returns
        
        # Add to target, subtract proportionally from others
        for state in self.amplitudes:
            if state == target_state:
                self.amplitudes[state] += boost
            else:
                self.amplitudes[state] *= (1 - boost / (1 - current + 0.001))
        
        self._normalize()
    
    def increase_variance(self, factor: float = 1.5):
        """
        Increase variance (more extreme outcomes likely).
        Used for volatile teams or high-stakes games.
        """
        # Move mass from center to tails
        baseline_mass = self.amplitudes[PerformanceState.BASELINE]
        transfer = baseline_mass * (1 - 1/factor) * 0.5
        
        self.amplitudes[PerformanceState.BASELINE] -= transfer * 2
        self.amplitudes[PerformanceState.DOMINANT] += transfer
        self.amplitudes[PerformanceState.STRUGGLING] += transfer
        
        self._normalize()
    
    def decrease_variance(self, factor: float = 1.5):
        """
        Decrease variance (outcomes closer to expected).
        Used for consistent teams or grinding games.
        """
        # Move mass from tails to center
        for extreme in [PerformanceState.DOMINANT, PerformanceState.STRUGGLING]:
            transfer = self.amplitudes[extreme] * (1 - 1/factor)
            self.amplitudes[extreme] -= transfer
            self.amplitudes[PerformanceState.BASELINE] += transfer
        
        self._normalize()


@dataclass 
class QuantumConfig:
    """Configuration for quantum mechanics parameters."""
    
    # Entanglement: How correlated are team performances?
    # Higher = when one team plays well, other tends to play poorly
    entanglement_strength: float = 0.35
    
    # Tunneling: Minimum upset probability (any given Sunday)
    # Even 20-point favorites can lose sometimes
    tunneling_base_rate: float = 0.08
    tunneling_min_prob: float = 0.02
    
    # Interference: How much do matchups amplify/cancel?
    interference_factor: float = 0.15
    
    # Decoherence: Environmental factors that reduce quantum effects
    weather_decoherence: float = 0.10  # Bad weather = more random
    primetime_variance_boost: float = 0.05  # Big games = more variance
    
    # Home court advantage
    base_hca: float = 3.0  # Points
    
    # Rest advantages (points per day of rest difference)
    rest_advantage_per_day: float = 1.2
    max_rest_advantage: float = 4.0
    b2b_penalty: float = 2.5  # Points off for back-to-back
    
    # Simulation settings
    default_simulations: int = 20000


class EntanglementEngine:
    """
    Models the entanglement between team performances.
    
    In quantum mechanics, entangled particles have correlated states.
    In basketball, team performances are correlated through:
    - Pace: If one team speeds up, both score more
    - Game flow: Blowouts vs close games affect both teams
    - Possessions: More turnovers = more possessions for opponent
    """
    
    def __init__(self, strength: float = 0.35):
        self.strength = strength
    
    def generate_correlated_performance(
        self, 
        n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate correlated performance factors for two teams.
        
        Returns arrays of multipliers that are negatively correlated:
        When one team overperforms, the other tends to underperform.
        """
        # Correlation matrix: negative correlation
        corr_matrix = np.array([
            [1.0, -self.strength],
            [-self.strength, 1.0]
        ])
        
        # Generate correlated normal samples
        mean = [0, 0]
        samples = np.random.multivariate_normal(mean, corr_matrix, n_samples)
        
        # Convert to multipliers (centered at 1.0)
        # Using smaller std to keep multipliers reasonable
        std = 0.08
        home_mult = 1.0 + samples[:, 0] * std
        away_mult = 1.0 + samples[:, 1] * std
        
        return home_mult, away_mult
    
    def apply_pace_entanglement(
        self,
        home_pace: float,
        away_pace: float,
        league_avg_pace: float = 100.0
    ) -> float:
        """
        Calculate game pace considering both teams.
        Pace is inherently entangled - both teams play at same pace.
        """
        # Weighted average with slight home court influence
        raw_pace = (home_pace * 0.52 + away_pace * 0.48)
        
        # Regress toward league average slightly
        regression = 0.15
        game_pace = raw_pace * (1 - regression) + league_avg_pace * regression
        
        return game_pace


class InterferenceCalculator:
    """
    Models constructive and destructive interference in matchups.
    
    Like wave interference:
    - Constructive: Great offense vs bad defense = amplified scoring
    - Destructive: Great offense vs great defense = effects cancel
    """
    
    def __init__(self, factor: float = 0.15):
        self.factor = factor
    
    def calculate_matchup_modifier(
        self,
        offense_rating: float,
        defense_rating: float,  # Lower is better
        league_avg_ortg: float = 114.0,
        league_avg_drtg: float = 114.0
    ) -> float:
        """
        Calculate interference-adjusted scoring modifier.
        
        Returns a multiplier > 1 for favorable matchups, < 1 for tough matchups.
        """
        # Normalize ratings relative to league average
        off_strength = (offense_rating - league_avg_ortg) / league_avg_ortg
        def_weakness = (defense_rating - league_avg_drtg) / league_avg_drtg  # Higher = weaker defense
        
        # Interference: Multiply strengths
        # Good offense + Bad defense = constructive (amplified)
        # Good offense + Good defense = destructive (cancelled)
        interference = off_strength * def_weakness
        
        # Convert to multiplier
        modifier = 1.0 + (interference * self.factor * 10)
        
        # Clamp to reasonable range
        return np.clip(modifier, 0.85, 1.15)


class TunnelingModel:
    """
    Models quantum tunneling - the ability to overcome "impossible" barriers.
    
    In sports: Underdogs can win even when they "shouldn't".
    This creates fat tails in the outcome distribution.
    """
    
    def __init__(self, base_rate: float = 0.08, min_prob: float = 0.02):
        self.base_rate = base_rate
        self.min_prob = min_prob
    
    def calculate_upset_floor(self, strength_gap: float) -> float:
        """
        Calculate minimum win probability for underdog.
        
        strength_gap: Positive means underdog is weaker (e.g., 10 point spread)
        
        Returns minimum probability the underdog still has.
        """
        # Exponential decay but with a floor
        # Even 20-point underdogs have ~2% chance
        decay_rate = 0.08
        upset_prob = self.base_rate * np.exp(-strength_gap * decay_rate)
        
        return max(upset_prob, self.min_prob)
    
    def apply_tunneling(
        self,
        base_win_prob: float,
        is_underdog: bool,
        strength_gap: float
    ) -> float:
        """
        Adjust win probability to ensure minimum upset chance.
        """
        if not is_underdog:
            # Favorite: Cap their probability based on underdog's floor
            underdog_floor = self.calculate_upset_floor(strength_gap)
            return min(base_win_prob, 1.0 - underdog_floor)
        else:
            # Underdog: Ensure they have minimum probability
            upset_floor = self.calculate_upset_floor(strength_gap)
            return max(base_win_prob, upset_floor)


class QuantumSimulator:
    """
    Main quantum simulation engine.
    
    Combines all quantum effects to simulate game outcomes.
    """
    
    def __init__(self, config: QuantumConfig = None):
        self.config = config or QuantumConfig()
        self.entanglement = EntanglementEngine(self.config.entanglement_strength)
        self.interference = InterferenceCalculator(self.config.interference_factor)
        self.tunneling = TunnelingModel(
            self.config.tunneling_base_rate,
            self.config.tunneling_min_prob
        )
    
    def simulate_game(
        self,
        home_expected: float,
        away_expected: float,
        home_std: float,
        away_std: float,
        home_state: QuantumState = None,
        away_state: QuantumState = None,
        n_sims: int = None
    ) -> Dict:
        """
        Run full quantum simulation of a game.
        
        Parameters
        ----------
        home_expected : float
            Expected points for home team
        away_expected : float
            Expected points for away team
        home_std : float
            Standard deviation of home team scoring
        away_std : float
            Standard deviation of away team scoring
        home_state : QuantumState, optional
            Quantum state for home team
        away_state : QuantumState, optional
            Quantum state for away team
        n_sims : int, optional
            Number of simulations
            
        Returns
        -------
        dict with simulation results
        """
        n_sims = n_sims or self.config.default_simulations
        
        # Initialize quantum states if not provided
        home_state = home_state or QuantumState()
        away_state = away_state or QuantumState()
        
        # Generate entangled performance factors
        home_entangle, away_entangle = self.entanglement.generate_correlated_performance(n_sims)
        
        # Arrays to store results
        home_scores = np.zeros(n_sims)
        away_scores = np.zeros(n_sims)
        
        for i in range(n_sims):
            # Collapse quantum states
            home_perf_state = home_state.collapse()
            away_perf_state = away_state.collapse()
            
            # Get performance multipliers
            home_mult = home_state.get_multiplier(home_perf_state)
            away_mult = away_state.get_multiplier(away_perf_state)
            
            # Apply entanglement
            home_mult *= home_entangle[i]
            away_mult *= away_entangle[i]
            
            # Generate scores
            home_base = np.random.normal(home_expected, home_std)
            away_base = np.random.normal(away_expected, away_std)
            
            home_scores[i] = max(70, home_base * home_mult)
            away_scores[i] = max(70, away_base * away_mult)
        
        # Handle ties (OT)
        ties = np.abs(home_scores - away_scores) < 1.0
        if np.any(ties):
            ot_points = np.random.normal(10, 3, np.sum(ties))
            home_ot_advantage = np.random.random(np.sum(ties)) < 0.52  # Slight home edge in OT
            home_scores[ties] += np.where(home_ot_advantage, ot_points * 0.55, ot_points * 0.45)
            away_scores[ties] += np.where(home_ot_advantage, ot_points * 0.45, ot_points * 0.55)
        
        # Calculate results
        home_wins = home_scores > away_scores
        home_win_prob = np.mean(home_wins)
        
        # Apply tunneling (ensure minimum upset probability)
        spread = home_expected - away_expected
        if spread > 0:  # Home favored
            home_win_prob = self.tunneling.apply_tunneling(home_win_prob, False, spread)
        else:  # Away favored
            home_win_prob = self.tunneling.apply_tunneling(home_win_prob, True, -spread)
        
        return {
            'home_win_prob': home_win_prob,
            'away_win_prob': 1 - home_win_prob,
            'home_score_mean': np.mean(home_scores),
            'away_score_mean': np.mean(away_scores),
            'home_score_std': np.std(home_scores),
            'away_score_std': np.std(away_scores),
            'predicted_spread': np.mean(home_scores - away_scores),
            'predicted_total': np.mean(home_scores + away_scores),
            'spread_std': np.std(home_scores - away_scores),
            'total_std': np.std(home_scores + away_scores),
            'home_scores': home_scores,
            'away_scores': away_scores,
        }


# Convenience functions
def create_quantum_state_from_volatility(volatility: float) -> QuantumState:
    """
    Create a quantum state based on team volatility.
    
    High volatility = wider distribution (more extreme outcomes)
    Low volatility = peaked at baseline (consistent team)
    """
    state = QuantumState()
    
    # volatility is coefficient of variation (std/mean), typically 0.05-0.15
    if volatility > 0.12:  # High volatility
        state.increase_variance(1.5)
    elif volatility < 0.08:  # Low volatility
        state.decrease_variance(1.3)
    
    return state


def adjust_state_for_situation(
    state: QuantumState,
    is_home: bool = False,
    rest_days: int = 2,
    is_b2b: bool = False,
    momentum: float = 0.0  # -1 to 1, negative = cold streak
) -> QuantumState:
    """
    Adjust quantum state for situational factors.
    """
    # Home court shifts toward better performance
    if is_home:
        state.shift_toward(PerformanceState.ELEVATED, strength=0.1)
    
    # Rest advantage
    if rest_days >= 3:
        state.shift_toward(PerformanceState.ELEVATED, strength=0.08)
    elif is_b2b:
        state.shift_toward(PerformanceState.DIMINISHED, strength=0.12)
    
    # Momentum
    if momentum > 0.3:  # Hot streak
        state.shift_toward(PerformanceState.ELEVATED, strength=momentum * 0.15)
    elif momentum < -0.3:  # Cold streak
        state.shift_toward(PerformanceState.DIMINISHED, strength=-momentum * 0.15)
    
    return state
