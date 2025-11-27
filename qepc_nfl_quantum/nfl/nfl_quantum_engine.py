"""
QEPC NFL Quantum Engine v1.0
============================

A quantum-inspired NFL prediction model that goes beyond basic Poisson.

QUANTUM CONCEPTS APPLIED:
-------------------------

1. SUPERPOSITION
   Teams exist in multiple "states" simultaneously until gametime.
   We model each team as a weighted mixture of archetypes:
   - DOMINANT: Playing at peak level
   - BASELINE: Normal performance  
   - STRUGGLING: Underperforming
   The weights collapse based on situational factors.

2. ENTANGLEMENT
   - Offensive and defensive performances are correlated
   - When one team's offense excels, the other's defense degrades
   - Pace entanglement: both teams' tempos interact
   - Turnover entanglement: turnovers affect both sides

3. INTERFERENCE PATTERNS
   - Constructive: Strength meets weakness (amplified effect)
   - Destructive: Strength meets strength (canceled out)
   - Models non-linear matchup effects

4. QUANTUM TUNNELING
   - Models upset probability as "tunneling through" expected barriers
   - Heavy underdogs have non-zero upset probability regardless of spread
   - Fat tails in the distribution

5. DECOHERENCE / ENVIRONMENTAL FACTORS
   - Weather "decoheres" passing game effectiveness
   - Altitude, turf type, noise levels
   - Prime time / pressure games

6. MOMENTUM WAVES
   - Teams on streaks have momentum wave functions
   - Can constructively or destructively interfere with opponent's momentum

NFL-SPECIFIC SCORING:
--------------------
Unlike NBA (continuous), NFL scoring is discrete:
- Touchdowns: 6 pts (+ PAT/2pt)
- Field Goals: 3 pts
- Safeties: 2 pts

We model DRIVES and their outcomes, not just total points.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
import warnings


# =============================================================================
# QUANTUM STATE DEFINITIONS
# =============================================================================

class TeamState(Enum):
    """Possible quantum states a team can collapse into."""
    DOMINANT = "dominant"      # Playing at 90th percentile
    ELEVATED = "elevated"      # Playing above average
    BASELINE = "baseline"      # Normal performance
    DIMINISHED = "diminished"  # Below average
    STRUGGLING = "struggling"  # Playing at 10th percentile


@dataclass
class QuantumState:
    """
    Represents a team's superposition of possible states.
    
    The team exists in ALL states simultaneously with different amplitudes.
    When the game is "observed" (simulated), the state collapses.
    """
    # Amplitude squared = probability of each state
    amplitudes: Dict[TeamState, float] = field(default_factory=dict)
    
    # Performance multipliers for each state
    multipliers: Dict[TeamState, float] = field(default_factory=lambda: {
        TeamState.DOMINANT: 1.25,
        TeamState.ELEVATED: 1.10,
        TeamState.BASELINE: 1.00,
        TeamState.DIMINISHED: 0.90,
        TeamState.STRUGGLING: 0.75,
    })
    
    def __post_init__(self):
        if not self.amplitudes:
            # Default: mostly baseline with tails
            self.amplitudes = {
                TeamState.DOMINANT: 0.10,
                TeamState.ELEVATED: 0.20,
                TeamState.BASELINE: 0.40,
                TeamState.DIMINISHED: 0.20,
                TeamState.STRUGGLING: 0.10,
            }
    
    def collapse(self) -> Tuple[TeamState, float]:
        """
        Collapse the superposition into a definite state.
        Returns (state, multiplier).
        """
        states = list(self.amplitudes.keys())
        probs = np.array([self.amplitudes[s] for s in states])
        probs = probs / probs.sum()  # Normalize
        
        chosen_state = np.random.choice(states, p=probs)
        return chosen_state, self.multipliers[chosen_state]
    
    def shift_toward(self, target_state: TeamState, strength: float = 0.2):
        """
        Shift probability amplitude toward a target state.
        Used for situational adjustments (home field, rest, etc.)
        """
        # Increase target amplitude
        current = self.amplitudes.get(target_state, 0.1)
        boost = strength * (1 - current)
        self.amplitudes[target_state] = current + boost
        
        # Reduce other amplitudes proportionally
        for state in self.amplitudes:
            if state != target_state:
                self.amplitudes[state] *= (1 - strength)
        
        # Renormalize
        total = sum(self.amplitudes.values())
        for state in self.amplitudes:
            self.amplitudes[state] /= total


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class NFLQuantumConfig:
    """Configuration for the quantum NFL model."""
    
    # === DRIVE OUTCOMES ===
    # Probability distribution of drive outcomes (baseline)
    DRIVE_OUTCOMES = {
        'touchdown': 0.22,      # ~22% of drives end in TD
        'field_goal': 0.18,     # ~18% end in FG
        'punt': 0.38,           # ~38% end in punt
        'turnover': 0.12,       # ~12% end in turnover
        'turnover_td': 0.02,    # ~2% defensive/ST TD
        'end_of_half': 0.06,    # ~6% time expires
        'safety': 0.01,         # ~1% safety
        'missed_fg': 0.01,      # ~1% missed FG
    }
    
    # Points per outcome
    POINTS = {
        'touchdown': 6.95,      # TD + ~95% PAT rate
        'field_goal': 3.0,
        'safety': 2.0,
        'turnover_td': 6.95,    # Defensive TD
    }
    
    # === DRIVES PER GAME ===
    AVG_DRIVES_PER_TEAM = 11.5  # Average drives per team per game
    DRIVES_STD = 1.5
    
    # === HOME FIELD ===
    HOME_FIELD_ADVANTAGE = 0.03  # 3% boost to home team
    
    # === QUANTUM PARAMETERS ===
    ENTANGLEMENT_STRENGTH = 0.35   # How much performances are correlated
    TUNNELING_PROBABILITY = 0.08   # Base upset probability floor
    INTERFERENCE_FACTOR = 0.20     # Strength of constructive/destructive interference
    
    # === MOMENTUM ===
    MOMENTUM_DECAY = 0.85          # Momentum carries 85% week to week
    MOMENTUM_IMPACT = 0.05         # Max 5% shift from momentum
    
    # === WEATHER DECOHERENCE ===
    WEATHER_IMPACT = {
        'dome': 1.0,           # No effect
        'clear': 1.0,
        'rain': 0.85,          # 15% reduction to passing
        'snow': 0.75,          # 25% reduction
        'wind': 0.80,          # 20% reduction (wind > 15mph)
        'extreme_cold': 0.90,  # 10% reduction
    }


# =============================================================================
# TEAM STRENGTH MODEL
# =============================================================================

@dataclass
class NFLTeamStrength:
    """
    Team strength ratings for NFL.
    
    Unlike NBA, we separate:
    - Offensive efficiency (points per drive)
    - Defensive efficiency (points allowed per drive)
    - Special teams
    - Turnover tendency
    """
    team: str
    
    # Offensive metrics (higher = better)
    off_efficiency: float = 1.0       # Points per drive vs league avg
    off_explosiveness: float = 1.0    # Big play rate
    off_consistency: float = 0.5      # Variance (lower = more consistent)
    pass_tendency: float = 0.55       # Pass rate
    red_zone_efficiency: float = 0.55 # TD% in red zone
    
    # Defensive metrics (lower = better for efficiency)
    def_efficiency: float = 1.0       # Points allowed per drive vs league avg
    def_explosiveness: float = 1.0    # Big plays allowed
    def_consistency: float = 0.5
    pass_rush: float = 1.0            # Pressure rate
    coverage: float = 1.0             # Coverage grade
    
    # Special teams
    st_efficiency: float = 1.0        # Field position impact
    
    # Turnover metrics
    turnover_rate: float = 0.12       # Giveaway rate
    takeaway_rate: float = 0.12       # Takeaway rate
    
    # Situational
    home_boost: float = 0.03          # Additional home advantage
    dome_team: bool = False           # Plays in dome
    
    # Momentum/Form (updated weekly)
    momentum: float = 0.0             # -1 to +1 scale
    
    # Quantum state distribution
    quantum_state: QuantumState = field(default_factory=QuantumState)
    
    def calculate_volatility(self) -> float:
        """Calculate team's overall volatility/unpredictability."""
        return (self.off_consistency + self.def_consistency) / 2


# =============================================================================
# INTERFERENCE CALCULATOR
# =============================================================================

class InterferenceCalculator:
    """
    Calculates quantum interference effects between team matchups.
    
    When strengths meet weaknesses: CONSTRUCTIVE interference (amplified)
    When strengths meet strengths: DESTRUCTIVE interference (canceled)
    """
    
    @staticmethod
    def calculate(
        team_a_strength: float,
        team_b_weakness: float,  # Opponent's weakness in that area
        interference_factor: float = 0.2
    ) -> float:
        """
        Calculate interference multiplier.
        
        Returns value typically between 0.8 and 1.2
        """
        # Difference between strength and opponent's ability to counter
        delta = team_a_strength - team_b_weakness
        
        # Sigmoid-like transformation for smooth interference
        interference = 1 + interference_factor * np.tanh(delta)
        
        return interference
    
    @staticmethod
    def entangle_performances(
        off_state: float,
        def_state: float,
        entanglement: float = 0.35
    ) -> Tuple[float, float]:
        """
        Entangle offensive and defensive performances.
        
        When offense overperforms, defense tends to underperform (and vice versa)
        due to game flow, time of possession, etc.
        """
        # Create correlated random adjustments
        correlation_matrix = np.array([
            [1.0, -entanglement],
            [-entanglement, 1.0]
        ])
        
        # Generate correlated random factors
        mean = [0, 0]
        adjustments = np.random.multivariate_normal(mean, correlation_matrix * 0.1)
        
        return off_state + adjustments[0], def_state + adjustments[1]


# =============================================================================
# TUNNELING MODEL (UPSETS)
# =============================================================================

class TunnelingModel:
    """
    Models upset probability using quantum tunneling analogy.
    
    Even when a team is heavily favored, there's a non-zero probability
    the underdog "tunnels through" the barrier and wins.
    """
    
    @staticmethod
    def upset_probability(
        favorite_strength: float,
        underdog_strength: float,
        base_tunneling: float = 0.08
    ) -> float:
        """
        Calculate upset probability floor.
        
        The larger the gap, the lower the tunneling probability,
        but it never goes to zero.
        """
        strength_gap = favorite_strength - underdog_strength
        
        # Exponential decay but with floor
        tunneling_prob = base_tunneling * np.exp(-strength_gap * 0.5)
        
        # Floor at 2% for any game (any given Sunday!)
        return max(0.02, tunneling_prob)
    
    @staticmethod
    def apply_tunneling(
        favorite_win_prob: float,
        tunneling_prob: float
    ) -> float:
        """
        Adjust win probability to account for tunneling.
        
        Ensures underdog always has at least tunneling_prob chance.
        """
        underdog_prob = 1 - favorite_win_prob
        
        if underdog_prob < tunneling_prob:
            return 1 - tunneling_prob
        
        return favorite_win_prob


# =============================================================================
# DRIVE SIMULATOR
# =============================================================================

class DriveSimulator:
    """
    Simulates individual drives rather than just total points.
    
    This captures the discrete nature of NFL scoring better than
    continuous distributions.
    """
    
    def __init__(self, config: NFLQuantumConfig = None):
        self.config = config or NFLQuantumConfig()
    
    def simulate_drive(
        self,
        off_team: NFLTeamStrength,
        def_team: NFLTeamStrength,
        off_multiplier: float = 1.0,
        def_multiplier: float = 1.0,
        field_position: float = 0.25,  # 0-1 scale, 0 = own goal line
    ) -> Tuple[str, float]:
        """
        Simulate a single drive.
        
        Returns (outcome, points_scored)
        """
        # Adjust base probabilities
        probs = self.config.DRIVE_OUTCOMES.copy()
        
        # Offensive efficiency effect
        off_factor = off_team.off_efficiency * off_multiplier
        def_factor = def_team.def_efficiency * def_multiplier
        
        # Net efficiency
        net_efficiency = off_factor / def_factor
        
        # Adjust TD and FG probability
        probs['touchdown'] *= net_efficiency
        probs['field_goal'] *= net_efficiency ** 0.5  # Less affected
        
        # Adjust turnover probability
        turnover_factor = off_team.turnover_rate / def_team.takeaway_rate
        probs['turnover'] *= turnover_factor
        probs['turnover_td'] *= (1 / turnover_factor)  # Defensive TD
        
        # Field position effect
        if field_position > 0.6:  # Already in FG range
            probs['field_goal'] *= 1.3
            probs['touchdown'] *= 1.2
            probs['punt'] *= 0.5
        elif field_position < 0.2:  # Backed up
            probs['safety'] *= 2.0
            probs['punt'] *= 1.2
        
        # Normalize
        total = sum(probs.values())
        probs = {k: v/total for k, v in probs.items()}
        
        # Sample outcome
        outcomes = list(probs.keys())
        probabilities = list(probs.values())
        outcome = np.random.choice(outcomes, p=probabilities)
        
        # Calculate points
        points = self.config.POINTS.get(outcome, 0)
        
        # For TDs, simulate 2pt vs PAT decision
        if outcome == 'touchdown':
            if np.random.random() < 0.05:  # 5% go for 2
                points = 6 + (2 if np.random.random() < 0.48 else 0)
            else:
                points = 6 + (1 if np.random.random() < 0.94 else 0)
        
        return outcome, points
    
    def simulate_game_drives(
        self,
        home_team: NFLTeamStrength,
        away_team: NFLTeamStrength,
        home_multiplier: float = 1.0,
        away_multiplier: float = 1.0,
        weather: str = 'clear'
    ) -> Dict:
        """
        Simulate all drives in a game.
        
        Returns detailed drive log and final scores.
        """
        # Weather decoherence
        weather_factor = self.config.WEATHER_IMPACT.get(weather, 1.0)
        
        # Number of drives (with some randomness)
        home_drives = int(np.random.normal(
            self.config.AVG_DRIVES_PER_TEAM, 
            self.config.DRIVES_STD
        ))
        away_drives = int(np.random.normal(
            self.config.AVG_DRIVES_PER_TEAM,
            self.config.DRIVES_STD
        ))
        
        home_drives = max(8, min(15, home_drives))
        away_drives = max(8, min(15, away_drives))
        
        # Simulate drives
        home_score = 0
        away_score = 0
        drive_log = []
        
        # Alternate possessions (simplified)
        total_drives = home_drives + away_drives
        home_possession = np.random.random() < 0.5  # Coin flip for first possession
        
        for i in range(total_drives):
            if home_possession and home_drives > 0:
                outcome, points = self.simulate_drive(
                    home_team, away_team,
                    home_multiplier * weather_factor,
                    away_multiplier
                )
                home_score += points
                if outcome == 'turnover_td':
                    away_score += points
                    home_score -= points
                drive_log.append(('home', outcome, points))
                home_drives -= 1
            elif away_drives > 0:
                outcome, points = self.simulate_drive(
                    away_team, home_team,
                    away_multiplier * weather_factor,
                    home_multiplier
                )
                away_score += points
                if outcome == 'turnover_td':
                    home_score += points
                    away_score -= points
                drive_log.append(('away', outcome, points))
                away_drives -= 1
            
            home_possession = not home_possession
        
        return {
            'home_score': int(home_score),
            'away_score': int(away_score),
            'home_drives': len([d for d in drive_log if d[0] == 'home']),
            'away_drives': len([d for d in drive_log if d[0] == 'away']),
            'drive_log': drive_log,
        }


# =============================================================================
# MAIN QUANTUM ENGINE
# =============================================================================

class NFLQuantumEngine:
    """
    Main prediction engine combining all quantum elements.
    
    Usage:
        engine = NFLQuantumEngine()
        engine.load_team_strengths(df)
        
        result = engine.predict_game(
            home_team="Kansas City Chiefs",
            away_team="Buffalo Bills",
            weather="clear"
        )
    """
    
    def __init__(self, config: NFLQuantumConfig = None):
        self.config = config or NFLQuantumConfig()
        self.teams: Dict[str, NFLTeamStrength] = {}
        self.drive_sim = DriveSimulator(self.config)
        self.interference = InterferenceCalculator()
        self.tunneling = TunnelingModel()
    
    def add_team(self, strength: NFLTeamStrength):
        """Add a team to the engine."""
        self.teams[strength.team] = strength
    
    def load_team_strengths(self, df: pd.DataFrame):
        """
        Load team strengths from a DataFrame.
        
        Expected columns:
        - team
        - off_efficiency, def_efficiency
        - turnover_rate, takeaway_rate
        - (optional) momentum, home_boost, etc.
        """
        for _, row in df.iterrows():
            strength = NFLTeamStrength(
                team=row['team'],
                off_efficiency=row.get('off_efficiency', 1.0),
                def_efficiency=row.get('def_efficiency', 1.0),
                off_explosiveness=row.get('off_explosiveness', 1.0),
                def_explosiveness=row.get('def_explosiveness', 1.0),
                turnover_rate=row.get('turnover_rate', 0.12),
                takeaway_rate=row.get('takeaway_rate', 0.12),
                momentum=row.get('momentum', 0.0),
                home_boost=row.get('home_boost', 0.03),
            )
            
            # Set quantum state based on consistency
            volatility = row.get('volatility', 0.5)
            if volatility > 0.6:  # High variance team
                strength.quantum_state = QuantumState(amplitudes={
                    TeamState.DOMINANT: 0.15,
                    TeamState.ELEVATED: 0.15,
                    TeamState.BASELINE: 0.30,
                    TeamState.DIMINISHED: 0.20,
                    TeamState.STRUGGLING: 0.20,
                })
            elif volatility < 0.3:  # Consistent team
                strength.quantum_state = QuantumState(amplitudes={
                    TeamState.DOMINANT: 0.05,
                    TeamState.ELEVATED: 0.25,
                    TeamState.BASELINE: 0.45,
                    TeamState.DIMINISHED: 0.20,
                    TeamState.STRUGGLING: 0.05,
                })
            
            self.teams[row['team']] = strength
    
    def _apply_situational_shifts(
        self,
        team: NFLTeamStrength,
        is_home: bool,
        weather: str,
        primetime: bool = False,
    ) -> QuantumState:
        """
        Apply situational factors to shift quantum state.
        """
        state = QuantumState(amplitudes=team.quantum_state.amplitudes.copy())
        
        # Home field shifts toward elevated/dominant
        if is_home:
            state.shift_toward(TeamState.ELEVATED, strength=0.15)
        
        # Bad weather shifts toward baseline (less variance)
        if weather in ['rain', 'snow', 'wind']:
            state.shift_toward(TeamState.BASELINE, strength=0.20)
        
        # Primetime can go either way - increases variance
        if primetime:
            state.amplitudes[TeamState.DOMINANT] *= 1.3
            state.amplitudes[TeamState.STRUGGLING] *= 1.3
            # Renormalize
            total = sum(state.amplitudes.values())
            for s in state.amplitudes:
                state.amplitudes[s] /= total
        
        # Momentum effect
        if team.momentum > 0.3:
            state.shift_toward(TeamState.ELEVATED, strength=team.momentum * 0.2)
        elif team.momentum < -0.3:
            state.shift_toward(TeamState.DIMINISHED, strength=abs(team.momentum) * 0.2)
        
        return state
    
    def simulate_game(
        self,
        home_team: str,
        away_team: str,
        weather: str = 'clear',
        primetime: bool = False,
        neutral_site: bool = False,
    ) -> Dict:
        """
        Simulate a single game with full quantum effects.
        """
        home = self.teams.get(home_team)
        away = self.teams.get(away_team)
        
        if home is None or away is None:
            raise ValueError(f"Team not found: {home_team if home is None else away_team}")
        
        # === PHASE 1: QUANTUM STATE COLLAPSE ===
        home_state = self._apply_situational_shifts(home, not neutral_site, weather, primetime)
        away_state = self._apply_situational_shifts(away, False, weather, primetime)
        
        home_collapsed, home_mult = home_state.collapse()
        away_collapsed, away_mult = away_state.collapse()
        
        # === PHASE 2: INTERFERENCE CALCULATION ===
        # Offensive vs defensive matchup interference
        home_off_interference = self.interference.calculate(
            home.off_efficiency,
            away.def_efficiency,
            self.config.INTERFERENCE_FACTOR
        )
        away_off_interference = self.interference.calculate(
            away.off_efficiency,
            home.def_efficiency,
            self.config.INTERFERENCE_FACTOR
        )
        
        # === PHASE 3: ENTANGLEMENT ===
        home_mult, away_mult = self.interference.entangle_performances(
            home_mult, away_mult, self.config.ENTANGLEMENT_STRENGTH
        )
        
        # Apply interference to multipliers
        home_mult *= home_off_interference
        away_mult *= away_off_interference
        
        # === PHASE 4: HOME FIELD (if not neutral) ===
        if not neutral_site:
            home_mult *= (1 + self.config.HOME_FIELD_ADVANTAGE + home.home_boost)
        
        # === PHASE 5: SIMULATE DRIVES ===
        result = self.drive_sim.simulate_game_drives(
            home, away, home_mult, away_mult, weather
        )
        
        # Add metadata
        result['home_team'] = home_team
        result['away_team'] = away_team
        result['home_state'] = home_collapsed.value
        result['away_state'] = away_collapsed.value
        result['home_multiplier'] = home_mult
        result['away_multiplier'] = away_mult
        result['weather'] = weather
        
        return result
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        n_simulations: int = 10000,
        weather: str = 'clear',
        primetime: bool = False,
        neutral_site: bool = False,
    ) -> Dict:
        """
        Run Monte Carlo simulation and return prediction.
        """
        home_wins = 0
        home_scores = []
        away_scores = []
        margins = []
        totals = []
        
        state_distribution = {s.value: 0 for s in TeamState}
        
        for _ in range(n_simulations):
            result = self.simulate_game(
                home_team, away_team, weather, primetime, neutral_site
            )
            
            home_scores.append(result['home_score'])
            away_scores.append(result['away_score'])
            margins.append(result['home_score'] - result['away_score'])
            totals.append(result['home_score'] + result['away_score'])
            
            if result['home_score'] > result['away_score']:
                home_wins += 1
            elif result['home_score'] == result['away_score']:
                home_wins += 0.5  # Ties count as half
            
            state_distribution[result['home_state']] += 1
        
        # Calculate base win probability
        home_win_prob = home_wins / n_simulations
        
        # === APPLY TUNNELING ===
        # Ensure underdog always has a chance
        home_strength = self.teams[home_team].off_efficiency / self.teams[home_team].def_efficiency
        away_strength = self.teams[away_team].off_efficiency / self.teams[away_team].def_efficiency
        
        if home_win_prob > 0.5:
            tunneling_prob = self.tunneling.upset_probability(home_strength, away_strength)
            home_win_prob = self.tunneling.apply_tunneling(home_win_prob, tunneling_prob)
        else:
            tunneling_prob = self.tunneling.upset_probability(away_strength, home_strength)
            away_win_prob = self.tunneling.apply_tunneling(1 - home_win_prob, tunneling_prob)
            home_win_prob = 1 - away_win_prob
        
        return {
            'home_team': home_team,
            'away_team': away_team,
            'home_win_prob': round(home_win_prob, 3),
            'away_win_prob': round(1 - home_win_prob, 3),
            'predicted_spread': round(np.mean(margins), 1),
            'spread_std': round(np.std(margins), 1),
            'predicted_total': round(np.mean(totals), 1),
            'total_std': round(np.std(totals), 1),
            'home_score_avg': round(np.mean(home_scores), 1),
            'away_score_avg': round(np.mean(away_scores), 1),
            'home_score_median': round(np.median(home_scores), 1),
            'away_score_median': round(np.median(away_scores), 1),
            'simulations': n_simulations,
            'state_distribution': {k: v/n_simulations for k, v in state_distribution.items()},
            'weather': weather,
            'primetime': primetime,
            'neutral_site': neutral_site,
        }
    
    def over_under_probability(
        self,
        home_team: str,
        away_team: str,
        line: float,
        n_simulations: int = 10000,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate over/under probability for a total line.
        """
        overs = 0
        
        for _ in range(n_simulations):
            result = self.simulate_game(home_team, away_team, **kwargs)
            total = result['home_score'] + result['away_score']
            if total > line:
                overs += 1
            elif total == line:
                overs += 0.5
        
        over_prob = overs / n_simulations
        return over_prob, 1 - over_prob
    
    def spread_probability(
        self,
        home_team: str,
        away_team: str,
        spread: float,  # Negative = home favored
        n_simulations: int = 10000,
        **kwargs
    ) -> Tuple[float, float]:
        """
        Calculate probability of covering a spread.
        
        spread = -7 means home favored by 7
        Home covers if they win by more than 7
        """
        home_covers = 0
        
        for _ in range(n_simulations):
            result = self.simulate_game(home_team, away_team, **kwargs)
            margin = result['home_score'] - result['away_score']
            
            # Home covers if margin > -spread (or margin + spread > 0)
            if margin + spread > 0:
                home_covers += 1
            elif margin + spread == 0:
                home_covers += 0.5
        
        home_cover_prob = home_covers / n_simulations
        return home_cover_prob, 1 - home_cover_prob


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_sample_teams() -> Dict[str, NFLTeamStrength]:
    """Create sample team strengths for testing."""
    teams = {
        "Kansas City Chiefs": NFLTeamStrength(
            team="Kansas City Chiefs",
            off_efficiency=1.25,
            def_efficiency=0.92,
            off_explosiveness=1.20,
            turnover_rate=0.10,
            takeaway_rate=0.14,
            momentum=0.3,
        ),
        "Buffalo Bills": NFLTeamStrength(
            team="Buffalo Bills",
            off_efficiency=1.20,
            def_efficiency=0.88,
            off_explosiveness=1.15,
            turnover_rate=0.11,
            takeaway_rate=0.15,
            momentum=0.2,
        ),
        "San Francisco 49ers": NFLTeamStrength(
            team="San Francisco 49ers",
            off_efficiency=1.15,
            def_efficiency=0.85,
            off_explosiveness=1.10,
            turnover_rate=0.09,
            takeaway_rate=0.13,
            momentum=0.1,
        ),
        "Detroit Lions": NFLTeamStrength(
            team="Detroit Lions",
            off_efficiency=1.22,
            def_efficiency=0.95,
            off_explosiveness=1.18,
            turnover_rate=0.13,
            takeaway_rate=0.11,
            momentum=0.4,
        ),
        "Philadelphia Eagles": NFLTeamStrength(
            team="Philadelphia Eagles",
            off_efficiency=1.10,
            def_efficiency=0.90,
            turnover_rate=0.12,
            takeaway_rate=0.14,
            momentum=-0.1,
        ),
    }
    return teams


def quick_predict(
    home: str,
    away: str,
    weather: str = 'clear'
) -> Dict:
    """Quick prediction with sample teams."""
    engine = NFLQuantumEngine()
    
    for team in create_sample_teams().values():
        engine.add_team(team)
    
    return engine.predict_game(home, away, weather=weather)
