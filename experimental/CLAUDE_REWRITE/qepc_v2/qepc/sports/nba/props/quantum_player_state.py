"""
Quantum Player State System
Players exist in superposition until game-time measurement
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PerformanceState:
    """Single quantum state with weight and multiplier"""
    name: str
    weight: float
    multiplier: float
    description: str


class QuantumPlayerState:
    """
    Represents a player's performance as quantum superposition
    
    Key Concepts:
    - Superposition: Player exists in multiple performance states simultaneously
    - Collapse: External factors (matchup, rest, etc.) collapse to actual prediction
    - Entanglement: Performance correlated with teammates/opponents
    - Decoherence: Fatigue/injuries reduce quantum coherence
    """
    
    def __init__(self, player_id: str, player_name: str, historical_games: pd.DataFrame):
        self.player_id = player_id
        self.player_name = player_name
        self.historical_games = historical_games
        
        # Calculate base statistics
        self.base_stats = self._calculate_base_stats()
        
        # Create superposition states
        self.superposition_states = self._create_superposition()
        
        # Quantum coherence (decreases with fatigue/injury)
        self.coherence = 1.0
    
    def _calculate_base_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate baseline statistics from historical games
        
        Returns:
            Dict of {stat: {mean, std, median, q25, q75}}
        """
        stats = {}
        
        for stat_name in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M']:
            if stat_name in self.historical_games.columns:
                values = self.historical_games[stat_name].dropna()
                
                stats[stat_name] = {
                    'mean': values.mean(),
                    'std': values.std(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'min': values.min(),
                    'max': values.max(),
                    'games': len(values)
                }
        
        return stats
    
    def _create_superposition(self) -> List[PerformanceState]:
        """
        Create quantum superposition of performance states
        
        Returns:
            List of PerformanceStates with weights that sum to 1.0
        """
        # Recent form check (last 5 games)
        recent_games = self.historical_games.tail(5)
        recent_avg = recent_games['PTS'].mean() if 'PTS' in recent_games else 0
        season_avg = self.base_stats.get('PTS', {}).get('mean', 0)
        
        # Determine if hot or cold
        hot_threshold = season_avg + self.base_stats.get('PTS', {}).get('std', 5)
        cold_threshold = season_avg - self.base_stats.get('PTS', {}).get('std', 5)
        
        is_hot = recent_avg > hot_threshold
        is_cold = recent_avg < cold_threshold
        
        # Adjust weights based on form
        if is_hot:
            weights = {'HOT': 0.35, 'BASE': 0.45, 'COLD': 0.15, 'VARIANCE': 0.05}
        elif is_cold:
            weights = {'HOT': 0.15, 'BASE': 0.45, 'COLD': 0.35, 'VARIANCE': 0.05}
        else:
            weights = {'HOT': 0.25, 'BASE': 0.50, 'COLD': 0.20, 'VARIANCE': 0.05}
        
        # Create states
        states = [
            PerformanceState(
                name='HOT',
                weight=weights['HOT'],
                multiplier=1.15,
                description='Above-average performance state'
            ),
            PerformanceState(
                name='BASE',
                weight=weights['BASE'],
                multiplier=1.00,
                description='Season average state'
            ),
            PerformanceState(
                name='COLD',
                weight=weights['COLD'],
                multiplier=0.85,
                description='Below-average performance state'
            ),
            PerformanceState(
                name='VARIANCE',
                weight=weights['VARIANCE'],
                multiplier=0.0,  # Will be random
                description='High-variance boom/bust state'
            )
        ]
        
        return states
    
    def collapse_state(self, 
                      opponent_defense_rating: float = 1.0,
                      rest_days: int = 1,
                      is_home: bool = True,
                      injury_status: str = 'healthy') -> Dict[str, float]:
        """
        Collapse quantum superposition to actual prediction
        
        This is the "measurement" in quantum mechanics - the superposition
        collapses to a single outcome based on external factors
        
        Args:
            opponent_defense_rating: Defensive strength (0.8 = weak, 1.2 = strong)
            rest_days: Days of rest (0 = back-to-back, 1+ = normal)
            is_home: Home court advantage
            injury_status: 'healthy', 'questionable', 'out'
        
        Returns:
            Dict of predicted stats with quantum-enhanced values
        """
        # Calculate decoherence from external factors
        self._apply_decoherence(rest_days, injury_status)
        
        # Weighted average across superposition states
        predictions = {}
        
        for stat_name, stat_data in self.base_stats.items():
            base_value = stat_data['mean']
            std_dev = stat_data['std']
            
            # Quantum superposition: weighted sum of states
            quantum_value = 0
            for state in self.superposition_states:
                if state.name == 'VARIANCE':
                    # Boom/bust: sample from extremes
                    variance_sample = np.random.choice([
                        stat_data['q75'] + std_dev,  # Boom
                        stat_data['q25'] - std_dev   # Bust
                    ])
                    quantum_value += state.weight * variance_sample
                else:
                    quantum_value += state.weight * (base_value * state.multiplier)
            
            # Apply external factors
            quantum_value *= opponent_defense_rating  # Opponent effect
            quantum_value *= (1.05 if is_home else 0.95)  # Home court
            quantum_value *= self.coherence  # Decoherence effect
            
            predictions[stat_name] = {
                'predicted': quantum_value,
                'confidence': self.coherence,
                'base': base_value,
                'std': std_dev
            }
        
        return predictions
    
    def _apply_decoherence(self, rest_days: int, injury_status: str):
        """
        Decoherence: Loss of quantum properties due to environment
        
        In physics: Quantum systems lose coherence when interacting with environment
        In sports: Fatigue and injury reduce prediction confidence
        """
        # Rest-based decoherence
        if rest_days == 0:  # Back-to-back
            self.coherence *= 0.90
        elif rest_days >= 3:  # Well-rested
            self.coherence = min(1.0, self.coherence * 1.05)
        
        # Injury-based decoherence
        injury_factors = {
            'healthy': 1.0,
            'questionable': 0.85,
            'doubtful': 0.70,
            'out': 0.0
        }
        self.coherence *= injury_factors.get(injury_status, 1.0)
        
        # Keep coherence in valid range [0, 1]
        self.coherence = max(0.0, min(1.0, self.coherence))
    
    def get_superposition_info(self) -> Dict:
        """Get current superposition state info"""
        return {
            'player': self.player_name,
            'coherence': self.coherence,
            'states': [
                {
                    'name': s.name,
                    'weight': s.weight,
                    'multiplier': s.multiplier,
                    'description': s.description
                }
                for s in self.superposition_states
            ],
            'base_stats': self.base_stats
        }


if __name__ == "__main__":
    # Test with sample data
    print("🔮 Testing Quantum Player State...")
    
    # Create sample player data
    sample_games = pd.DataFrame({
        'PTS': np.random.normal(25, 6, 30),
        'REB': np.random.normal(8, 3, 30),
        'AST': np.random.normal(7, 2, 30)
    })
    
    # Create quantum state
    player = QuantumPlayerState(
        player_id="203999",
        player_name="Test Player",
        historical_games=sample_games
    )
    
    # Show superposition
    info = player.get_superposition_info()
    print(f"\n👤 Player: {info['player']}")
    print(f"🌊 Coherence: {info['coherence']:.2f}")
    print(f"\n🎭 Superposition States:")
    for state in info['states']:
        print(f"   |{state['name']}⟩: {state['weight']:.1%} × {state['multiplier']:.2f}")
    
    # Collapse state
    print(f"\n📏 Collapsing state (measurement)...")
    prediction = player.collapse_state(
        opponent_defense_rating=1.1,  # Strong defense
        rest_days=1,
        is_home=True
    )
    
    print(f"\n🎯 Predictions:")
    for stat, values in prediction.items():
        print(f"   {stat}: {values['predicted']:.1f} "
              f"(base: {values['base']:.1f}, confidence: {values['confidence']:.0%})")
