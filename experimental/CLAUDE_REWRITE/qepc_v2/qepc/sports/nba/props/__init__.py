"""
QEPC Player Props Module
========================
Quantum-inspired player prop predictions.

Quick Start:
    from qepc.sports.nba.props import quick_prop
    
    pred = quick_prop("LeBron James", "pts", vegas_line=25.5)
    
Full Usage:
    from qepc.sports.nba.props import PlayerPropsPredictor
    
    predictor = PlayerPropsPredictor()
    predictor.prepare()
    
    # Single prop
    pred = predictor.predict_prop("Stephen Curry", "fg3m", vegas_line=4.5)
    
    # Full slate
    slate = [
        ("LeBron James", "pts", 25.5),
        ("Stephen Curry", "fg3m", 4.5),
        ("Nikola Jokic", "ast", 8.5),
    ]
    predictions = predictor.predict_slate(slate)
    
    # Find edges
    edges = predictor.find_edges(predictions)
"""

from qepc.sports.nba.props.player_state import (
    PlayerProfile,
    PlayerQuantumState,
    PerformanceState,
    create_player_state,
    simulate_player_game,
)

from qepc.sports.nba.props.predictor import (
    PropPrediction,
    PlayerPropsPredictor,
    quick_prop,
)

from qepc.sports.nba.props.matchups import (
    MatchupAnalyzer,
    MatchupProfile,
    UsageEntanglement,
    calculate_teammate_adjustment,
)

__all__ = [
    # Player state
    'PlayerProfile',
    'PlayerQuantumState', 
    'PerformanceState',
    'create_player_state',
    'simulate_player_game',
    
    # Predictor
    'PropPrediction',
    'PlayerPropsPredictor',
    'quick_prop',
    
    # Matchups
    'MatchupAnalyzer',
    'MatchupProfile',
    'UsageEntanglement',
    'calculate_teammate_adjustment',
]
