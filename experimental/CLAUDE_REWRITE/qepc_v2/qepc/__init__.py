"""
QEPC - Quantum Enhanced Prediction Calculator
==============================================

A quantum-inspired sports prediction engine that uses:
- Monte Carlo simulation with quantum state modeling
- Entanglement (correlated team performances)
- Interference (matchup effects)
- Tunneling (upset probability floor)
- Recency-weighted team strengths
- Vegas odds comparison

Quick Start:
-----------
    from qepc import quick_predict, find_edges
    
    # Predict a single game
    pred = quick_predict("Boston Celtics", "Los Angeles Lakers")
    
    # Find games where QEPC disagrees with Vegas
    edges = find_edges()

Full Usage:
----------
    from qepc import GamePredictor, DataLoader
    
    loader = DataLoader(project_root=Path("C:/Users/wdors/qepc_project"))
    predictor = GamePredictor(data_loader=loader)
    predictor.prepare()
    
    # Today's predictions with Vegas comparison
    predictions = predictor.predict_today()
    
    # Find value bets
    edges = predictor.find_edges()

Version: 2.1.0 (with Vegas odds integration)
"""

__version__ = "2.1.0"
__author__ = "Will"

# Core imports
from qepc.core.config import get_config, QEPCConfig
from qepc.core.quantum import (
    QuantumSimulator,
    QuantumState,
    PerformanceState,
)

# Data loading
from qepc.data.loader import DataLoader, load_data

# NBA prediction
from qepc.sports.nba.predictor import (
    GamePredictor,
    GamePrediction,
    VegasComparison,
    quick_predict,
    predict_today,
    find_edges,
)
from qepc.sports.nba.strengths import (
    StrengthCalculator,
    TeamStrength,
    calculate_strengths,
)
from qepc.sports.nba.backtest import (
    BacktestEngine,
    BacktestResult,
    BacktestSummary,
    run_backtest,
)

# Convenience aliases
predict = quick_predict
edges = find_edges

__all__ = [
    # Version
    '__version__',
    
    # Config
    'get_config',
    'QEPCConfig',
    
    # Quantum
    'QuantumSimulator',
    'QuantumState',
    'PerformanceState',
    
    # Data
    'DataLoader',
    'load_data',
    
    # Prediction
    'GamePredictor',
    'GamePrediction',
    'VegasComparison',
    'quick_predict',
    'predict_today',
    'find_edges',
    'predict',
    'edges',
    
    # Strengths
    'StrengthCalculator',
    'TeamStrength',
    'calculate_strengths',
    
    # Backtest
    'BacktestEngine',
    'BacktestResult',
    'BacktestSummary',
    'run_backtest',
]
