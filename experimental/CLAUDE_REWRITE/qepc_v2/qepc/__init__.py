"""
QEPC - Quantum Enhanced Prediction Calculator
==============================================

A quantum-inspired sports prediction engine.

Quick Start:
    from qepc import predict_today, quick_predict, run_backtest
    
    # Predict today's games
    predictions = predict_today()
    
    # Predict a single game
    pred = quick_predict("Boston Celtics", "Los Angeles Lakers")
    
    # Run backtest
    results = run_backtest(n_days=30)
"""

__version__ = "2.0.0"

# Core imports
from qepc.core.config import get_config, QEPCConfig
from qepc.core.quantum import (
    QuantumSimulator,
    QuantumState,
    QuantumConfig,
)

# Data imports
from qepc.data.loader import DataLoader, load_data

# NBA imports
from qepc.sports.nba.predictor import (
    GamePredictor,
    GamePrediction,
    quick_predict,
    predict_today,
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

__all__ = [
    # Version
    '__version__',
    
    # Config
    'get_config',
    'QEPCConfig',
    
    # Quantum
    'QuantumSimulator',
    'QuantumState',
    'QuantumConfig',
    
    # Data
    'DataLoader',
    'load_data',
    
    # Predictor
    'GamePredictor',
    'GamePrediction',
    'quick_predict',
    'predict_today',
    
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
