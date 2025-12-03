"""
QEPC NBA Module
===============
NBA-specific prediction components.
"""

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

__all__ = [
    # Prediction
    'GamePredictor',
    'GamePrediction',
    'VegasComparison',
    'quick_predict',
    'predict_today',
    'find_edges',
    
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
