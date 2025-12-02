"""
QEPC Backtest Engine
====================
Validates predictions against actual game results.

Key metrics:
- Win prediction accuracy
- Spread error (MAE, RMSE)
- Total error
- Calibration (do 60% predictions win 60% of the time?)
- Brier score
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from qepc.core.config import get_config
from qepc.data.loader import DataLoader
from qepc.sports.nba.predictor import GamePredictor, GamePrediction


@dataclass
class BacktestResult:
    """Results from a single game backtest."""
    date: datetime
    home_team: str
    away_team: str
    
    # Predictions
    pred_home_score: float
    pred_away_score: float
    pred_spread: float
    pred_total: float
    home_win_prob: float
    confidence: float
    
    # Actuals
    actual_home_score: int
    actual_away_score: int
    actual_spread: int
    actual_total: int
    home_won: bool
    
    # Errors
    spread_error: float
    total_error: float
    winner_correct: bool
    
    @property
    def brier_score(self) -> float:
        """Brier score for this prediction (lower is better)."""
        actual = 1.0 if self.home_won else 0.0
        return (self.home_win_prob - actual) ** 2


@dataclass 
class BacktestSummary:
    """Summary statistics from a backtest run."""
    
    # Sample info
    n_games: int
    date_start: datetime
    date_end: datetime
    
    # Win accuracy
    win_accuracy: float
    win_accuracy_high_conf: float  # When confidence > 0.65
    
    # Spread metrics
    spread_mae: float  # Mean Absolute Error
    spread_rmse: float  # Root Mean Squared Error
    spread_bias: float  # Positive = overestimate home team
    
    # Total metrics
    total_mae: float
    total_rmse: float
    total_bias: float
    
    # Probabilistic metrics
    brier_score: float
    
    # Confidence breakdown
    high_conf_games: int
    high_conf_accuracy: float
    
    def print_report(self):
        """Print a formatted report."""
        print("\n" + "=" * 60)
        print("📊 QEPC BACKTEST RESULTS")
        print("=" * 60)
        print(f"""
📅 Period: {self.date_start.date()} to {self.date_end.date()}
📈 Games Analyzed: {self.n_games}

🎯 WIN PREDICTION
   Overall Accuracy:    {self.win_accuracy:.1%}
   High Confidence:     {self.win_accuracy_high_conf:.1%} ({self.high_conf_games} games)

📏 SPREAD ERROR  
   Mean Absolute Error: {self.spread_mae:.1f} pts
   RMSE:                {self.spread_rmse:.1f} pts
   Bias:                {self.spread_bias:+.1f} pts

📊 TOTAL ERROR
   Mean Absolute Error: {self.total_mae:.1f} pts
   RMSE:                {self.total_rmse:.1f} pts
   Bias:                {self.total_bias:+.1f} pts

🎲 PROBABILISTIC
   Brier Score:         {self.brier_score:.4f} (lower is better)
   (Perfect = 0.0, Random = 0.25)
""")
        print("=" * 60)


class BacktestEngine:
    """
    Backtests QEPC predictions against actual results.
    
    Uses time-travel: Only uses data available before each game.
    """
    
    def __init__(self, data_loader: DataLoader = None):
        self.loader = data_loader or DataLoader()
        self.config = get_config()
        
        self.results: List[BacktestResult] = []
    
    def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None,
        n_days: int = 30,
        verbose: bool = True
    ) -> BacktestSummary:
        """
        Run backtest over a date range.
        
        Parameters
        ----------
        start_date : str, optional
            Start date (YYYY-MM-DD). If None, uses end_date - n_days
        end_date : str, optional
            End date. If None, uses latest data
        n_days : int
            Number of days to backtest (if start_date not specified)
        verbose : bool
            Print progress
            
        Returns
        -------
        BacktestSummary with results
        """
        # Load game results
        game_results = self.loader.load_game_results()
        
        if game_results is None or game_results.empty:
            # Try team stats as fallback
            game_results = self.loader.load_team_stats()
            
        if game_results is None or game_results.empty:
            print("❌ No game results data found for backtesting")
            return None
        
        # Ensure date column
        if 'gameDate' not in game_results.columns:
            print("❌ No date column found in game results")
            return None
        
        # Determine date range
        if end_date:
            end_dt = pd.Timestamp(end_date)
        else:
            end_dt = game_results['gameDate'].max()
        
        if start_date:
            start_dt = pd.Timestamp(start_date)
        else:
            start_dt = end_dt - timedelta(days=n_days)
        
        if verbose:
            print(f"🔬 Running backtest: {start_dt.date()} to {end_dt.date()}")
        
        # Filter to date range
        mask = (game_results['gameDate'] >= start_dt) & (game_results['gameDate'] <= end_dt)
        test_games = game_results[mask].copy()
        
        # Get only home games to avoid duplicates
        if 'home' in test_games.columns:
            test_games = test_games[test_games['home'] == 1]
        elif 'Home_Team' not in test_games.columns:
            # Assume each row is already a unique game
            pass
        
        if verbose:
            print(f"📊 Found {len(test_games)} games to backtest")
        
        # Run predictions
        self.results = []
        predictor = GamePredictor(self.loader)
        
        # Group by date for time-travel backtesting
        test_games['date_only'] = test_games['gameDate'].dt.date
        dates = sorted(test_games['date_only'].unique())
        
        for i, game_date in enumerate(dates):
            if verbose and (i + 1) % 5 == 0:
                print(f"   Processing {i+1}/{len(dates)} dates...")
            
            # Prepare predictor with data BEFORE this date
            cutoff = pd.Timestamp(game_date).strftime('%Y-%m-%d')
            predictor.prepare(cutoff_date=cutoff, verbose=False)
            
            # Get games on this date
            day_games = test_games[test_games['date_only'] == game_date]
            
            for _, game in day_games.iterrows():
                result = self._backtest_single_game(predictor, game)
                if result is not None:
                    self.results.append(result)
        
        if verbose:
            print(f"✅ Completed {len(self.results)} predictions")
        
        # Calculate summary
        summary = self._calculate_summary()
        
        if verbose:
            summary.print_report()
        
        return summary
    
    def _backtest_single_game(
        self,
        predictor: GamePredictor,
        game: pd.Series
    ) -> Optional[BacktestResult]:
        """Backtest a single game."""
        
        # Extract team names
        if 'Home_Team' in game.index:
            home_team = game['Home_Team']
            away_team = game['Away_Team']
        elif 'Team' in game.index:
            home_team = game['Team']
            away_team = game.get('Opponent', game.get('opponentTeamName', ''))
        else:
            return None
        
        # Extract actual scores
        if 'Home_Score' in game.index:
            actual_home = int(game['Home_Score'])
            actual_away = int(game['Away_Score'])
        elif 'teamScore' in game.index:
            actual_home = int(game['teamScore'])
            actual_away = int(game['opponentScore'])
        else:
            return None
        
        # Make prediction
        pred = predictor.predict_game(home_team, away_team)
        
        if pred is None:
            return None
        
        # Calculate results
        actual_spread = actual_home - actual_away
        actual_total = actual_home + actual_away
        home_won = actual_home > actual_away
        
        spread_error = pred.predicted_spread - actual_spread
        total_error = pred.predicted_total - actual_total
        winner_correct = (pred.home_win_prob > 0.5) == home_won
        
        return BacktestResult(
            date=game['gameDate'],
            home_team=home_team,
            away_team=away_team,
            pred_home_score=pred.home_score,
            pred_away_score=pred.away_score,
            pred_spread=pred.predicted_spread,
            pred_total=pred.predicted_total,
            home_win_prob=pred.home_win_prob,
            confidence=pred.confidence,
            actual_home_score=actual_home,
            actual_away_score=actual_away,
            actual_spread=actual_spread,
            actual_total=actual_total,
            home_won=home_won,
            spread_error=spread_error,
            total_error=total_error,
            winner_correct=winner_correct,
        )
    
    def _calculate_summary(self) -> BacktestSummary:
        """Calculate summary statistics from results."""
        if not self.results:
            return None
        
        n = len(self.results)
        
        # Win accuracy
        win_correct = sum(1 for r in self.results if r.winner_correct)
        win_accuracy = win_correct / n
        
        # High confidence subset
        high_conf = [r for r in self.results if r.confidence > 0.65]
        high_conf_correct = sum(1 for r in high_conf if r.winner_correct)
        high_conf_accuracy = high_conf_correct / len(high_conf) if high_conf else 0
        
        # Spread metrics
        spread_errors = [r.spread_error for r in self.results]
        spread_mae = np.mean(np.abs(spread_errors))
        spread_rmse = np.sqrt(np.mean(np.square(spread_errors)))
        spread_bias = np.mean(spread_errors)
        
        # Total metrics
        total_errors = [r.total_error for r in self.results]
        total_mae = np.mean(np.abs(total_errors))
        total_rmse = np.sqrt(np.mean(np.square(total_errors)))
        total_bias = np.mean(total_errors)
        
        # Brier score
        brier = np.mean([r.brier_score for r in self.results])
        
        # Date range
        dates = [r.date for r in self.results]
        
        return BacktestSummary(
            n_games=n,
            date_start=min(dates),
            date_end=max(dates),
            win_accuracy=win_accuracy,
            win_accuracy_high_conf=high_conf_accuracy,
            spread_mae=spread_mae,
            spread_rmse=spread_rmse,
            spread_bias=spread_bias,
            total_mae=total_mae,
            total_rmse=total_rmse,
            total_bias=total_bias,
            brier_score=brier,
            high_conf_games=len(high_conf),
            high_conf_accuracy=high_conf_accuracy,
        )
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for r in self.results:
            rows.append({
                'Date': r.date,
                'Home_Team': r.home_team,
                'Away_Team': r.away_team,
                'Pred_Home': round(r.pred_home_score, 1),
                'Pred_Away': round(r.pred_away_score, 1),
                'Pred_Spread': round(r.pred_spread, 1),
                'Home_Win_Prob': round(r.home_win_prob, 3),
                'Confidence': round(r.confidence, 3),
                'Actual_Home': r.actual_home_score,
                'Actual_Away': r.actual_away_score,
                'Actual_Spread': r.actual_spread,
                'Winner_Correct': r.winner_correct,
                'Spread_Error': round(r.spread_error, 1),
                'Total_Error': round(r.total_error, 1),
            })
        
        return pd.DataFrame(rows)
    
    def save_results(self, output_dir: Path = None) -> Path:
        """Save results to CSV."""
        if output_dir is None:
            output_dir = self.loader.project_root / "data" / "results" / "backtests"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"QEPC_Backtest_{timestamp}.csv"
        output_path = output_dir / filename
        
        df = self.results_to_dataframe()
        df.to_csv(output_path, index=False)
        
        print(f"💾 Saved to: {output_path}")
        return output_path
    
    def calibration_analysis(self) -> pd.DataFrame:
        """
        Analyze calibration: Do X% predictions win X% of the time?
        
        Good calibration = predicted probabilities match actual win rates
        """
        if not self.results:
            return pd.DataFrame()
        
        # Bin predictions by probability
        bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0]
        labels = ['0-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-100%']
        
        df = self.results_to_dataframe()
        df['Prob_Bin'] = pd.cut(df['Home_Win_Prob'], bins=bins, labels=labels)
        
        calibration = df.groupby('Prob_Bin', observed=True).agg({
            'Winner_Correct': ['count', 'mean'],
            'Home_Win_Prob': 'mean'
        }).round(3)
        
        calibration.columns = ['Games', 'Actual_Win_Rate', 'Avg_Predicted_Prob']
        
        return calibration


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def run_backtest(
    n_days: int = 30,
    verbose: bool = True
) -> BacktestSummary:
    """
    Quick backtest function.
    
    Usage:
        results = run_backtest(n_days=30)
    """
    engine = BacktestEngine()
    return engine.run_backtest(n_days=n_days, verbose=verbose)
