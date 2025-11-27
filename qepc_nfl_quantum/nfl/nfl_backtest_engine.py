"""
QEPC NFL Quantum Backtest Engine
================================

Backtests the NFL quantum model against historical results.

Features:
- Time-travel backtesting (no lookahead)
- Against-the-spread (ATS) accuracy
- Over/under accuracy
- Quantum state analysis
- Calibration metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from nfl_quantum_engine import NFLQuantumEngine, NFLTeamStrength, NFLQuantumConfig
from nfl_strengths import NFLStrengthCalculator


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class BacktestConfig:
    """Backtest configuration."""
    
    # Simulation settings
    N_SIMULATIONS = 5000          # Per game
    
    # Line generation (if actual lines not provided)
    SPREAD_NOISE_STD = 2.0        # Points of noise
    TOTAL_NOISE_STD = 3.0
    
    # Edge thresholds
    MIN_EDGE_ATS = 0.03           # 3% edge to bet spread
    MIN_EDGE_TOTAL = 0.03


# =============================================================================
# BACKTEST RESULT
# =============================================================================

@dataclass
class GameBacktestResult:
    """Result of backtesting a single game."""
    game_date: str
    home_team: str
    away_team: str
    
    # Predictions
    predicted_home_win_prob: float
    predicted_spread: float
    predicted_total: float
    
    # Actuals
    actual_home_score: int
    actual_away_score: int
    actual_spread: float
    actual_total: int
    
    # Lines (actual or simulated)
    spread_line: float
    total_line: float
    
    # Outcomes
    home_won: bool
    correct_winner: bool
    home_covered: bool
    correct_ats: bool
    over_hit: bool
    correct_ou: bool
    
    # Errors
    spread_error: float
    total_error: float
    
    # Quantum analysis
    home_state: str
    away_state: str


@dataclass
class BacktestSummary:
    """Summary of backtest results."""
    total_games: int
    
    # Straight up
    correct_winners: int
    winner_accuracy: float
    
    # Against the spread
    ats_record: Tuple[int, int, int]  # W-L-P
    ats_accuracy: float
    
    # Totals
    ou_record: Tuple[int, int, int]
    ou_accuracy: float
    
    # Error metrics
    avg_spread_error: float
    avg_total_error: float
    
    # Calibration
    brier_score: float
    
    # ROI simulation
    ats_roi: float
    ou_roi: float


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class NFLBacktestEngine:
    """
    Backtests NFL quantum model.
    
    Usage:
        engine = NFLBacktestEngine("path/to/games.csv")
        results = engine.run_backtest("2024-09-01", "2024-11-15")
        summary = engine.get_summary()
    """
    
    def __init__(self, games_path: str = None):
        self.games_path = games_path
        self.config = BacktestConfig()
        self.games: Optional[pd.DataFrame] = None
        self.results: List[GameBacktestResult] = []
    
    def load_games(self, filepath: str = None):
        """Load historical games."""
        path = filepath or self.games_path
        
        if path is None:
            raise ValueError("No games file specified")
        
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        
        # Parse dates
        for col in ['game_date', 'gamedate', 'date']:
            if col in df.columns:
                df['game_date'] = pd.to_datetime(df[col], errors='coerce')
                break
        
        self.games = df
        print(f"[Backtest] Loaded {len(df)} games")
    
    def _generate_line(self, actual: float, noise_std: float, round_to: float = 0.5) -> float:
        """Generate a simulated betting line."""
        noise = np.random.normal(0, noise_std)
        line = actual + noise
        
        # Round to nearest 0.5
        line = round(line * 2) / 2
        
        return line
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        actual_lines: Optional[pd.DataFrame] = None,
        verbose: bool = True,
    ) -> List[GameBacktestResult]:
        """
        Run backtest over a date range.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        actual_lines : DataFrame, optional
            Actual betting lines. Columns: game_date, home_team, spread, total
        verbose : bool
            Print progress
        
        Returns
        -------
        List of GameBacktestResult
        """
        if self.games is None:
            self.load_games()
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        # Filter games
        games_in_range = self.games[
            (self.games['game_date'] >= start) &
            (self.games['game_date'] <= end)
        ].copy()
        
        if verbose:
            print(f"[Backtest] {len(games_in_range)} games from {start_date} to {end_date}")
        
        # Get unique game dates
        game_dates = sorted(games_in_range['game_date'].unique())
        
        results = []
        
        for game_date in game_dates:
            date_str = game_date.strftime('%Y-%m-%d')
            
            # Games on this date
            todays_games = games_in_range[games_in_range['game_date'] == game_date]
            
            # Build model with data BEFORE this date (no lookahead)
            try:
                strength_calc = NFLStrengthCalculator()
                strength_calc.load_games(self.games_path, cutoff_date=date_str)
                team_strengths = strength_calc.calculate_all_strengths()
            except Exception as e:
                if verbose:
                    print(f"  Skipping {date_str}: {e}")
                continue
            
            # Create prediction engine
            engine = NFLQuantumEngine()
            
            for _, row in team_strengths.iterrows():
                strength = NFLTeamStrength(
                    team=row['team'],
                    off_efficiency=row['off_efficiency'],
                    def_efficiency=row['def_efficiency'],
                    off_explosiveness=row.get('off_explosiveness', 1.0),
                    def_explosiveness=row.get('def_explosiveness', 1.0),
                    turnover_rate=row.get('turnover_rate', 0.12),
                    takeaway_rate=row.get('takeaway_rate', 0.12),
                    momentum=row.get('momentum', 0.0),
                )
                engine.add_team(strength)
            
            # Predict each game
            for _, game in todays_games.iterrows():
                home_team = game['home_team']
                away_team = game['away_team']
                
                # Check if teams exist in model
                if home_team not in engine.teams or away_team not in engine.teams:
                    continue
                
                # Get prediction
                try:
                    pred = engine.predict_game(
                        home_team, away_team,
                        n_simulations=self.config.N_SIMULATIONS
                    )
                except Exception as e:
                    if verbose:
                        print(f"  Error predicting {home_team} vs {away_team}: {e}")
                    continue
                
                # Actual results
                actual_home = int(game['home_score'])
                actual_away = int(game['away_score'])
                actual_spread = actual_home - actual_away
                actual_total = actual_home + actual_away
                
                # Get or generate lines
                if actual_lines is not None:
                    line_row = actual_lines[
                        (actual_lines['game_date'] == game_date) &
                        (actual_lines['home_team'] == home_team)
                    ]
                    if not line_row.empty:
                        spread_line = line_row.iloc[0]['spread']
                        total_line = line_row.iloc[0]['total']
                    else:
                        spread_line = self._generate_line(actual_spread, self.config.SPREAD_NOISE_STD)
                        total_line = self._generate_line(actual_total, self.config.TOTAL_NOISE_STD)
                else:
                    spread_line = self._generate_line(actual_spread, self.config.SPREAD_NOISE_STD)
                    total_line = self._generate_line(actual_total, self.config.TOTAL_NOISE_STD)
                
                # Determine outcomes
                home_won = actual_home > actual_away
                predicted_home_win = pred['home_win_prob'] > 0.5
                correct_winner = home_won == predicted_home_win
                
                # ATS: spread_line is from home perspective
                # If spread_line = -7, home favored by 7
                # Home covers if margin > -spread_line
                home_covered_actual = actual_spread > -spread_line
                home_covered_pred = pred['predicted_spread'] > -spread_line
                correct_ats = home_covered_actual == home_covered_pred
                
                # Over/under
                over_hit = actual_total > total_line
                over_pred = pred['predicted_total'] > total_line
                correct_ou = over_hit == over_pred
                
                # Errors
                spread_error = pred['predicted_spread'] - actual_spread
                total_error = pred['predicted_total'] - actual_total
                
                result = GameBacktestResult(
                    game_date=date_str,
                    home_team=home_team,
                    away_team=away_team,
                    predicted_home_win_prob=pred['home_win_prob'],
                    predicted_spread=pred['predicted_spread'],
                    predicted_total=pred['predicted_total'],
                    actual_home_score=actual_home,
                    actual_away_score=actual_away,
                    actual_spread=actual_spread,
                    actual_total=actual_total,
                    spread_line=spread_line,
                    total_line=total_line,
                    home_won=home_won,
                    correct_winner=correct_winner,
                    home_covered=home_covered_actual,
                    correct_ats=correct_ats,
                    over_hit=over_hit,
                    correct_ou=correct_ou,
                    spread_error=spread_error,
                    total_error=total_error,
                    home_state=pred['state_distribution'].get('dominant', 0),
                    away_state=pred['state_distribution'].get('baseline', 0),
                )
                
                results.append(result)
            
            if verbose and len(results) % 20 == 0:
                print(f"  Processed {len(results)} games...")
        
        self.results = results
        
        if verbose:
            print(f"[Backtest] Complete! {len(results)} games evaluated")
        
        return results
    
    def get_summary(self) -> BacktestSummary:
        """Calculate summary statistics."""
        if not self.results:
            raise ValueError("No results - run backtest first")
        
        n = len(self.results)
        
        # Straight up
        correct_winners = sum(r.correct_winner for r in self.results)
        winner_acc = correct_winners / n
        
        # ATS
        ats_wins = sum(r.correct_ats for r in self.results)
        ats_losses = n - ats_wins
        # Count pushes as half
        ats_acc = ats_wins / n
        
        # Over/under
        ou_wins = sum(r.correct_ou for r in self.results)
        ou_losses = n - ou_wins
        ou_acc = ou_wins / n
        
        # Errors
        avg_spread_error = np.mean([abs(r.spread_error) for r in self.results])
        avg_total_error = np.mean([abs(r.total_error) for r in self.results])
        
        # Brier score
        brier_scores = []
        for r in self.results:
            outcome = 1 if r.home_won else 0
            brier_scores.append((r.predicted_home_win_prob - outcome) ** 2)
        brier_score = np.mean(brier_scores)
        
        # ROI (assuming -110 odds)
        ats_profit = ats_wins * 0.909 - ats_losses * 1.0
        ats_roi = ats_profit / n
        
        ou_profit = ou_wins * 0.909 - ou_losses * 1.0
        ou_roi = ou_profit / n
        
        return BacktestSummary(
            total_games=n,
            correct_winners=correct_winners,
            winner_accuracy=winner_acc,
            ats_record=(ats_wins, ats_losses, 0),
            ats_accuracy=ats_acc,
            ou_record=(ou_wins, ou_losses, 0),
            ou_accuracy=ou_acc,
            avg_spread_error=avg_spread_error,
            avg_total_error=avg_total_error,
            brier_score=brier_score,
            ats_roi=ats_roi,
            ou_roi=ou_roi,
        )
    
    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to DataFrame."""
        rows = []
        for r in self.results:
            rows.append({
                'date': r.game_date,
                'home_team': r.home_team,
                'away_team': r.away_team,
                'pred_home_prob': r.predicted_home_win_prob,
                'pred_spread': r.predicted_spread,
                'pred_total': r.predicted_total,
                'actual_home': r.actual_home_score,
                'actual_away': r.actual_away_score,
                'actual_spread': r.actual_spread,
                'actual_total': r.actual_total,
                'spread_line': r.spread_line,
                'total_line': r.total_line,
                'home_won': r.home_won,
                'correct_winner': r.correct_winner,
                'correct_ats': r.correct_ats,
                'correct_ou': r.correct_ou,
                'spread_error': r.spread_error,
                'total_error': r.total_error,
            })
        
        return pd.DataFrame(rows)
    
    def print_summary(self):
        """Print formatted summary."""
        s = self.get_summary()
        
        print("=" * 60)
        print("NFL QUANTUM MODEL BACKTEST RESULTS")
        print("=" * 60)
        print(f"\nTotal Games: {s.total_games}")
        print(f"\n--- STRAIGHT UP ---")
        print(f"Correct Winners: {s.correct_winners}/{s.total_games}")
        print(f"Accuracy: {s.winner_accuracy:.1%}")
        print(f"\n--- AGAINST THE SPREAD ---")
        print(f"Record: {s.ats_record[0]}-{s.ats_record[1]}")
        print(f"Accuracy: {s.ats_accuracy:.1%}")
        print(f"ROI: {s.ats_roi:.1%}")
        print(f"\n--- OVER/UNDER ---")
        print(f"Record: {s.ou_record[0]}-{s.ou_record[1]}")
        print(f"Accuracy: {s.ou_accuracy:.1%}")
        print(f"ROI: {s.ou_roi:.1%}")
        print(f"\n--- ERROR METRICS ---")
        print(f"Avg Spread Error: {s.avg_spread_error:.1f} pts")
        print(f"Avg Total Error: {s.avg_total_error:.1f} pts")
        print(f"Brier Score: {s.brier_score:.4f}")
        print("=" * 60)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_backtest(
    games_path: str,
    start_date: str,
    end_date: str,
) -> BacktestSummary:
    """Run a quick backtest and return summary."""
    engine = NFLBacktestEngine(games_path)
    engine.run_backtest(start_date, end_date)
    return engine.get_summary()
