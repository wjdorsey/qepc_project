"""
QEPC Player Props Backtest Engine
==================================

Backtests player prop predictions against historical results.

Features:
- Time-travel backtesting (no lookahead bias)
- Multiple metrics: accuracy, Brier score, calibration
- Breakdown by prop type, player, confidence level
- Edge analysis for simulated betting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from player_props_engine import PlayerPropsEngine, PropPrediction


# =============================================================================
# CONFIGURATION
# =============================================================================

class BacktestConfig:
    """Configuration for backtesting."""
    
    # Line generation (simulate betting lines from actual results)
    LINE_NOISE_STD = 1.5      # Std dev of noise added to actual result
    LINE_ROUND_TO = 0.5       # Round lines to nearest 0.5
    
    # Edge thresholds for simulated betting
    MIN_EDGE_TO_BET = 0.05    # 5% minimum edge
    
    # Default props to backtest
    DEFAULT_PROPS = ['PTS', 'REB', 'AST', '3PM', 'PRA']


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class BacktestResult:
    """Result of a single prop backtest."""
    player_name: str
    game_date: str
    opponent: str
    prop_type: str
    projection: float
    actual: float
    std_dev: float
    line: float  # Simulated or actual line
    over_prob: float
    confidence: str
    
    # Outcomes
    prediction_error: float  # projection - actual
    abs_error: float
    over_hit: bool  # Did OVER hit?
    under_hit: bool
    correct_side: bool  # Did we predict the right side?
    brier_score: float  # (prob - outcome)^2
    
    @property
    def pct_error(self) -> float:
        """Percentage error."""
        if self.actual == 0:
            return 0 if self.projection == 0 else 1.0
        return abs(self.prediction_error) / self.actual


@dataclass
class BacktestSummary:
    """Summary statistics for a backtest run."""
    total_predictions: int
    
    # Accuracy metrics
    mean_absolute_error: float
    mean_pct_error: float
    median_absolute_error: float
    
    # Directional accuracy (vs line)
    over_accuracy: float  # When we said over, how often was it over?
    under_accuracy: float
    overall_accuracy: float
    
    # Calibration
    brier_score: float
    
    # By confidence
    high_conf_accuracy: float
    medium_conf_accuracy: float
    low_conf_accuracy: float
    
    # Edge simulation
    simulated_bets: int
    simulated_wins: int
    simulated_roi: float


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class PropsBacktestEngine:
    """
    Backtests player prop predictions.
    
    Usage:
        engine = PropsBacktestEngine("path/to/PlayerStatistics.csv")
        results = engine.run_backtest("2024-11-01", "2024-11-15")
        summary = engine.get_summary(results)
    """
    
    def __init__(self, data_path: str = None):
        """
        Initialize backtest engine.
        
        Parameters
        ----------
        data_path : str
            Path to PlayerStatistics.csv
        """
        self.data_path = Path(data_path) if data_path else None
        self.config = BacktestConfig()
        self.raw_data: Optional[pd.DataFrame] = None
        self.results: List[BacktestResult] = []
    
    def _load_raw_data(self) -> pd.DataFrame:
        """Load and prepare raw data for backtesting."""
        if self.data_path is None:
            # Try to find it
            candidates = [
                Path("data/raw/PlayerStatistics.csv"),
                Path("../data/raw/PlayerStatistics.csv"),
                Path.cwd() / "data" / "raw" / "PlayerStatistics.csv",
            ]
            for p in candidates:
                if p.exists():
                    self.data_path = p
                    break
        
        if self.data_path is None or not self.data_path.exists():
            raise FileNotFoundError("Cannot find PlayerStatistics.csv")
        
        df = pd.read_csv(self.data_path)
        
        # Standardize columns
        df.columns = df.columns.str.strip()
        
        # Player name
        if 'playerName' not in df.columns:
            if 'firstName' in df.columns and 'lastName' in df.columns:
                df['playerName'] = df['firstName'].astype(str) + ' ' + df['lastName'].astype(str)
        
        # Parse date
        df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')
        
        # Team/opponent
        for col in ['playerteamName', 'teamName']:
            if col in df.columns:
                df['team'] = df[col]
                break
        
        for col in ['opponentTeamName', 'opponentteamName']:
            if col in df.columns:
                df['opponent'] = df[col]
                break
        
        # Home indicator
        if 'home' in df.columns:
            df['is_home'] = df['home'].astype(int) == 1
        else:
            df['is_home'] = True
        
        return df
    
    def _get_actual_stat(self, row: pd.Series, prop_type: str) -> Optional[float]:
        """Get actual stat value from a game row."""
        mapping = {
            'PTS': 'points',
            'REB': 'reboundsTotal',
            'AST': 'assists',
            '3PM': 'threePointersMade',
            'STL': 'steals',
            'BLK': 'blocks',
            'TOV': 'turnovers',
        }
        
        if prop_type in ['PRA', 'PR', 'PA', 'RA']:
            # Combo props
            components = {
                'PRA': ['PTS', 'REB', 'AST'],
                'PR': ['PTS', 'REB'],
                'PA': ['PTS', 'AST'],
                'RA': ['REB', 'AST'],
            }
            total = 0
            for comp in components[prop_type]:
                val = self._get_actual_stat(row, comp)
                if val is None:
                    return None
                total += val
            return total
        
        col = mapping.get(prop_type)
        if col is None or col not in row.index:
            return None
        
        val = row[col]
        if pd.isna(val):
            return None
        
        return float(val)
    
    def _generate_line(self, actual: float) -> float:
        """
        Generate a simulated betting line from actual result.
        
        In real backtesting, you'd use actual Vegas lines.
        This simulates lines with some noise.
        """
        noise = np.random.normal(0, self.config.LINE_NOISE_STD)
        line = actual + noise
        
        # Round to nearest 0.5
        line = round(line * 2) / 2
        
        # Floor at 0.5
        line = max(0.5, line)
        
        return line
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        props: List[str] = None,
        min_minutes: float = 15.0,
        sample_players: Optional[int] = None,
        verbose: bool = True,
    ) -> List[BacktestResult]:
        """
        Run a backtest over a date range.
        
        Parameters
        ----------
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        props : list
            Prop types to backtest
        min_minutes : float
            Minimum average minutes to include player
        sample_players : int, optional
            Randomly sample N players (for faster testing)
        verbose : bool
            Print progress
        
        Returns
        -------
        List of BacktestResult objects
        """
        if props is None:
            props = self.config.DEFAULT_PROPS
        
        # Load raw data
        if self.raw_data is None:
            self.raw_data = self._load_raw_data()
        
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        
        if verbose:
            print(f"[Backtest] Running from {start_date} to {end_date}")
            print(f"[Backtest] Props: {props}")
        
        # Get games in date range
        games_in_range = self.raw_data[
            (self.raw_data['gameDate'] >= start) &
            (self.raw_data['gameDate'] <= end)
        ].copy()
        
        if games_in_range.empty:
            print("[Backtest] No games found in date range!")
            return []
        
        if verbose:
            print(f"[Backtest] Found {len(games_in_range)} player-game records")
        
        # Get unique game dates
        game_dates = sorted(games_in_range['gameDate'].unique())
        
        results = []
        
        for game_date in game_dates:
            # Get games on this date
            todays_games = games_in_range[games_in_range['gameDate'] == game_date]
            
            # Create prediction engine with cutoff (no lookahead!)
            cutoff = game_date.strftime('%Y-%m-%d')
            
            try:
                engine = PlayerPropsEngine(self.data_path)
                engine.load_data(cutoff_date=cutoff, verbose=False)
            except Exception as e:
                if verbose:
                    print(f"[Backtest] Skipping {cutoff}: {e}")
                continue
            
            # Get players to predict
            players_today = todays_games['playerName'].unique()
            
            if sample_players:
                players_today = np.random.choice(
                    players_today, 
                    min(sample_players, len(players_today)),
                    replace=False
                )
            
            for player_name in players_today:
                player_games = todays_games[todays_games['playerName'] == player_name]
                
                for _, game_row in player_games.iterrows():
                    opponent = game_row.get('opponent', 'Unknown')
                    is_home = game_row.get('is_home', True)
                    
                    for prop_type in props:
                        # Get actual result
                        actual = self._get_actual_stat(game_row, prop_type)
                        if actual is None:
                            continue
                        
                        # Get prediction
                        pred = engine.predict(
                            player_name,
                            prop_type,
                            opponent=opponent,
                            is_home=is_home,
                            game_date=cutoff,
                        )
                        
                        if pred is None:
                            continue
                        
                        # Generate simulated line
                        line = self._generate_line(actual)
                        
                        # Calculate outcomes
                        over_prob = pred.over_prob(line)
                        prediction_error = pred.projection - actual
                        abs_error = abs(prediction_error)
                        
                        over_hit = actual > line
                        under_hit = actual < line
                        
                        # Did we predict correct side?
                        if over_prob > 0.5:
                            correct_side = over_hit
                        else:
                            correct_side = under_hit
                        
                        # Brier score
                        outcome = 1 if over_hit else 0
                        brier = (over_prob - outcome) ** 2
                        
                        result = BacktestResult(
                            player_name=player_name,
                            game_date=cutoff,
                            opponent=opponent,
                            prop_type=prop_type,
                            projection=pred.projection,
                            actual=actual,
                            std_dev=pred.std_dev,
                            line=line,
                            over_prob=over_prob,
                            confidence=pred.confidence,
                            prediction_error=prediction_error,
                            abs_error=abs_error,
                            over_hit=over_hit,
                            under_hit=under_hit,
                            correct_side=correct_side,
                            brier_score=brier,
                        )
                        
                        results.append(result)
            
            if verbose and len(results) % 500 == 0:
                print(f"[Backtest] Processed {len(results)} predictions...")
        
        self.results = results
        
        if verbose:
            print(f"[Backtest] Complete! {len(results)} total predictions")
        
        return results
    
    def get_summary(self, results: List[BacktestResult] = None) -> BacktestSummary:
        """
        Calculate summary statistics from backtest results.
        
        Parameters
        ----------
        results : list, optional
            Results to summarize. Uses self.results if not provided.
        
        Returns
        -------
        BacktestSummary
        """
        if results is None:
            results = self.results
        
        if not results:
            raise ValueError("No results to summarize!")
        
        n = len(results)
        
        # Basic metrics
        abs_errors = [r.abs_error for r in results]
        pct_errors = [r.pct_error for r in results]
        
        mae = np.mean(abs_errors)
        mpe = np.mean(pct_errors)
        median_ae = np.median(abs_errors)
        
        # Directional accuracy
        over_preds = [r for r in results if r.over_prob > 0.5]
        under_preds = [r for r in results if r.over_prob <= 0.5]
        
        over_acc = np.mean([r.over_hit for r in over_preds]) if over_preds else 0
        under_acc = np.mean([r.under_hit for r in under_preds]) if under_preds else 0
        overall_acc = np.mean([r.correct_side for r in results])
        
        # Brier score
        brier = np.mean([r.brier_score for r in results])
        
        # By confidence
        high_conf = [r for r in results if r.confidence == 'HIGH']
        med_conf = [r for r in results if r.confidence == 'MEDIUM']
        low_conf = [r for r in results if r.confidence == 'LOW']
        
        high_acc = np.mean([r.correct_side for r in high_conf]) if high_conf else 0
        med_acc = np.mean([r.correct_side for r in med_conf]) if med_conf else 0
        low_acc = np.mean([r.correct_side for r in low_conf]) if low_conf else 0
        
        # Simulated betting (bet when edge > threshold)
        bets = []
        for r in results:
            edge = abs(r.over_prob - 0.5)  # Edge vs 50%
            if edge >= self.config.MIN_EDGE_TO_BET / 2:  # Adjusted threshold
                bet_over = r.over_prob > 0.5
                won = r.over_hit if bet_over else r.under_hit
                bets.append(won)
        
        sim_bets = len(bets)
        sim_wins = sum(bets)
        
        # ROI assuming -110 odds
        if sim_bets > 0:
            # Win = +0.909, Loss = -1.0
            profit = sim_wins * 0.909 - (sim_bets - sim_wins) * 1.0
            sim_roi = profit / sim_bets
        else:
            sim_roi = 0
        
        return BacktestSummary(
            total_predictions=n,
            mean_absolute_error=mae,
            mean_pct_error=mpe,
            median_absolute_error=median_ae,
            over_accuracy=over_acc,
            under_accuracy=under_acc,
            overall_accuracy=overall_acc,
            brier_score=brier,
            high_conf_accuracy=high_acc,
            medium_conf_accuracy=med_acc,
            low_conf_accuracy=low_acc,
            simulated_bets=sim_bets,
            simulated_wins=sim_wins,
            simulated_roi=sim_roi,
        )
    
    def results_to_dataframe(self, results: List[BacktestResult] = None) -> pd.DataFrame:
        """Convert results to a DataFrame."""
        if results is None:
            results = self.results
        
        rows = []
        for r in results:
            rows.append({
                'player': r.player_name,
                'date': r.game_date,
                'opponent': r.opponent,
                'prop': r.prop_type,
                'projection': r.projection,
                'actual': r.actual,
                'line': r.line,
                'over_prob': r.over_prob,
                'confidence': r.confidence,
                'error': r.prediction_error,
                'abs_error': r.abs_error,
                'pct_error': r.pct_error,
                'correct_side': r.correct_side,
                'brier': r.brier_score,
            })
        
        return pd.DataFrame(rows)
    
    def breakdown_by_prop(self, results: List[BacktestResult] = None) -> pd.DataFrame:
        """Get accuracy breakdown by prop type."""
        if results is None:
            results = self.results
        
        df = self.results_to_dataframe(results)
        
        summary = df.groupby('prop').agg({
            'abs_error': ['mean', 'median'],
            'pct_error': 'mean',
            'correct_side': 'mean',
            'brier': 'mean',
            'player': 'count',
        }).round(3)
        
        summary.columns = ['MAE', 'MedAE', 'MAPE', 'Accuracy', 'Brier', 'Count']
        
        return summary.sort_values('Accuracy', ascending=False)
    
    def breakdown_by_confidence(self, results: List[BacktestResult] = None) -> pd.DataFrame:
        """Get accuracy breakdown by confidence level."""
        if results is None:
            results = self.results
        
        df = self.results_to_dataframe(results)
        
        summary = df.groupby('confidence').agg({
            'abs_error': 'mean',
            'correct_side': 'mean',
            'brier': 'mean',
            'player': 'count',
        }).round(3)
        
        summary.columns = ['MAE', 'Accuracy', 'Brier', 'Count']
        
        # Reorder
        order = ['HIGH', 'MEDIUM', 'LOW']
        summary = summary.reindex([c for c in order if c in summary.index])
        
        return summary
    
    def calibration_analysis(self, results: List[BacktestResult] = None, bins: int = 10) -> pd.DataFrame:
        """
        Analyze probability calibration.
        
        If model is well-calibrated, 60% predictions should hit 60% of time.
        """
        if results is None:
            results = self.results
        
        df = self.results_to_dataframe(results)
        
        # Create probability bins
        df['prob_bin'] = pd.cut(df['over_prob'], bins=bins)
        
        # Calculate actual hit rate per bin
        calibration = df.groupby('prob_bin').agg({
            'over_prob': 'mean',  # Average predicted prob
            'correct_side': 'mean',  # Actual hit rate (for over side)
            'player': 'count',
        }).round(3)
        
        # Need to calculate actual over hit rate, not correct_side
        over_hits = df.groupby('prob_bin').apply(
            lambda x: (x['actual'] > x['line']).mean()
        )
        
        calibration['actual_over_rate'] = over_hits
        calibration.columns = ['Predicted', 'DirectionalAcc', 'Count', 'ActualOverRate']
        
        return calibration


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_quick_backtest(
    start_date: str,
    end_date: str,
    data_path: str = None,
    props: List[str] = ['PTS', 'REB', 'AST'],
) -> Tuple[BacktestSummary, pd.DataFrame]:
    """
    Run a quick backtest and return summary + details.
    
    Returns (summary, results_df)
    """
    engine = PropsBacktestEngine(data_path)
    results = engine.run_backtest(start_date, end_date, props=props)
    summary = engine.get_summary()
    df = engine.results_to_dataframe()
    
    return summary, df
