"""
QEPC NBA Game Predictor
=======================
The main prediction engine that combines:
- Team strengths with recency weighting
- Quantum state simulation
- Situational adjustments (home court, rest, injuries)
- Vegas odds comparison (NEW!)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from qepc.core.config import get_config, QEPCConfig
from qepc.core.quantum import (
    QuantumSimulator, QuantumState, QuantumConfig,
    create_quantum_state_from_volatility, adjust_state_for_situation
)
from qepc.data.loader import DataLoader
from qepc.sports.nba.strengths import StrengthCalculator, TeamStrength


@dataclass
class VegasComparison:
    """Comparison between QEPC prediction and Vegas line."""
    vegas_spread: Optional[float]
    qepc_spread: float
    spread_diff: Optional[float]  # Positive = QEPC likes home more than Vegas
    
    vegas_home_prob: Optional[float]
    qepc_home_prob: float
    prob_diff: Optional[float]  # Positive = QEPC more confident in home
    
    edge: Optional[str]  # "HOME", "AWAY", or None
    edge_size: Optional[float]  # How big is the disagreement
    
    @property
    def has_edge(self) -> bool:
        return self.edge is not None and self.edge_size is not None and self.edge_size > 2.0


@dataclass
class GamePrediction:
    """Complete prediction for a single game."""
    
    # Teams
    home_team: str
    away_team: str
    game_id: Optional[str] = None
    
    # Win probabilities
    home_win_prob: float = 0.5
    away_win_prob: float = 0.5
    
    # Score predictions
    home_score: float = 110.0
    away_score: float = 108.0
    predicted_spread: float = 0.0  # Home - Away (negative = away favored)
    predicted_total: float = 218.0
    
    # Uncertainty
    spread_std: float = 10.0
    total_std: float = 12.0
    
    # Confidence (0-1, higher = more confident)
    confidence: float = 0.5
    
    # Components (for analysis)
    home_expected_raw: float = 110.0
    away_expected_raw: float = 108.0
    home_court_adjustment: float = 0.0
    rest_adjustment: float = 0.0
    injury_adjustment: float = 0.0
    
    # Vegas comparison (NEW!)
    vegas: Optional[VegasComparison] = None
    
    @property
    def predicted_winner(self) -> str:
        return self.home_team if self.home_win_prob > 0.5 else self.away_team
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence > 0.65
    
    @property
    def has_vegas_edge(self) -> bool:
        return self.vegas is not None and self.vegas.has_edge
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        result = {
            'Home_Team': self.home_team,
            'Away_Team': self.away_team,
            'Home_Win_Prob': round(self.home_win_prob, 3),
            'Away_Win_Prob': round(self.away_win_prob, 3),
            'Predicted_Winner': self.predicted_winner,
            'Home_Score': round(self.home_score, 1),
            'Away_Score': round(self.away_score, 1),
            'QEPC_Spread': round(self.predicted_spread, 1),
            'Total': round(self.predicted_total, 1),
            'Confidence': round(self.confidence, 3),
        }
        
        # Add Vegas comparison if available
        if self.vegas:
            result['Vegas_Spread'] = self.vegas.vegas_spread
            result['Spread_Diff'] = round(self.vegas.spread_diff, 1) if self.vegas.spread_diff else None
            result['Edge'] = self.vegas.edge
            result['Edge_Size'] = round(self.vegas.edge_size, 1) if self.vegas.edge_size else None
        
        return result


class GamePredictor:
    """
    Main game prediction engine.
    
    Combines team strengths, quantum simulation, situational factors,
    and Vegas odds comparison.
    """
    
    def __init__(
        self,
        data_loader: DataLoader = None,
        config: QEPCConfig = None
    ):
        self.loader = data_loader or DataLoader()
        self.config = config or get_config()
        
        # Initialize components
        self.strength_calc = StrengthCalculator(self.loader, self.config)
        self.quantum_sim = QuantumSimulator(QuantumConfig(
            entanglement_strength=self.config.quantum.entanglement_strength,
            tunneling_base_rate=self.config.quantum.tunneling_base_rate,
            tunneling_min_prob=self.config.quantum.tunneling_min_prob,
            interference_factor=self.config.quantum.interference_factor,
        ))
        
        # Cache
        self._strengths: Dict[str, TeamStrength] = {}
        self._injuries: pd.DataFrame = None
        self._rest_data: pd.DataFrame = None
        self._vegas_odds: pd.DataFrame = None
    
    def prepare(self, cutoff_date: str = None, verbose: bool = True):
        """
        Prepare the predictor by loading all required data.
        """
        if verbose:
            print("🔮 Preparing QEPC Predictor...")
        
        # Calculate team strengths
        self._strengths = self.strength_calc.calculate_all_strengths(
            cutoff_date=cutoff_date,
            verbose=verbose
        )
        
        # Load injuries
        self._injuries = self.loader.load_injuries()
        if self._injuries is not None and verbose:
            print(f"🏥 Loaded {len(self._injuries)} injury records")
        
        # Load rest data
        self._rest_data = self.loader.load_schedule_with_rest()
        
        # Load Vegas odds (NEW!)
        self._vegas_odds = self.loader.load_vegas_odds()
        if self._vegas_odds is not None and verbose:
            print(f"💰 Loaded Vegas odds for {len(self._vegas_odds)} games")
        
        if verbose:
            print(f"✅ Ready to predict! ({len(self._strengths)} teams)")
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_id: str = None,
        game_date: str = None,
        n_simulations: int = None
    ) -> Optional[GamePrediction]:
        """
        Predict a single game.
        """
        n_sims = n_simulations or self.config.simulation.default_simulations
        
        # Get team strengths
        home_strength = self._strengths.get(home_team)
        away_strength = self._strengths.get(away_team)
        
        if home_strength is None:
            print(f"⚠️  Team not found: {home_team}")
            return None
        if away_strength is None:
            print(f"⚠️  Team not found: {away_team}")
            return None
        
        # Calculate expected scores
        game_pace = (home_strength.pace + away_strength.pace) / 2
        
        # Use interference for matchup effects
        home_matchup_mod = self.quantum_sim.interference.calculate_matchup_modifier(
            home_strength.ortg, 
            away_strength.drtg,
            self.config.league.league_avg_ortg,
            self.config.league.league_avg_drtg
        )
        away_matchup_mod = self.quantum_sim.interference.calculate_matchup_modifier(
            away_strength.ortg,
            home_strength.drtg,
            self.config.league.league_avg_ortg,
            self.config.league.league_avg_drtg
        )
        
        # Base expected scores
        home_expected_raw = home_strength.ortg * (game_pace / 100) * home_matchup_mod
        away_expected_raw = away_strength.ortg * (game_pace / 100) * away_matchup_mod
        
        # Apply adjustments
        home_court_adj = self.config.home_court.get_advantage(home_team)
        rest_adj = self._calculate_rest_adjustment(home_team, away_team, game_date)
        injury_adj_home, injury_adj_away = self._calculate_injury_adjustment(home_team, away_team)
        
        # Final expected scores
        home_expected = home_expected_raw + home_court_adj + rest_adj + injury_adj_home
        away_expected = away_expected_raw + injury_adj_away
        
        # Create quantum states
        home_state = create_quantum_state_from_volatility(
            home_strength.volatility / home_strength.ppg if home_strength.ppg > 0 else 0.1
        )
        away_state = create_quantum_state_from_volatility(
            away_strength.volatility / away_strength.ppg if away_strength.ppg > 0 else 0.1
        )
        
        # Adjust for situation
        home_state = adjust_state_for_situation(
            home_state, is_home=True, momentum=home_strength.momentum
        )
        away_state = adjust_state_for_situation(
            away_state, is_home=False, momentum=away_strength.momentum
        )
        
        # Run quantum simulation
        sim_result = self.quantum_sim.simulate_game(
            home_expected=home_expected,
            away_expected=away_expected,
            home_std=home_strength.volatility,
            away_std=away_strength.volatility,
            home_state=home_state,
            away_state=away_state,
            n_sims=n_sims
        )
        
        # Calculate confidence
        spread = abs(sim_result['predicted_spread'])
        confidence = self._calculate_confidence(
            spread=spread,
            spread_std=sim_result['spread_std'],
            home_strength=home_strength,
            away_strength=away_strength
        )
        
        # Get Vegas comparison (NEW!)
        vegas_comparison = self._compare_to_vegas(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            qepc_spread=sim_result['predicted_spread'],
            qepc_home_prob=sim_result['home_win_prob']
        )
        
        return GamePrediction(
            home_team=home_team,
            away_team=away_team,
            game_id=game_id,
            home_win_prob=sim_result['home_win_prob'],
            away_win_prob=sim_result['away_win_prob'],
            home_score=sim_result['home_score_mean'],
            away_score=sim_result['away_score_mean'],
            predicted_spread=sim_result['predicted_spread'],
            predicted_total=sim_result['predicted_total'],
            spread_std=sim_result['spread_std'],
            total_std=sim_result['total_std'],
            confidence=confidence,
            home_expected_raw=home_expected_raw,
            away_expected_raw=away_expected_raw,
            home_court_adjustment=home_court_adj,
            rest_adjustment=rest_adj,
            injury_adjustment=injury_adj_home - injury_adj_away,
            vegas=vegas_comparison,
        )
    
    def _compare_to_vegas(
        self,
        game_id: str,
        home_team: str,
        away_team: str,
        qepc_spread: float,
        qepc_home_prob: float
    ) -> Optional[VegasComparison]:
        """Compare QEPC prediction to Vegas line."""
        if self._vegas_odds is None or self._vegas_odds.empty:
            return None
        
        # Try to find by game_id first
        vegas_line = None
        if game_id:
            match = self._vegas_odds[self._vegas_odds['game_id'] == game_id]
            if not match.empty:
                vegas_line = match.iloc[0]
        
        if vegas_line is None:
            return None
        
        vegas_spread = vegas_line.get('vegas_spread_home')
        vegas_home_prob = vegas_line.get('vegas_implied_home_prob')
        
        # Calculate differences
        spread_diff = None
        if vegas_spread is not None and not pd.isna(vegas_spread):
            spread_diff = qepc_spread - vegas_spread  # Positive = QEPC likes home more
        
        prob_diff = None
        if vegas_home_prob is not None and not pd.isna(vegas_home_prob):
            prob_diff = qepc_home_prob - vegas_home_prob
        
        # Determine if there's an edge
        edge = None
        edge_size = None
        if spread_diff is not None:
            edge_size = abs(spread_diff)
            if spread_diff > 2.0:  # QEPC likes home team more than Vegas
                edge = "HOME"
            elif spread_diff < -2.0:  # QEPC likes away team more than Vegas
                edge = "AWAY"
        
        return VegasComparison(
            vegas_spread=vegas_spread,
            qepc_spread=qepc_spread,
            spread_diff=spread_diff,
            vegas_home_prob=vegas_home_prob,
            qepc_home_prob=qepc_home_prob,
            prob_diff=prob_diff,
            edge=edge,
            edge_size=edge_size,
        )
    
    def predict_games(
        self,
        games: List[Tuple[str, str]],
        game_date: str = None,
        verbose: bool = True
    ) -> List[GamePrediction]:
        """Predict multiple games."""
        predictions = []
        
        for home, away in games:
            pred = self.predict_game(home, away, game_date=game_date)
            if pred is not None:
                predictions.append(pred)
                
                if verbose:
                    winner = pred.predicted_winner
                    prob = max(pred.home_win_prob, pred.away_win_prob)
                    line = f"🏀 {away} @ {home}: {winner} ({prob:.1%}) | Spread: {pred.predicted_spread:+.1f}"
                    
                    # Add Vegas comparison
                    if pred.vegas and pred.vegas.vegas_spread is not None:
                        line += f" (Vegas: {pred.vegas.vegas_spread:+.1f})"
                        if pred.has_vegas_edge:
                            line += f" ⭐ EDGE: {pred.vegas.edge}"
                    
                    print(line)
        
        return predictions
    
    def predict_today(self, verbose: bool = True) -> List[GamePrediction]:
        """Predict all games scheduled for today."""
        today_games = self.loader.load_today_games()
        
        if today_games is None or today_games.empty:
            if verbose:
                print("❌ No games found for today")
            return []
        
        # Extract games
        games = []
        game_ids = []
        
        for _, row in today_games.iterrows():
            home = row.get('Home Team', row.get('home_team', None))
            away = row.get('Away Team', row.get('away_team', None))
            game_id = row.get('game_id', None)
            
            if home and away:
                games.append((home, away))
                game_ids.append(game_id)
        
        if verbose:
            print(f"📅 Found {len(games)} games today")
        
        # Predict with game IDs for Vegas matching
        predictions = []
        for (home, away), game_id in zip(games, game_ids):
            pred = self.predict_game(home, away, game_id=game_id)
            if pred is not None:
                predictions.append(pred)
                
                if verbose:
                    winner = pred.predicted_winner
                    prob = max(pred.home_win_prob, pred.away_win_prob)
                    line = f"🏀 {away} @ {home}: {winner} ({prob:.1%}) | Spread: {pred.predicted_spread:+.1f}"
                    
                    if pred.vegas and pred.vegas.vegas_spread is not None:
                        line += f" (Vegas: {pred.vegas.vegas_spread:+.1f})"
                        if pred.has_vegas_edge:
                            line += f" ⭐ EDGE"
                    
                    print(line)
        
        return predictions
    
    def find_edges(self, predictions: List[GamePrediction] = None) -> List[GamePrediction]:
        """
        Find games where QEPC disagrees with Vegas by 2+ points.
        
        These are potential value bets!
        """
        if predictions is None:
            predictions = self.predict_today(verbose=False)
        
        edges = [p for p in predictions if p.has_vegas_edge]
        
        print(f"\n⭐ FOUND {len(edges)} POTENTIAL EDGES")
        print("=" * 60)
        
        for pred in sorted(edges, key=lambda x: x.vegas.edge_size, reverse=True):
            v = pred.vegas
            print(f"\n{pred.away_team} @ {pred.home_team}")
            print(f"   QEPC Spread: {pred.predicted_spread:+.1f}")
            print(f"   Vegas Spread: {v.vegas_spread:+.1f}")
            print(f"   Difference: {v.spread_diff:+.1f} pts")
            print(f"   Edge: Bet {v.edge} ({v.edge_size:.1f} pts)")
        
        return edges
    
    def _calculate_rest_adjustment(self, home_team: str, away_team: str, game_date: str = None) -> float:
        """Calculate rest-based point adjustment."""
        # Simplified for now
        return 0.0
    
    def _calculate_injury_adjustment(self, home_team: str, away_team: str) -> Tuple[float, float]:
        """Calculate injury-based point adjustments."""
        if self._injuries is None or self._injuries.empty:
            return 0.0, 0.0
        
        home_adj = 0.0
        away_adj = 0.0
        
        for team, is_home in [(home_team, True), (away_team, False)]:
            team_injuries = self._injuries[
                self._injuries['Team'].fillna('').str.contains(team.split()[-1], case=False, na=False) |
                self._injuries['PlayerName'].fillna('').str.contains(team.split()[-1], case=False, na=False)
            ]
            
            if not team_injuries.empty and 'Impact' in team_injuries.columns:
                total_impact = team_injuries['Impact'].sum()
                point_impact = -total_impact * 3
                
                if is_home:
                    home_adj = point_impact
                else:
                    away_adj = point_impact
        
        return home_adj, away_adj
    
    def _calculate_confidence(
        self,
        spread: float,
        spread_std: float,
        home_strength: TeamStrength,
        away_strength: TeamStrength
    ) -> float:
        """Calculate confidence in the prediction."""
        spread_conf = 1 - np.exp(-spread / 10)
        uncertainty_conf = 1 / (1 + spread_std / 10)
        min_games = min(home_strength.games_played, away_strength.games_played)
        sample_conf = min(1.0, min_games / 15)
        avg_volatility = (home_strength.volatility + away_strength.volatility) / 2
        vol_conf = 1 / (1 + avg_volatility / 15)
        
        confidence = (spread_conf * 0.4 + uncertainty_conf * 0.3 + 
                     sample_conf * 0.2 + vol_conf * 0.1)
        
        return np.clip(confidence, 0, 1)
    
    def get_power_rankings(self) -> pd.DataFrame:
        """Get current power rankings."""
        return self.strength_calc.power_rankings()
    
    def predictions_to_dataframe(self, predictions: List[GamePrediction]) -> pd.DataFrame:
        """Convert list of predictions to DataFrame."""
        return pd.DataFrame([p.to_dict() for p in predictions])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_predict(home_team: str, away_team: str, verbose: bool = True) -> Optional[GamePrediction]:
    """Quick prediction for a single game."""
    predictor = GamePredictor()
    predictor.prepare(verbose=False)
    
    pred = predictor.predict_game(home_team, away_team)
    
    if pred and verbose:
        print(f"\n🏀 {away_team} @ {home_team}")
        print(f"   Winner: {pred.predicted_winner} ({max(pred.home_win_prob, pred.away_win_prob):.1%})")
        print(f"   Score: {pred.away_score:.0f} - {pred.home_score:.0f}")
        print(f"   Spread: {pred.predicted_spread:+.1f}")
        print(f"   Total: {pred.predicted_total:.0f}")
        
        if pred.vegas and pred.vegas.vegas_spread is not None:
            print(f"\n   📊 Vegas Comparison:")
            print(f"      Vegas Spread: {pred.vegas.vegas_spread:+.1f}")
            print(f"      Difference: {pred.vegas.spread_diff:+.1f}")
            if pred.has_vegas_edge:
                print(f"      ⭐ EDGE: Bet {pred.vegas.edge} ({pred.vegas.edge_size:.1f} pts)")
    
    return pred


def predict_today(verbose: bool = True) -> List[GamePrediction]:
    """Predict all games scheduled for today."""
    predictor = GamePredictor()
    predictor.prepare(verbose=verbose)
    return predictor.predict_today(verbose=verbose)


def find_edges(verbose: bool = True) -> List[GamePrediction]:
    """Find games where QEPC disagrees with Vegas."""
    predictor = GamePredictor()
    predictor.prepare(verbose=False)
    return predictor.find_edges()
