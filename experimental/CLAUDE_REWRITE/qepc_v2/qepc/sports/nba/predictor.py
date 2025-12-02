"""
QEPC NBA Game Predictor
=======================
The main prediction engine that combines:
- Team strengths with recency weighting
- Quantum state simulation
- Situational adjustments (home court, rest, injuries)
- Entanglement, interference, and tunneling effects
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
class GamePrediction:
    """Complete prediction for a single game."""
    
    # Teams
    home_team: str
    away_team: str
    
    # Win probabilities
    home_win_prob: float
    away_win_prob: float
    
    # Score predictions
    home_score: float
    away_score: float
    predicted_spread: float  # Home - Away (negative = away favored)
    predicted_total: float
    
    # Uncertainty
    spread_std: float
    total_std: float
    
    # Confidence (0-1, higher = more confident)
    confidence: float
    
    # Components (for analysis)
    home_expected_raw: float  # Before adjustments
    away_expected_raw: float
    home_court_adjustment: float
    rest_adjustment: float
    injury_adjustment: float
    
    @property
    def predicted_winner(self) -> str:
        return self.home_team if self.home_win_prob > 0.5 else self.away_team
    
    @property
    def is_high_confidence(self) -> bool:
        return self.confidence > 0.65
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Home_Team': self.home_team,
            'Away_Team': self.away_team,
            'Home_Win_Prob': round(self.home_win_prob, 3),
            'Away_Win_Prob': round(self.away_win_prob, 3),
            'Predicted_Winner': self.predicted_winner,
            'Home_Score': round(self.home_score, 1),
            'Away_Score': round(self.away_score, 1),
            'Spread': round(self.predicted_spread, 1),
            'Total': round(self.predicted_total, 1),
            'Confidence': round(self.confidence, 3),
        }


class GamePredictor:
    """
    Main game prediction engine.
    
    Combines team strengths, quantum simulation, and situational factors.
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
    
    def prepare(self, cutoff_date: str = None, verbose: bool = True):
        """
        Prepare the predictor by loading all required data.
        
        Call this before making predictions.
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
        if self._rest_data is not None and verbose:
            print(f"😴 Loaded rest day data")
        
        if verbose:
            print(f"✅ Ready to predict! ({len(self._strengths)} teams)")
    
    def predict_game(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None,
        n_simulations: int = None
    ) -> Optional[GamePrediction]:
        """
        Predict a single game.
        
        Parameters
        ----------
        home_team : str
            Home team name
        away_team : str  
            Away team name
        game_date : str, optional
            Date of game (for rest calculations)
        n_simulations : int, optional
            Number of Monte Carlo simulations
            
        Returns
        -------
        GamePrediction or None if teams not found
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
        
        # Calculate expected scores (before adjustments)
        game_pace = (home_strength.pace + away_strength.pace) / 2
        
        # Use interference to adjust for matchup
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
        
        # Create quantum states based on volatility
        home_state = create_quantum_state_from_volatility(
            home_strength.volatility / home_strength.ppg  # CV
        )
        away_state = create_quantum_state_from_volatility(
            away_strength.volatility / away_strength.ppg
        )
        
        # Adjust states for situation
        home_state = adjust_state_for_situation(
            home_state,
            is_home=True,
            momentum=home_strength.momentum
        )
        away_state = adjust_state_for_situation(
            away_state,
            is_home=False,
            momentum=away_strength.momentum
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
        
        return GamePrediction(
            home_team=home_team,
            away_team=away_team,
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
        )
    
    def predict_games(
        self,
        games: List[Tuple[str, str]],
        game_date: str = None,
        verbose: bool = True
    ) -> List[GamePrediction]:
        """
        Predict multiple games.
        
        Parameters
        ----------
        games : list of (home_team, away_team) tuples
        game_date : str, optional
        verbose : bool
        
        Returns
        -------
        List of GamePrediction objects
        """
        predictions = []
        
        for home, away in games:
            pred = self.predict_game(home, away, game_date)
            if pred is not None:
                predictions.append(pred)
                
                if verbose:
                    winner = pred.predicted_winner
                    prob = max(pred.home_win_prob, pred.away_win_prob)
                    print(f"🏀 {away} @ {home}: {winner} ({prob:.1%}) | Spread: {pred.predicted_spread:+.1f}")
        
        return predictions
    
    def predict_today(self, verbose: bool = True) -> List[GamePrediction]:
        """Predict all games scheduled for today."""
        today_games = self.loader.load_today_games()
        
        if today_games is None or today_games.empty:
            if verbose:
                print("❌ No games found for today")
            return []
        
        # Extract home/away teams
        games = []
        for _, row in today_games.iterrows():
            home = row.get('Home Team', row.get('HOME_TEAM_NAME', None))
            away = row.get('Away Team', row.get('AWAY_TEAM_NAME', None))
            if home and away:
                games.append((home, away))
        
        if verbose:
            print(f"📅 Found {len(games)} games today")
        
        today_str = datetime.now().strftime('%Y-%m-%d')
        return self.predict_games(games, game_date=today_str, verbose=verbose)
    
    def _calculate_rest_adjustment(
        self,
        home_team: str,
        away_team: str,
        game_date: str = None
    ) -> float:
        """Calculate rest-based point adjustment."""
        if self._rest_data is None or game_date is None:
            return 0.0
        
        # Try to find rest data for both teams
        # (This is simplified - you may need to adjust based on your data structure)
        home_rest = 2  # Default
        away_rest = 2
        home_b2b = False
        away_b2b = False
        
        game_dt = pd.Timestamp(game_date)
        
        home_data = self._rest_data[
            (self._rest_data['Team'] == home_team) & 
            (self._rest_data['gameDate'].dt.date == game_dt.date())
        ]
        if not home_data.empty:
            home_rest = home_data.iloc[0].get('days_since_last_game', 2)
            home_b2b = home_data.iloc[0].get('is_back_to_back', False)
        
        away_data = self._rest_data[
            (self._rest_data['Team'] == away_team) &
            (self._rest_data['gameDate'].dt.date == game_dt.date())
        ]
        if not away_data.empty:
            away_rest = away_data.iloc[0].get('days_since_last_game', 2)
            away_b2b = away_data.iloc[0].get('is_back_to_back', False)
        
        # Calculate adjustment
        rest_diff = home_rest - away_rest
        adjustment = rest_diff * self.config.rest.rest_advantage_per_day
        adjustment = np.clip(adjustment, -self.config.rest.max_rest_advantage, 
                           self.config.rest.max_rest_advantage)
        
        # B2B penalties
        if home_b2b:
            adjustment -= self.config.rest.b2b_penalty
        if away_b2b:
            adjustment += self.config.rest.b2b_penalty
        
        return adjustment
    
    def _calculate_injury_adjustment(
        self,
        home_team: str,
        away_team: str
    ) -> Tuple[float, float]:
        """Calculate injury-based point adjustments."""
        if self._injuries is None or self._injuries.empty:
            return 0.0, 0.0
        
        home_adj = 0.0
        away_adj = 0.0
        
        # Sum up injury impacts for each team
        for team, adj_var in [(home_team, 'home_adj'), (away_team, 'away_adj')]:
            team_injuries = self._injuries[
                (self._injuries['Team'] == team) |
                (self._injuries['PlayerName'].str.contains(team, case=False, na=False))
            ]
            
            if not team_injuries.empty and 'Impact' in team_injuries.columns:
                # Impact is typically 0-1, convert to points
                # A star player (impact=1.0) out = ~4 points
                total_impact = team_injuries['Impact'].sum()
                point_impact = -total_impact * 4  # Negative because injury hurts team
                
                if adj_var == 'home_adj':
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
        """
        Calculate confidence in the prediction.
        
        Higher confidence when:
        - Larger spread
        - Lower spread uncertainty
        - Teams have more games played
        - Lower volatility teams
        """
        # Spread-based confidence
        spread_conf = 1 - np.exp(-spread / 10)  # Approaches 1 for large spreads
        
        # Uncertainty-based
        uncertainty_conf = 1 / (1 + spread_std / 10)
        
        # Sample size confidence
        min_games = min(home_strength.games_played, away_strength.games_played)
        sample_conf = min(1.0, min_games / 15)
        
        # Volatility penalty
        avg_volatility = (home_strength.volatility + away_strength.volatility) / 2
        vol_conf = 1 / (1 + avg_volatility / 15)
        
        # Combine
        confidence = (spread_conf * 0.4 + uncertainty_conf * 0.3 + 
                     sample_conf * 0.2 + vol_conf * 0.1)
        
        return np.clip(confidence, 0, 1)
    
    def get_power_rankings(self) -> pd.DataFrame:
        """Get current power rankings."""
        return self.strength_calc.power_rankings()
    
    def predictions_to_dataframe(
        self,
        predictions: List[GamePrediction]
    ) -> pd.DataFrame:
        """Convert list of predictions to DataFrame."""
        return pd.DataFrame([p.to_dict() for p in predictions])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_predict(
    home_team: str,
    away_team: str,
    verbose: bool = True
) -> Optional[GamePrediction]:
    """
    Quick prediction for a single game.
    
    Usage:
        pred = quick_predict("Boston Celtics", "Los Angeles Lakers")
    """
    predictor = GamePredictor()
    predictor.prepare(verbose=False)
    
    pred = predictor.predict_game(home_team, away_team)
    
    if pred and verbose:
        print(f"\n🏀 {away_team} @ {home_team}")
        print(f"   Winner: {pred.predicted_winner} ({max(pred.home_win_prob, pred.away_win_prob):.1%})")
        print(f"   Score: {pred.away_score:.0f} - {pred.home_score:.0f}")
        print(f"   Spread: {pred.predicted_spread:+.1f}")
        print(f"   Total: {pred.predicted_total:.0f}")
        print(f"   Confidence: {pred.confidence:.1%}")
    
    return pred


def predict_today(verbose: bool = True) -> List[GamePrediction]:
    """
    Predict all games scheduled for today.
    
    Usage:
        predictions = predict_today()
    """
    predictor = GamePredictor()
    predictor.prepare(verbose=verbose)
    return predictor.predict_today(verbose=verbose)
