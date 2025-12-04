"""
QEPC Player Props Predictor
===========================
Quantum-inspired player prop predictions.

Features:
- Monte Carlo simulation with quantum states
- Matchup-based adjustments (opponent defense)
- Pace adjustments (fast/slow games)
- Minutes projections
- Vegas line comparison and edge detection

Props Supported:
- Points
- Rebounds  
- Assists
- 3-Pointers Made
- PRA (Points + Rebounds + Assists)
- Steals + Blocks
- Fantasy Points
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from qepc.sports.nba.props.player_state import (
    PlayerProfile,
    PlayerQuantumState,
    PerformanceState,
    create_player_state,
    simulate_player_game,
)

from qepc.sports.nba.props.matchups import MatchupAnalyzer, MatchupProfile


@dataclass
class PropPrediction:
    """Prediction for a single player prop."""
    player_name: str
    team: str
    opponent: str
    prop_type: str  # 'pts', 'reb', 'ast', 'fg3m', 'pra'
    
    # Prediction
    predicted_value: float
    predicted_std: float
    
    # Distribution
    p10: float  # 10th percentile
    p25: float  
    p50: float  # Median
    p75: float
    p90: float
    
    # Vegas comparison
    vegas_line: Optional[float] = None
    over_prob: Optional[float] = None
    under_prob: Optional[float] = None
    edge: Optional[str] = None  # "OVER" or "UNDER" or None
    edge_size: Optional[float] = None  # How confident we are
    
    # Adjustments applied
    matchup_adj: float = 0.0
    pace_adj: float = 0.0
    home_adj: float = 0.0
    form_adj: float = 0.0
    
    # Confidence
    confidence: float = 0.5
    
    @property
    def has_edge(self) -> bool:
        return self.edge is not None and self.edge_size is not None and self.edge_size > 0.55
    
    def to_dict(self) -> Dict:
        return {
            'Player': self.player_name,
            'Team': self.team,
            'Opponent': self.opponent,
            'Prop': self.prop_type.upper(),
            'Prediction': round(self.predicted_value, 1),
            'Std': round(self.predicted_std, 1),
            'P25': round(self.p25, 1),
            'Median': round(self.p50, 1),
            'P75': round(self.p75, 1),
            'Vegas_Line': self.vegas_line,
            'Over_Prob': f"{self.over_prob:.1%}" if self.over_prob else None,
            'Edge': self.edge,
            'Edge_Size': f"{self.edge_size:.1%}" if self.edge_size else None,
            'Confidence': round(self.confidence, 2),
        }


class PlayerPropsPredictor:
    """
    Main prediction engine for player props.
    
    Uses quantum-inspired Monte Carlo simulation with:
    - Player performance state modeling
    - Opponent defensive adjustments
    - Pace and game environment factors
    - Minutes projections
    """
    
    def __init__(self, data_dir: Path = None):
        """
        Initialize the props predictor.
        
        Parameters
        ----------
        data_dir : Path
            Path to data directory containing player stats
        """
        self.data_dir = data_dir or self._find_data_dir()
        
        # Caches
        self._player_profiles: Dict[str, PlayerProfile] = {}
        self._player_logs: pd.DataFrame = None
        self._player_averages: pd.DataFrame = None
        self._team_defense: Dict[str, Dict] = {}
        
        # Matchup analyzer
        self._matchup_analyzer = MatchupAnalyzer(self.data_dir)
        
        self.n_simulations = 10000
    
    def _find_data_dir(self) -> Path:
        """Find the data directory."""
        current = Path.cwd()
        for p in [current] + list(current.parents)[:5]:
            if (p / "data").exists():
                return p / "data"
        return current / "data"
    
    def prepare(self, verbose: bool = True):
        """Load all required data."""
        if verbose:
            print("🏀 Preparing Player Props Predictor...")
        
        # Load player game logs
        self._load_player_logs(verbose)
        
        # Load player averages
        self._load_player_averages(verbose)
        
        # Load matchup data
        self._matchup_analyzer.load_data(verbose)
        
        # Calculate team defensive ratings
        self._calculate_team_defense(verbose)
        
        # Build player profiles
        self._build_player_profiles(verbose)
        
        if verbose:
            print(f"✅ Ready! {len(self._player_profiles)} players loaded")
    
    def _load_player_logs(self, verbose: bool = True):
        """Load player game logs."""
        paths = [
            self.data_dir / "raw" / "Player_Game_Logs_All_Seasons.csv",
            self.data_dir / "raw" / "Player_Game_Logs.csv",
            self.data_dir / "props" / "Player_Props_Full_Logs.csv",
        ]
        
        for path in paths:
            if path.exists():
                try:
                    self._player_logs = pd.read_csv(path)
                    if verbose:
                        print(f"✅ Loaded player logs: {len(self._player_logs):,} rows")
                    return
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error loading {path.name}: {e}")
        
        if verbose:
            print("❌ Player logs not found")
    
    def _load_player_averages(self, verbose: bool = True):
        """Load player season averages."""
        paths = [
            self.data_dir / "props" / "Player_Season_Averages.csv",
            self.data_dir / "raw" / "Player_Season_Averages.csv",
        ]
        
        for path in paths:
            if path.exists():
                try:
                    self._player_averages = pd.read_csv(path)
                    if verbose:
                        print(f"✅ Loaded player averages: {len(self._player_averages):,} players")
                    return
                except Exception as e:
                    if verbose:
                        print(f"⚠️  Error loading {path.name}: {e}")
        
        if verbose:
            print("⚠️  Player averages not found - will calculate from logs")
    
    def _calculate_team_defense(self, verbose: bool = True):
        """Calculate team defensive ratings vs each stat category."""
        if self._player_logs is None:
            return
        
        # Group by opponent team and calculate average stats allowed
        # This tells us how good each team is at defending each stat
        
        # For now, use simple league-relative approach
        # We'll enhance this with actual opponent data later
        self._team_defense = {}
        
        if verbose:
            print("📊 Calculated team defense ratings")
    
    def _build_player_profiles(self, verbose: bool = True):
        """Build player profiles from available data."""
        if self._player_averages is not None:
            self._build_from_averages()
        elif self._player_logs is not None:
            self._build_from_logs()
        else:
            if verbose:
                print("❌ No player data available")
    
    def _build_from_averages(self):
        """Build profiles from season averages file."""
        df = self._player_averages
        
        for _, row in df.iterrows():
            player_name = row.get('Player', row.get('PLAYER_NAME', ''))
            if not player_name:
                continue
            
            profile = PlayerProfile(
                player_id=str(row.get('PLAYER_ID', '')),
                player_name=player_name,
                team=row.get('Team', row.get('TEAM_ABBREVIATION', '')),
                position=row.get('Position', row.get('POS', '')),
                games_played=int(row.get('GP', row.get('Games', 0))),
                minutes_avg=float(row.get('MIN', row.get('MPG', 0))),
                pts_avg=float(row.get('PTS', row.get('PPG', 0))),
                reb_avg=float(row.get('REB', row.get('RPG', 0))),
                ast_avg=float(row.get('AST', row.get('APG', 0))),
                stl_avg=float(row.get('STL', row.get('SPG', 0))),
                blk_avg=float(row.get('BLK', row.get('BPG', 0))),
                tov_avg=float(row.get('TOV', row.get('TOPG', 0))),
                fg3m_avg=float(row.get('FG3M', row.get('3PM', 0))),
                pts_std=float(row.get('PTS_STD', row.get('pts_std', row.get('PTS', 0) * 0.3))),
                reb_std=float(row.get('REB_STD', row.get('reb_std', row.get('REB', 0) * 0.4))),
                ast_std=float(row.get('AST_STD', row.get('ast_std', row.get('AST', 0) * 0.4))),
                fg3m_std=float(row.get('FG3M_STD', row.get('fg3m_std', row.get('FG3M', 0) * 0.5))),
            )
            
            # Calculate CV if we have std
            if profile.pts_avg > 0:
                profile.pts_cv = profile.pts_std / profile.pts_avg
            if profile.reb_avg > 0:
                profile.reb_cv = profile.reb_std / profile.reb_avg
            if profile.ast_avg > 0:
                profile.ast_cv = profile.ast_std / profile.ast_avg
            
            # Calculate per-minute rates
            if profile.minutes_avg > 0:
                profile.pts_per_min = profile.pts_avg / profile.minutes_avg
                profile.reb_per_min = profile.reb_avg / profile.minutes_avg
                profile.ast_per_min = profile.ast_avg / profile.minutes_avg
            
            # Recent form (if available)
            profile.pts_l5 = float(row.get('PTS_L5', row.get('pts_last_5', profile.pts_avg)))
            profile.reb_l5 = float(row.get('REB_L5', row.get('reb_last_5', profile.reb_avg)))
            profile.ast_l5 = float(row.get('AST_L5', row.get('ast_last_5', profile.ast_avg)))
            
            self._player_profiles[player_name.lower()] = profile
    
    def _build_from_logs(self):
        """Build profiles from game logs."""
        df = self._player_logs
        
        # Get player name column
        name_col = None
        for col in ['PLAYER_NAME', 'Player', 'player_name']:
            if col in df.columns:
                name_col = col
                break
        
        if name_col is None:
            return
        
        # Group by player and calculate averages
        for player_name, games in df.groupby(name_col):
            if len(games) < 5:  # Need minimum games
                continue
            
            profile = PlayerProfile(
                player_id=str(games.iloc[0].get('PLAYER_ID', '')),
                player_name=player_name,
                team=str(games.iloc[-1].get('TEAM_ABBREVIATION', '')),  # Most recent team
                position='',
                games_played=len(games),
                minutes_avg=games['MIN'].mean() if 'MIN' in games.columns else 0,
                pts_avg=games['PTS'].mean() if 'PTS' in games.columns else 0,
                reb_avg=games['REB'].mean() if 'REB' in games.columns else 0,
                ast_avg=games['AST'].mean() if 'AST' in games.columns else 0,
                stl_avg=games['STL'].mean() if 'STL' in games.columns else 0,
                blk_avg=games['BLK'].mean() if 'BLK' in games.columns else 0,
                tov_avg=games['TOV'].mean() if 'TOV' in games.columns else 0,
                fg3m_avg=games['FG3M'].mean() if 'FG3M' in games.columns else 0,
                pts_std=games['PTS'].std() if 'PTS' in games.columns else 0,
                reb_std=games['REB'].std() if 'REB' in games.columns else 0,
                ast_std=games['AST'].std() if 'AST' in games.columns else 0,
                fg3m_std=games['FG3M'].std() if 'FG3M' in games.columns else 0,
            )
            
            # Calculate CV
            if profile.pts_avg > 0:
                profile.pts_cv = profile.pts_std / profile.pts_avg
            
            # Recent form (last 5 games)
            recent = games.tail(5)
            profile.pts_l5 = recent['PTS'].mean() if 'PTS' in recent.columns else profile.pts_avg
            profile.reb_l5 = recent['REB'].mean() if 'REB' in recent.columns else profile.reb_avg
            profile.ast_l5 = recent['AST'].mean() if 'AST' in recent.columns else profile.ast_avg
            
            self._player_profiles[player_name.lower()] = profile
    
    def get_player(self, name: str) -> Optional[PlayerProfile]:
        """Get a player profile by name (case-insensitive)."""
        return self._player_profiles.get(name.lower())
    
    def search_players(self, query: str) -> List[str]:
        """Search for players by partial name match."""
        query = query.lower()
        matches = [
            profile.player_name 
            for name, profile in self._player_profiles.items() 
            if query in name
        ]
        return sorted(matches)
    
    def predict_prop(
        self,
        player_name: str,
        prop_type: str = 'pts',
        opponent: str = None,
        is_home: bool = True,
        vegas_line: float = None,
        minutes_projection: float = None,
    ) -> Optional[PropPrediction]:
        """
        Predict a player prop.
        
        Parameters
        ----------
        player_name : str
            Player's name
        prop_type : str
            Type of prop ('pts', 'reb', 'ast', 'fg3m', 'pra')
        opponent : str
            Opponent team name (for matchup adjustment)
        is_home : bool
            Whether player is at home
        vegas_line : float
            Vegas line to compare against
        minutes_projection : float
            Expected minutes (if different from average)
        
        Returns
        -------
        PropPrediction with full analysis
        """
        # Get player profile
        player = self.get_player(player_name)
        if player is None:
            print(f"⚠️  Player not found: {player_name}")
            print(f"   Try: {', '.join(self.search_players(player_name.split()[0])[:5])}")
            return None
        
        # Calculate adjustments using matchup analyzer
        matchup_mod = self._get_matchup_modifier(opponent, prop_type, player.position) if opponent else 1.0
        pace_mod = self._get_pace_modifier(player.team, opponent) if opponent else 1.0
        minutes_factor = minutes_projection / player.minutes_avg if minutes_projection and player.minutes_avg > 0 else 1.0
        
        # Create quantum state
        state = create_player_state(
            player=player,
            is_home=is_home,
            momentum=player.momentum,
            matchup_modifier=matchup_mod,
            minutes_factor=minutes_factor,
        )
        
        # Run simulation
        sim_result = simulate_player_game(state, stat=prop_type, n_sims=self.n_simulations)
        
        # Apply pace adjustment
        adjusted_mean = sim_result['mean'] * pace_mod
        adjusted_std = sim_result['std'] * pace_mod
        
        # Calculate over/under probabilities if Vegas line provided
        over_prob = None
        under_prob = None
        edge = None
        edge_size = None
        
        if vegas_line is not None:
            distribution = sim_result['distribution'] * pace_mod
            over_prob = np.mean(distribution > vegas_line)
            under_prob = 1 - over_prob
            
            # Determine edge (need >55% to have edge, accounting for juice)
            if over_prob > 0.55:
                edge = "OVER"
                edge_size = over_prob
            elif under_prob > 0.55:
                edge = "UNDER"
                edge_size = under_prob
        
        # Calculate confidence
        confidence = self._calculate_confidence(player, prop_type, sim_result)
        
        return PropPrediction(
            player_name=player.player_name,
            team=player.team,
            opponent=opponent or "Unknown",
            prop_type=prop_type,
            predicted_value=adjusted_mean,
            predicted_std=adjusted_std,
            p10=sim_result['p10'] * pace_mod,
            p25=sim_result['p25'] * pace_mod,
            p50=sim_result['median'] * pace_mod,
            p75=sim_result['p75'] * pace_mod,
            p90=sim_result['p90'] * pace_mod,
            vegas_line=vegas_line,
            over_prob=over_prob,
            under_prob=under_prob,
            edge=edge,
            edge_size=edge_size,
            matchup_adj=(matchup_mod - 1) * 100,
            pace_adj=(pace_mod - 1) * 100,
            home_adj=3.0 if is_home else 0.0,
            form_adj=player.momentum * 10,
            confidence=confidence,
        )
    
    def _get_matchup_modifier(self, opponent: str, prop_type: str, position: str = None) -> float:
        """
        Get matchup modifier based on opponent defense.
        
        Returns >1 if opponent is bad at defending this stat,
        <1 if opponent is good at defending.
        """
        return self._matchup_analyzer.get_matchup_modifier(opponent, prop_type, position)
    
    def _get_pace_modifier(self, player_team: str, opponent: str) -> float:
        """
        Get pace modifier based on game environment.
        
        Fast-paced games = more possessions = more stats.
        """
        return self._matchup_analyzer.get_pace_modifier(player_team, opponent)
    
    def _calculate_confidence(
        self, 
        player: PlayerProfile, 
        prop_type: str,
        sim_result: Dict
    ) -> float:
        """Calculate confidence in the prediction."""
        # Factors that increase confidence:
        # - More games played
        # - Lower volatility
        # - Tighter distribution
        
        games_conf = min(1.0, player.games_played / 20)
        
        cv_map = {'pts': player.pts_cv, 'reb': player.reb_cv, 'ast': player.ast_cv}
        cv = cv_map.get(prop_type, 0.3)
        vol_conf = 1 / (1 + cv)
        
        spread = sim_result['p75'] - sim_result['p25']
        mean = sim_result['mean']
        spread_conf = 1 / (1 + spread / mean) if mean > 0 else 0.5
        
        return np.clip(games_conf * 0.3 + vol_conf * 0.4 + spread_conf * 0.3, 0, 1)
    
    def predict_slate(
        self,
        players: List[Tuple[str, str, float]],  # (player_name, prop_type, vegas_line)
        verbose: bool = True,
    ) -> List[PropPrediction]:
        """
        Predict a full slate of player props.
        
        Parameters
        ----------
        players : List[Tuple]
            List of (player_name, prop_type, vegas_line) tuples
        verbose : bool
            Print results
        
        Returns
        -------
        List of PropPrediction objects
        """
        predictions = []
        
        for player_name, prop_type, vegas_line in players:
            pred = self.predict_prop(
                player_name=player_name,
                prop_type=prop_type,
                vegas_line=vegas_line,
            )
            
            if pred is not None:
                predictions.append(pred)
                
                if verbose:
                    edge_marker = f"⭐ {pred.edge}" if pred.has_edge else ""
                    print(f"🏀 {pred.player_name} {prop_type.upper()}: {pred.predicted_value:.1f} "
                          f"(Line: {vegas_line}) {edge_marker}")
        
        return predictions
    
    def find_edges(
        self,
        predictions: List[PropPrediction] = None,
        min_edge: float = 0.55,
    ) -> List[PropPrediction]:
        """Find props where we have an edge vs Vegas."""
        if predictions is None:
            return []
        
        edges = [p for p in predictions if p.edge_size and p.edge_size >= min_edge]
        
        # Sort by edge size
        edges.sort(key=lambda x: x.edge_size or 0, reverse=True)
        
        return edges
    
    def predictions_to_dataframe(self, predictions: List[PropPrediction]) -> pd.DataFrame:
        """Convert predictions to DataFrame."""
        return pd.DataFrame([p.to_dict() for p in predictions])


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_prop(
    player_name: str,
    prop_type: str = 'pts',
    vegas_line: float = None,
    verbose: bool = True,
) -> Optional[PropPrediction]:
    """
    Quick prediction for a single player prop.
    
    Usage:
        pred = quick_prop("LeBron James", "pts", vegas_line=25.5)
    """
    predictor = PlayerPropsPredictor()
    predictor.prepare(verbose=False)
    
    pred = predictor.predict_prop(
        player_name=player_name,
        prop_type=prop_type,
        vegas_line=vegas_line,
    )
    
    if pred and verbose:
        print(f"\n🏀 {pred.player_name} - {prop_type.upper()}")
        print("=" * 50)
        print(f"""
📊 PREDICTION
   Expected: {pred.predicted_value:.1f} ± {pred.predicted_std:.1f}
   
📈 DISTRIBUTION
   Floor (P10): {pred.p10:.1f}
   Low (P25): {pred.p25:.1f}
   Median: {pred.p50:.1f}
   High (P75): {pred.p75:.1f}
   Ceiling (P90): {pred.p90:.1f}
""")
        
        if pred.vegas_line:
            print(f"""💰 VS VEGAS (Line: {pred.vegas_line})
   Over: {pred.over_prob:.1%}
   Under: {pred.under_prob:.1%}
""")
            if pred.has_edge:
                print(f"   ⭐ EDGE: {pred.edge} ({pred.edge_size:.1%} probability)")
    
    return pred
