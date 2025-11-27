"""
QEPC Player Props Engine v1.0
=============================

A complete system for predicting NBA player stat lines.

Supported Props:
- PTS (Points)
- REB (Rebounds)  
- AST (Assists)
- 3PM (Three Pointers Made)
- PRA (Points + Rebounds + Assists)
- PR, PA, RA (combo props)
- STL, BLK, TOV

Features:
- Recency-weighted player averages
- Opponent defensive adjustments
- Home/away splits
- Minutes-based scaling
- Probability distributions for over/under
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings

# Try scipy for distributions, fall back to numpy if not available
try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    warnings.warn("scipy not installed - using approximations for probabilities")


# =============================================================================
# CONFIGURATION
# =============================================================================

class PropsConfig:
    """Configuration for the props engine."""
    
    # Recency weighting
    HALF_LIFE_GAMES = 10          # Games for 50% weight decay
    MIN_GAMES_REQUIRED = 3        # Minimum games for prediction
    MAX_GAMES_LOOKBACK = 50       # Maximum games to consider
    
    # Prop types and their stat columns
    STAT_MAPPING = {
        'PTS': 'points',
        'REB': 'reboundsTotal',
        'AST': 'assists',
        '3PM': 'threePointersMade',
        'STL': 'steals',
        'BLK': 'blocks',
        'TOV': 'turnovers',
        'MIN': 'numMinutes',
        'FGA': 'fieldGoalsAttempted',
        'FGM': 'fieldGoalsMade',
        'FTA': 'freeThrowsAttempted',
        'FTM': 'freeThrowsMade',
        '3PA': 'threePointersAttempted',
        'OREB': 'reboundsOffensive',
        'DREB': 'reboundsDefensive',
    }
    
    # How much opponent defense affects each stat (0-1 scale)
    OPPONENT_IMPACT = {
        'PTS': 0.12,
        'REB': 0.08,
        'AST': 0.10,
        '3PM': 0.15,
        'STL': 0.05,
        'BLK': 0.05,
        'TOV': 0.06,
    }
    
    # Home/away adjustment strength
    HOME_AWAY_WEIGHT = 0.6  # How much to trust home/away splits
    
    # Trend adjustment
    TREND_WEIGHT = 0.25  # How much recent trend affects prediction
    TREND_GAMES = 5      # Games to consider for trend


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PropPrediction:
    """A single prop prediction with probabilities."""
    player_name: str
    team: str
    opponent: str
    game_date: Optional[str]
    prop_type: str
    projection: float
    std_dev: float
    floor: float          # 10th percentile
    ceiling: float        # 90th percentile
    games_analyzed: int
    avg_minutes: float
    confidence: str       # HIGH, MEDIUM, LOW
    
    def over_prob(self, line: float) -> float:
        """Probability of exceeding the line."""
        if self.std_dev <= 0:
            return 0.5 if self.projection > line else 0.5
        
        if HAS_SCIPY:
            z = (line - self.projection) / self.std_dev
            return 1 - stats.norm.cdf(z)
        else:
            # Approximation without scipy
            z = (line - self.projection) / self.std_dev
            return 1 / (1 + np.exp(1.7 * z))  # Logistic approximation
    
    def under_prob(self, line: float) -> float:
        """Probability of staying under the line."""
        return 1 - self.over_prob(line)
    
    def edge(self, line: float, side: str = 'over', vig_breakeven: float = 0.524) -> float:
        """
        Calculate edge vs a betting line.
        
        Parameters
        ----------
        line : float
            The betting line
        side : str
            'over' or 'under'
        vig_breakeven : float
            Breakeven probability (0.524 for -110 odds)
        
        Returns
        -------
        Edge as decimal (0.05 = 5% edge)
        """
        prob = self.over_prob(line) if side.lower() == 'over' else self.under_prob(line)
        return prob - vig_breakeven
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'player_name': self.player_name,
            'team': self.team,
            'opponent': self.opponent,
            'game_date': self.game_date,
            'prop_type': self.prop_type,
            'projection': self.projection,
            'std_dev': self.std_dev,
            'floor': self.floor,
            'ceiling': self.ceiling,
            'games_analyzed': self.games_analyzed,
            'avg_minutes': self.avg_minutes,
            'confidence': self.confidence,
        }


@dataclass
class PlayerStats:
    """Aggregated statistics for a player."""
    name: str
    team: str
    games_played: int
    avg_minutes: float
    
    # Weighted averages for each stat
    averages: Dict[str, float] = field(default_factory=dict)
    
    # Standard deviations
    std_devs: Dict[str, float] = field(default_factory=dict)
    
    # Recent trend (last N games vs overall)
    trends: Dict[str, float] = field(default_factory=dict)
    
    # Home/away splits: {stat: (home_avg, away_avg)}
    splits: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    
    # Per-minute rates
    per_minute: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# MAIN ENGINE
# =============================================================================

class PlayerPropsEngine:
    """
    Main engine for player prop predictions.
    
    Usage:
        engine = PlayerPropsEngine("path/to/PlayerStatistics.csv")
        engine.load_data()
        
        # Single prediction
        pred = engine.predict("LeBron James", "PTS", opponent="Warriors")
        print(f"Projection: {pred.projection}, Over 25.5: {pred.over_prob(25.5):.1%}")
        
        # All props for a player
        props = engine.predict_all_props("LeBron James", opponent="Warriors")
    """
    
    def __init__(self, data_path: Union[str, Path, None] = None):
        """
        Initialize the engine.
        
        Parameters
        ----------
        data_path : str or Path, optional
            Path to PlayerStatistics.csv
        """
        self.data_path = Path(data_path) if data_path else None
        self.config = PropsConfig()
        self.raw_data: Optional[pd.DataFrame] = None
        self.player_stats: Dict[str, PlayerStats] = {}
        self.opponent_defense: Dict[str, Dict[str, float]] = {}
        self._loaded = False
    
    def _find_data_file(self) -> Path:
        """Auto-detect player statistics file."""
        search_paths = [
            Path("data/raw/PlayerStatistics.csv"),
            Path("data/PlayerStatistics.csv"),
            Path("../data/raw/PlayerStatistics.csv"),
            Path("../../data/raw/PlayerStatistics.csv"),
            Path.cwd() / "data" / "raw" / "PlayerStatistics.csv",
        ]
        
        for p in search_paths:
            if p.exists():
                return p
        
        raise FileNotFoundError(
            "Could not find PlayerStatistics.csv.\n"
            f"Searched: {[str(p) for p in search_paths]}\n"
            "Please provide the path explicitly."
        )
    
    def load_data(self, cutoff_date: Optional[str] = None, verbose: bool = True) -> None:
        """
        Load and process player statistics.
        
        Parameters
        ----------
        cutoff_date : str, optional
            Only use games before this date (YYYY-MM-DD format).
            Use this for backtesting to avoid lookahead bias.
        verbose : bool
            Print progress messages
        """
        if self.data_path is None:
            self.data_path = self._find_data_file()
        
        if verbose:
            print(f"[Props] Loading {self.data_path}...")
        
        # Load raw data
        df = pd.read_csv(self.data_path)
        
        # Standardize columns
        df = self._standardize_columns(df)
        
        # Parse dates
        if 'gameDate' in df.columns:
            df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')
        
        # Apply cutoff
        if cutoff_date:
            cutoff = pd.Timestamp(cutoff_date)
            before = len(df)
            df = df[df['gameDate'] < cutoff].copy()
            if verbose:
                print(f"[Props] Filtered to games before {cutoff_date}: {before} → {len(df)}")
        
        self.raw_data = df
        
        # Build player profiles
        self._build_player_stats(verbose)
        
        # Build opponent defensive ratings
        self._build_opponent_defense(verbose)
        
        self._loaded = True
        
        if verbose:
            print(f"[Props] Ready! {len(self.player_stats)} players loaded.")
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and create derived columns."""
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        # Player name
        if 'playerName' not in df.columns:
            if 'firstName' in df.columns and 'lastName' in df.columns:
                df['playerName'] = df['firstName'].astype(str).str.strip() + ' ' + df['lastName'].astype(str).str.strip()
            else:
                raise ValueError("Cannot find player name columns")
        
        # Team
        for col in ['playerteamName', 'teamName', 'team']:
            if col in df.columns:
                df['team'] = df[col].astype(str).str.strip()
                break
        
        # Opponent
        for col in ['opponentTeamName', 'opponentteamName', 'opponent']:
            if col in df.columns:
                df['opponent'] = df[col].astype(str).str.strip()
                break
        
        # Home indicator
        if 'home' in df.columns:
            df['is_home'] = df['home'].astype(int) == 1
        else:
            df['is_home'] = True
        
        # Convert stat columns to numeric
        for stat, col in self.config.STAT_MAPPING.items():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _calculate_weights(self, n_games: int) -> np.ndarray:
        """Calculate recency weights for a sequence of games (newest first)."""
        indices = np.arange(n_games)
        weights = 0.5 ** (indices / self.config.HALF_LIFE_GAMES)
        return weights / weights.sum()  # Normalize
    
    def _weighted_stats(self, values: pd.Series, weights: np.ndarray) -> Tuple[float, float]:
        """Calculate weighted mean and std dev."""
        values = values.values
        
        # Weighted mean
        mean = np.sum(values * weights)
        
        # Weighted variance
        variance = np.sum(weights * (values - mean) ** 2)
        std = np.sqrt(variance)
        
        return mean, std
    
    def _build_player_stats(self, verbose: bool = True) -> None:
        """Build statistical profiles for all players."""
        df = self.raw_data
        if df is None:
            return
        
        players_built = 0
        
        for player_name in df['playerName'].unique():
            # Get player's games, sorted by date (newest first)
            player_df = df[df['playerName'] == player_name].copy()
            player_df = player_df.sort_values('gameDate', ascending=False)
            
            # Limit lookback
            player_df = player_df.head(self.config.MAX_GAMES_LOOKBACK)
            
            n_games = len(player_df)
            if n_games < self.config.MIN_GAMES_REQUIRED:
                continue
            
            # Get most recent team
            team = player_df.iloc[0]['team']
            
            # Calculate weights
            weights = self._calculate_weights(n_games)
            
            # Build stats
            averages = {}
            std_devs = {}
            per_minute = {}
            
            # Get minutes for per-minute calculations
            min_col = self.config.STAT_MAPPING['MIN']
            if min_col in player_df.columns:
                minutes = player_df[min_col].values
                avg_minutes = np.sum(minutes * weights)
            else:
                avg_minutes = 30.0  # Assume 30 mpg if not available
                minutes = np.full(n_games, 30.0)
            
            for stat, col in self.config.STAT_MAPPING.items():
                if col not in player_df.columns:
                    continue
                if stat == 'MIN':
                    continue
                
                values = player_df[col].fillna(0)
                mean, std = self._weighted_stats(values, weights)
                
                averages[stat] = mean
                std_devs[stat] = std
                
                # Per-minute rate
                if avg_minutes > 0:
                    per_minute[stat] = mean / avg_minutes
            
            # Recent trend (last N games vs weighted average)
            trends = {}
            trend_games = min(self.config.TREND_GAMES, n_games)
            recent_df = player_df.head(trend_games)
            
            for stat, col in self.config.STAT_MAPPING.items():
                if col not in player_df.columns or stat == 'MIN':
                    continue
                if stat not in averages or averages[stat] == 0:
                    continue
                
                recent_avg = recent_df[col].mean()
                trends[stat] = (recent_avg - averages[stat]) / averages[stat]
            
            # Home/away splits
            splits = {}
            home_df = player_df[player_df['is_home'] == True]
            away_df = player_df[player_df['is_home'] == False]
            
            for stat, col in self.config.STAT_MAPPING.items():
                if col not in player_df.columns or stat == 'MIN':
                    continue
                
                home_avg = home_df[col].mean() if len(home_df) >= 2 else averages.get(stat, 0)
                away_avg = away_df[col].mean() if len(away_df) >= 2 else averages.get(stat, 0)
                splits[stat] = (home_avg, away_avg)
            
            # Store profile
            self.player_stats[player_name] = PlayerStats(
                name=player_name,
                team=team,
                games_played=n_games,
                avg_minutes=avg_minutes,
                averages=averages,
                std_devs=std_devs,
                trends=trends,
                splits=splits,
                per_minute=per_minute,
            )
            
            players_built += 1
        
        if verbose:
            print(f"[Props] Built profiles for {players_built} players")
    
    def _build_opponent_defense(self, verbose: bool = True) -> None:
        """Build opponent defensive ratings by stat."""
        df = self.raw_data
        if df is None:
            return
        
        # Group by opponent and calculate average stats allowed
        for stat, col in self.config.STAT_MAPPING.items():
            if col not in df.columns or stat == 'MIN':
                continue
            
            opp_avg = df.groupby('opponent')[col].mean()
            league_avg = df[col].mean()
            
            for opp, avg in opp_avg.items():
                if opp not in self.opponent_defense:
                    self.opponent_defense[opp] = {}
                
                # Ratio vs league average (>1 = allows more)
                if league_avg > 0:
                    self.opponent_defense[opp][stat] = avg / league_avg
                else:
                    self.opponent_defense[opp][stat] = 1.0
        
        if verbose:
            print(f"[Props] Built defensive ratings for {len(self.opponent_defense)} teams")
    

    def set_opponent_defense_from_df(
        self,
        defense_df: pd.DataFrame,
        team_col: str = "Team",
        value_cols: Optional[Dict[str, str]] = None,
        verbose: bool = True,
    ) -> None:
        """
        Override or supplement opponent defensive ratings from an external DataFrame.

        Parameters
        ----------
        defense_df : pd.DataFrame
            Table containing one row per opponent team and defensive effects
            for various stats. Values are interpreted as *multipliers* where
            1.0 is league average, >1 means the opponent allows more than
            average, and <1 means they allow less.
        team_col : str, default "Team"
            Column in `defense_df` that contains the team name matching the
            `opponent` field in PlayerStatistics.
        value_cols : dict, optional
            Mapping from prop type (e.g. "PTS", "REB") to the column name in
            `defense_df`. If None, this method will look for columns whose
            names match the prop types directly.

        Notes
        -----
        - This method replaces any existing entries in `self.opponent_defense`
          for teams found in `defense_df`.
        - It is safe to call this multiple times (e.g., after computing more
          advanced QEPC team defensive models).
        """
        if team_col not in defense_df.columns:
            raise ValueError(f"Team column {team_col!r} not found in defense_df.")

        if value_cols is None:
            # Assume columns are named exactly by prop type if present
            value_cols = {
                prop: prop for prop in self.config.STAT_MAPPING.keys()
                if prop in defense_df.columns
            }

        updated = 0
        for _, row in defense_df.iterrows():
            team_name = str(row[team_col]).strip()
            if not team_name:
                continue

            if team_name not in self.opponent_defense:
                self.opponent_defense[team_name] = {}

            for prop, col in value_cols.items():
                if col not in defense_df.columns:
                    continue
                val = row[col]
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if val <= 0:
                    continue
                self.opponent_defense[team_name][prop] = val
                updated += 1

        if verbose:
            print(
                f"[Props] set_opponent_defense_from_df: "
                f"updated {updated} defensive multipliers "
                f"for {defense_df[team_col].nunique()} teams."
            )

    def _apply_context_adjustments(
        self,
        player_name: str,
        prop_type: str,
        base: float,
        multiplier: float,
        std: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """
        Apply contextual adjustments (role, usage, injuries, etc.) to a projection.

        This is intentionally conservative by default and acts as a hook for
        QEPC-style roster/script context. The `context` dict can include keys
        such as:

        - "minutes_multiplier": float
        - "usage_multiplier": float
        - "is_starter": bool
        - "teammates_out": List[str]

        The default implementation applies only scalar multipliers if provided.
        More advanced logic (e.g., script-based adjustments) can be layered
        on top later without changing the engine API.
        """
        if not context:
            return multiplier, std

        # Simple scalar adjustments for now
        minutes_mult = float(context.get("minutes_multiplier", 1.0))
        usage_mult = float(context.get("usage_multiplier", 1.0))

        # Combine the two multiplicatively
        total_mult = minutes_mult * usage_mult
        if total_mult <= 0:
            total_mult = 1.0

        multiplier *= total_mult

        # Inflate uncertainty when we apply a sizable context bump
        if abs(total_mult - 1.0) > 0.15:
            std *= 1.1

        return multiplier, std


    def predict(
        self,
        player_name: str,
        prop_type: str,
        opponent: Optional[str] = None,
        is_home: bool = True,
        projected_minutes: Optional[float] = None,
        game_date: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PropPrediction]:
        """
        Generate a prop prediction.
        
        Parameters
        ----------
        player_name : str
            Player's full name
        prop_type : str
            Stat type: PTS, REB, AST, 3PM, PRA, PR, PA, RA, STL, BLK, TOV
        opponent : str, optional
            Opponent team name
        is_home : bool
            Whether player's team is home
        projected_minutes : float, optional
            Override for expected minutes
        game_date : str, optional
            Date of game (for reference)
        
        Returns
        -------
        PropPrediction or None if player not found
        """
        if not self._loaded:
            raise RuntimeError("Call load_data() first!")
        
        # Handle combo props
        if prop_type.upper() == 'PRA':
            return self._predict_combo(player_name, ['PTS', 'REB', 'AST'], 
                                      opponent, is_home, projected_minutes, game_date)
        elif prop_type.upper() == 'PR':
            return self._predict_combo(player_name, ['PTS', 'REB'],
                                      opponent, is_home, projected_minutes, game_date)
        elif prop_type.upper() == 'PA':
            return self._predict_combo(player_name, ['PTS', 'AST'],
                                      opponent, is_home, projected_minutes, game_date)
        elif prop_type.upper() == 'RA':
            return self._predict_combo(player_name, ['REB', 'AST'],
                                      opponent, is_home, projected_minutes, game_date)
        
        # Find player
        stats = self._find_player(player_name)
        if stats is None:
            return None
        
        prop_type = prop_type.upper()
        if prop_type not in stats.averages:
            return None
        
        # Base projection
        base = stats.averages[prop_type]
        std = stats.std_devs.get(prop_type, base * 0.35)
        
        # === ADJUSTMENTS ===
        multiplier = 1.0
        
        # 1. Home/away adjustment
        if prop_type in stats.splits:
            home_avg, away_avg = stats.splits[prop_type]
            expected = home_avg if is_home else away_avg
            if base > 0:
                split_factor = expected / base
                # Blend with overall average
                multiplier *= (1 - self.config.HOME_AWAY_WEIGHT) + self.config.HOME_AWAY_WEIGHT * split_factor
        
        # 2. Recent trend
        if prop_type in stats.trends:
            trend = stats.trends[prop_type]
            # Apply partial trend (regress to mean)
            multiplier *= (1 + trend * self.config.TREND_WEIGHT)
        
        # 3. Opponent adjustment
        if opponent and opponent in self.opponent_defense:
            opp_factor = self.opponent_defense[opponent].get(prop_type, 1.0)
            impact = self.config.OPPONENT_IMPACT.get(prop_type, 0.1)
            # Blend opponent factor with neutral
            opp_adj = 1 + (opp_factor - 1) * impact
            multiplier *= opp_adj
        
        # 4. Minutes adjustment
        if projected_minutes and stats.avg_minutes > 0:
            min_factor = projected_minutes / stats.avg_minutes
            multiplier *= min_factor
        
        # 5. Contextual adjustments (role, usage, injuries, etc.)
        multiplier, std = self._apply_context_adjustments(
            player_name=player_name,
            prop_type=prop_type,
            base=base,
            multiplier=multiplier,
            std=std,
            context=context,
        )

        # Final projection
        projection = base * multiplier
        
        # Adjust std for uncertainty
        if stats.games_played < 10:
            std *= 1.25  # More uncertain with small sample
        
        # Confidence level
        cv = std / projection if projection > 0 else 1.0
        if stats.games_played >= 20 and cv < 0.35:
            confidence = 'HIGH'
        elif stats.games_played >= 10 and cv < 0.50:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        # Floor/ceiling (10th/90th percentile)
        floor = max(0, projection - 1.28 * std)
        ceiling = projection + 1.28 * std
        
        return PropPrediction(
            player_name=stats.name,
            team=stats.team,
            opponent=opponent or "Unknown",
            game_date=game_date,
            prop_type=prop_type,
            projection=round(projection, 1),
            std_dev=round(std, 2),
            floor=round(floor, 1),
            ceiling=round(ceiling, 1),
            games_analyzed=stats.games_played,
            avg_minutes=round(stats.avg_minutes, 1),
            confidence=confidence,
        )
    
    def _find_player(self, name: str) -> Optional[PlayerStats]:
        """Find player by exact or fuzzy match."""
        # Exact match
        if name in self.player_stats:
            return self.player_stats[name]
        
        # Case-insensitive
        name_lower = name.lower()
        for player_name, stats in self.player_stats.items():
            if player_name.lower() == name_lower:
                return stats
        
        # Partial match
        for player_name, stats in self.player_stats.items():
            if name_lower in player_name.lower():
                return stats
        
        return None
    
    def _predict_combo(
        self,
        player_name: str,
        props: List[str],
        opponent: Optional[str],
        is_home: bool,
        projected_minutes: Optional[float],
        game_date: Optional[str],
    ) -> Optional[PropPrediction]:
        """Predict combo prop (PRA, PR, PA, RA).

        Note
        ----
        This uses a fixed correlation approximation between components,
        so PRA/PR/PA/RA variance is approximate rather than fully
        modeled from joint distributions. This is a good baseline and
        can be upgraded later with QEPC-style multiverse sampling.
        """
        preds = []
        for prop in props:
            p = self.predict(player_name, prop, opponent, is_home, projected_minutes, game_date)
            if p is None:
                return None
            preds.append(p)
        
        # Sum projections
        total = sum(p.projection for p in preds)
        
        # Combined variance (assume ~0.3 correlation between stats)
        correlation = 0.3
        total_var = sum(p.std_dev ** 2 for p in preds)
        for i in range(len(preds)):
            for j in range(i + 1, len(preds)):
                total_var += 2 * correlation * preds[i].std_dev * preds[j].std_dev
        total_std = np.sqrt(total_var)
        
        combo_name = '+'.join(props)
        
        return PropPrediction(
            player_name=preds[0].player_name,
            team=preds[0].team,
            opponent=opponent or "Unknown",
            game_date=game_date,
            prop_type=combo_name,
            projection=round(total, 1),
            std_dev=round(total_std, 2),
            floor=round(max(0, total - 1.28 * total_std), 1),
            ceiling=round(total + 1.28 * total_std, 1),
            games_analyzed=preds[0].games_analyzed,
            avg_minutes=preds[0].avg_minutes,
            confidence=min(p.confidence for p in preds),
        )
    
    def predict_all(
        self,
        player_name: str,
        opponent: Optional[str] = None,
        is_home: bool = True,
        props: List[str] = ['PTS', 'REB', 'AST', '3PM', 'PRA'],
    ) -> Dict[str, PropPrediction]:
        """Get all prop predictions for a player."""
        results = {}
        for prop in props:
            pred = self.predict(player_name, prop, opponent, is_home)
            if pred:
                results[prop] = pred
        return results
    
    def get_team_projections(
        self,
        team: str,
        opponent: str,
        is_home: bool = True,
        min_minutes: float = 15.0,
        props: List[str] = ['PTS', 'REB', 'AST'],
    ) -> pd.DataFrame:
        """
        Get projections for all players on a team.
        
        Returns DataFrame with all predictions.
        """
        rows = []
        
        for name, stats in self.player_stats.items():
            if stats.team != team:
                continue
            if stats.avg_minutes < min_minutes:
                continue
            
            for prop in props:
                pred = self.predict(name, prop, opponent, is_home)
                if pred:
                    rows.append(pred.to_dict())
        
        if not rows:
            return pd.DataFrame()
        
        df = pd.DataFrame(rows)
        df = df.sort_values(['prop_type', 'projection'], ascending=[True, False])
        
        return df
    
    def find_edges(
        self,
        lines: Dict[str, Dict[str, float]],
        min_edge: float = 0.05,
        min_confidence: str = 'LOW',
    ) -> pd.DataFrame:
        """
        Find betting edges given lines.
        
        Parameters
        ----------
        lines : dict
            {"Player Name": {"PTS": 24.5, "REB": 7.5, ...}}
        min_edge : float
            Minimum edge to include (0.05 = 5%)
        min_confidence : str
            Minimum confidence level (LOW, MEDIUM, HIGH)
        
        Returns
        -------
        DataFrame with edges
        """
        conf_order = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        min_conf_val = conf_order.get(min_confidence, 0)
        
        edges = []
        
        for player, player_lines in lines.items():
            for prop, line in player_lines.items():
                pred = self.predict(player, prop)
                if pred is None:
                    continue
                
                if conf_order.get(pred.confidence, 0) < min_conf_val:
                    continue
                
                over_edge = pred.edge(line, 'over')
                under_edge = pred.edge(line, 'under')
                
                if over_edge >= min_edge:
                    edges.append({
                        'player': pred.player_name,
                        'prop': prop,
                        'line': line,
                        'projection': pred.projection,
                        'side': 'OVER',
                        'probability': pred.over_prob(line),
                        'edge': over_edge,
                        'confidence': pred.confidence,
                    })
                
                if under_edge >= min_edge:
                    edges.append({
                        'player': pred.player_name,
                        'prop': prop,
                        'line': line,
                        'projection': pred.projection,
                        'side': 'UNDER',
                        'probability': pred.under_prob(line),
                        'edge': under_edge,
                        'confidence': pred.confidence,
                    })
        
        if not edges:
            return pd.DataFrame()
        
        df = pd.DataFrame(edges)
        df = df.sort_values('edge', ascending=False)
        
        return df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_predict(
    player: str,
    prop: str = 'PTS',
    opponent: str = None,
    data_path: str = None,
) -> Optional[PropPrediction]:
    """
    Quick one-off prediction.
    
    Example:
        pred = quick_predict("LeBron James", "PTS", "Warriors")
        print(f"{pred.projection} projected, {pred.over_prob(25.5):.0%} over 25.5")
    """
    engine = PlayerPropsEngine(data_path)
    engine.load_data(verbose=False)
    return engine.predict(player, prop, opponent)


def load_engine(data_path: str = None, cutoff_date: str = None) -> PlayerPropsEngine:
    """Load and return a ready-to-use engine."""
    engine = PlayerPropsEngine(data_path)
    engine.load_data(cutoff_date=cutoff_date)
    return engine
