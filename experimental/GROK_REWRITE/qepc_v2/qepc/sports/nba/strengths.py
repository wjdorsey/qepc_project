"""
QEPC NBA Team Strengths
=======================
Calculates team strength ratings with:
- Real volatility from game-to-game variance
- Recency weighting (recent games matter more)
- Live ratings integration (ORtg, DRtg, Pace)
- Momentum/form tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

from qepc.core.config import get_config, QEPCConfig
from qepc.data.loader import DataLoader


@dataclass
class TeamStrength:
    """Complete team strength profile."""
    team: str
    
    # Offensive/Defensive ratings (per 100 possessions)
    ortg: float
    drtg: float
    net_rtg: float
    
    # Pace (possessions per 48 min)
    pace: float
    
    # Volatility (game-to-game std dev of scoring)
    volatility: float
    
    # Recent form (-1 to 1, negative = cold)
    momentum: float
    
    # Record
    wins: int
    losses: int
    games_played: int
    
    # Points per game
    ppg: float
    opp_ppg: float
    
    @property
    def win_pct(self) -> float:
        if self.games_played == 0:
            return 0.5
        return self.wins / self.games_played
    
    @property
    def expected_score(self) -> float:
        """Expected score based on ratings."""
        return self.ortg * (self.pace / 100)


class StrengthCalculator:
    """
    Calculates team strength ratings from game data.
    
    Uses recency weighting to emphasize recent performance.
    """
    
    def __init__(self, data_loader: DataLoader = None, config: QEPCConfig = None):
        self.loader = data_loader or DataLoader()
        self.config = config or get_config()
        
        # Cache for calculated strengths
        self._strengths: Dict[str, TeamStrength] = {}
        self._last_calculated: datetime = None
    
    def _make_tz_naive(self, dt):
        """Convert any datetime to timezone-naive UTC."""
        if dt is None:
            return None
        if isinstance(dt, pd.Timestamp):
            if dt.tz is not None:
                return dt.tz_convert('UTC').tz_localize(None)
            return dt
        if isinstance(dt, datetime):
            return pd.Timestamp(dt).tz_localize(None)
        return pd.Timestamp(dt)
    
    def _make_series_tz_naive(self, series: pd.Series) -> pd.Series:
        """Convert a datetime series to timezone-naive."""
        # First ensure it's datetime
        if not pd.api.types.is_datetime64_any_dtype(series):
            series = pd.to_datetime(series, errors='coerce')
        
        # Now handle timezone
        if series.dt.tz is not None:
            return series.dt.tz_convert('UTC').dt.tz_localize(None)
        return series
    
    def calculate_all_strengths(
        self,
        cutoff_date: str = None,
        use_live_ratings: bool = True,
        verbose: bool = True
    ) -> Dict[str, TeamStrength]:
        """
        Calculate strength ratings for all teams.
        
        Parameters
        ----------
        cutoff_date : str, optional
            Only use games before this date (for backtesting)
        use_live_ratings : bool
            Whether to use live API ratings as base
        verbose : bool
            Print progress
            
        Returns
        -------
        Dict mapping team name to TeamStrength
        """
        # Load data - use full season if backtesting (cutoff_date provided)
        use_full_season = cutoff_date is not None
        team_stats = self.loader.load_team_stats(full_season=use_full_season)
        live_ratings = self.loader.load_live_ratings() if use_live_ratings else None
        team_form = self.loader.load_team_form()
        
        if team_stats is None or team_stats.empty:
            if verbose:
                print("❌ No team stats available")
            return {}
        
        # Ensure gameDate is timezone-naive
        if 'gameDate' in team_stats.columns:
            team_stats['gameDate'] = self._make_series_tz_naive(team_stats['gameDate'])
        
        # Apply cutoff for backtesting
        if cutoff_date:
            cutoff = self._make_tz_naive(pd.Timestamp(cutoff_date))
            team_stats = team_stats[team_stats['gameDate'] < cutoff].copy()
            if verbose:
                print(f"📅 Using data before {cutoff_date}")
        
        if team_stats.empty:
            if verbose:
                print("❌ No games in date range")
            return {}
        
        # Get unique teams
        teams = team_stats['Team'].dropna().unique()
        
        strengths = {}
        
        for team in teams:
            # Skip historical teams
            if team in ['Baltimore Bullets', 'Buffalo Braves', 'Seattle SuperSonics']:
                continue
            
            strength = self._calculate_team_strength(
                team=team,
                team_stats=team_stats,
                live_ratings=live_ratings,
                team_form=team_form,
                cutoff_date=cutoff_date
            )
            
            if strength is not None:
                strengths[team] = strength
        
        self._strengths = strengths
        self._last_calculated = datetime.now()
        
        if verbose:
            print(f"✅ Calculated strengths for {len(strengths)} teams")
        
        return strengths
    
    def _calculate_team_strength(
        self,
        team: str,
        team_stats: pd.DataFrame,
        live_ratings: pd.DataFrame,
        team_form: pd.DataFrame,
        cutoff_date: str = None
    ) -> Optional[TeamStrength]:
        """Calculate strength for a single team."""
        
        # Filter to this team's games
        team_games = team_stats[team_stats['Team'] == team].copy()
        
        if len(team_games) < self.config.recency.min_games:
            return None
        
        # Sort by date (newest first)
        team_games = team_games.sort_values('gameDate', ascending=False)
        
        # Calculate recency weights
        if cutoff_date:
            reference_date = self._make_tz_naive(pd.Timestamp(cutoff_date))
        else:
            reference_date = self._make_tz_naive(pd.Timestamp.now())
        
        weights = self._calculate_recency_weights(team_games['gameDate'], reference_date)
        
        # Get live ratings if available (most accurate)
        if live_ratings is not None and not live_ratings.empty:
            live_row = live_ratings[live_ratings['Team'] == team]
            if not live_row.empty:
                live_row = live_row.iloc[0]
                ortg = live_row.get('ORtg', self.config.league.league_avg_ortg)
                drtg = live_row.get('DRtg', self.config.league.league_avg_drtg)
                pace = live_row.get('Pace', self.config.league.league_avg_pace)
                wins = int(live_row.get('Wins', 0))
                losses = int(live_row.get('Losses', 0))
            else:
                ortg, drtg, pace = self._estimate_ratings_from_games(team_games, weights)
                wins = int(team_games['win'].sum()) if 'win' in team_games.columns else 0
                losses = len(team_games) - wins
        else:
            ortg, drtg, pace = self._estimate_ratings_from_games(team_games, weights)
            wins = int(team_games['win'].sum()) if 'win' in team_games.columns else 0
            losses = len(team_games) - wins
        
        # Calculate REAL volatility from game-to-game scoring variance
        if 'teamScore' in team_games.columns:
            scores = team_games['teamScore'].values
            # Weighted standard deviation
            volatility = self._weighted_std(scores, weights)
        else:
            volatility = self.config.league.default_team_std
        
        # Calculate momentum from recent form
        momentum = self._calculate_momentum(team_games, team_form, team)
        
        # Points per game
        if 'teamScore' in team_games.columns:
            ppg = np.average(team_games['teamScore'].values, weights=weights)
            opp_ppg = np.average(team_games['opponentScore'].values, weights=weights) if 'opponentScore' in team_games.columns else self.config.league.league_avg_points
        else:
            ppg = ortg * (pace / 100)
            opp_ppg = drtg * (pace / 100)
        
        return TeamStrength(
            team=team,
            ortg=ortg,
            drtg=drtg,
            net_rtg=ortg - drtg,
            pace=pace,
            volatility=volatility,
            momentum=momentum,
            wins=wins,
            losses=losses,
            games_played=wins + losses,
            ppg=ppg,
            opp_ppg=opp_ppg
        )
    
    def _calculate_recency_weights(
        self,
        dates: pd.Series,
        reference_date: pd.Timestamp
    ) -> np.ndarray:
        """
        Calculate exponential decay weights for recency.
        
        More recent games get higher weight.
        """
        # Ensure both are timezone-naive
        dates_naive = self._make_series_tz_naive(dates)
        reference_naive = self._make_tz_naive(reference_date)
        
        days_ago = (reference_naive - dates_naive).dt.total_seconds() / (24 * 3600)
        days_ago = np.maximum(days_ago, 0)  # Handle future dates
        
        # Exponential decay: weight = 0.5^(days / half_life)
        half_life = self.config.recency.half_life_days
        weights = 0.5 ** (days_ago / half_life)
        
        # Normalize
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        return weights.values
    
    def _weighted_std(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Calculate weighted standard deviation."""
        if len(values) < 2:
            return self.config.league.default_team_std
        
        mean = np.average(values, weights=weights)
        variance = np.average((values - mean) ** 2, weights=weights)
        return np.sqrt(variance)
    
    def _estimate_ratings_from_games(
        self,
        team_games: pd.DataFrame,
        weights: np.ndarray
    ) -> Tuple[float, float, float]:
        """Estimate ORtg, DRtg, Pace from game-by-game stats."""
        
        league_avg = self.config.league.league_avg_points
        
        if 'teamScore' in team_games.columns:
            ppg = np.average(team_games['teamScore'].values, weights=weights)
            opp_ppg = np.average(team_games['opponentScore'].values, weights=weights) if 'opponentScore' in team_games.columns else league_avg
            
            # Estimate ratings (simplified)
            ortg = (ppg / league_avg) * self.config.league.league_avg_ortg
            drtg = (opp_ppg / league_avg) * self.config.league.league_avg_drtg
            
            # Estimate pace from total points
            total_pts = ppg + opp_ppg
            pace = (total_pts / (2 * league_avg)) * self.config.league.league_avg_pace
        else:
            ortg = self.config.league.league_avg_ortg
            drtg = self.config.league.league_avg_drtg
            pace = self.config.league.league_avg_pace
        
        return ortg, drtg, pace
    
    def _calculate_momentum(
        self,
        team_games: pd.DataFrame,
        team_form: pd.DataFrame,
        team: str
    ) -> float:
        """
        Calculate momentum (-1 to 1).
        
        Positive = hot streak, Negative = cold streak
        """
        # Try to use team form data first
        if team_form is not None and not team_form.empty:
            form_row = team_form[team_form['Team'] == team]
            if not form_row.empty:
                form_row = form_row.iloc[0]
                win_pct = form_row.get('Last_N_Win_Pct', 0.5)
                
                # Convert win% to momentum
                # 0.5 = 0, 1.0 = 1, 0.0 = -1
                momentum = (win_pct - 0.5) * 2
                return np.clip(momentum, -1, 1)
        
        # Fallback: Calculate from recent games
        n_form = self.config.recency.form_games
        recent = team_games.head(n_form)
        
        if 'win' in recent.columns and len(recent) > 0:
            recent_win_pct = recent['win'].mean()
            
            # Compare to season win%
            season_win_pct = team_games['win'].mean() if len(team_games) > n_form else recent_win_pct
            
            # Momentum = difference from expected
            momentum = (recent_win_pct - season_win_pct) * 2
            return np.clip(momentum, -1, 1)
        
        return 0.0
    
    def get_strength(self, team: str) -> Optional[TeamStrength]:
        """Get cached strength for a team."""
        return self._strengths.get(team)
    
    def get_all_strengths(self) -> Dict[str, TeamStrength]:
        """Get all cached strengths."""
        return self._strengths
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert strengths to DataFrame for easy viewing."""
        if not self._strengths:
            return pd.DataFrame()
        
        rows = []
        for team, s in self._strengths.items():
            rows.append({
                'Team': team,
                'ORtg': round(s.ortg, 1),
                'DRtg': round(s.drtg, 1),
                'NetRtg': round(s.net_rtg, 1),
                'Pace': round(s.pace, 1),
                'Volatility': round(s.volatility, 1),
                'Momentum': round(s.momentum, 2),
                'PPG': round(s.ppg, 1),
                'OppPPG': round(s.opp_ppg, 1),
                'Wins': s.wins,
                'Losses': s.losses,
                'WinPct': round(s.win_pct, 3),
            })
        
        df = pd.DataFrame(rows)
        return df.sort_values('NetRtg', ascending=False).reset_index(drop=True)
    
    def power_rankings(self) -> pd.DataFrame:
        """Get power rankings based on net rating."""
        df = self.to_dataframe()
        if df.empty:
            return df
        
        df['Rank'] = range(1, len(df) + 1)
        return df[['Rank', 'Team', 'NetRtg', 'ORtg', 'DRtg', 'WinPct', 'Momentum']]


# Convenience function
def calculate_strengths(
    cutoff_date: str = None,
    verbose: bool = True
) -> Dict[str, TeamStrength]:
    """
    Quick function to calculate all team strengths.
    
    Parameters
    ----------
    cutoff_date : str, optional
        Only use games before this date
    verbose : bool
        Print progress
    
    Returns
    -------
    Dict mapping team name to TeamStrength
    """
    calc = StrengthCalculator()
    return calc.calculate_all_strengths(cutoff_date=cutoff_date, verbose=verbose)
