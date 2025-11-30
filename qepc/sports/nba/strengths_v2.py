"""
QEPC NBA Team Strengths v2 - IMPROVED
Real volatility from game variance + recency weighting
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

# Try to import from NBA API, fall back to CSV if not available
try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguedashopponentstats
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

# =============================================================================
# CONFIGURATION
# =============================================================================

RECENCY_HALF_LIFE_DAYS = 30     # 30 days for 50% weight decay
MIN_GAMES_REQUIRED = 5          # Minimum games for stable stats
CURRENT_SEASON = '2024-25'      # Update each season


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_recency_weights(dates: pd.Series, reference_date: pd.Timestamp = None) -> pd.Series:
    """Apply exponential decay weights based on game age."""
    if reference_date is None:
        reference_date = pd.Timestamp.now()
    
    if dates.empty:
        return pd.Series(dtype=float)
    
    # Ensure dates are timestamps
    dates = pd.to_datetime(dates)
    
    # Calculate days ago
    days_ago = (reference_date - dates).dt.days
    
    # Exponential decay: weight = 0.5^(days/half_life)
    weights = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    
    # Normalize
    return weights / weights.sum()


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted mean."""
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return np.average(values[mask], weights=weights[mask])


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted standard deviation."""
    mask = values.notna() & weights.notna()
    if mask.sum() < 2:
        return np.nan
    
    v = values[mask].values
    w = weights[mask].values
    
    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return np.sqrt(variance)


# =============================================================================
# MAIN FUNCTION - FROM GAME LOG DATA
# =============================================================================

def calculate_advanced_strengths(
    game_data: pd.DataFrame = None,
    cutoff_date: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate team strengths from game data with recency weighting and real volatility.
    
    Parameters
    ----------
    game_data : DataFrame, optional
        Game-by-game data. If None, tries to load from default path.
    cutoff_date : str, optional
        Only use games before this date (for backtesting)
    verbose : bool
        Print progress messages
    
    Returns
    -------
    DataFrame with columns: Team, ORtg, DRtg, Pace, Volatility, SOS
    """
    # Load data if not provided
    if game_data is None:
        game_data = _load_default_game_data()
    
    if game_data is None or game_data.empty:
        if verbose:
            print("❌ No game data available")
        return pd.DataFrame()
    
    df = game_data.copy()
    
    # Parse dates
    if 'gameDate' in df.columns:
        df['gameDate'] = pd.to_datetime(df['gameDate'], errors='coerce')
    elif 'date' in df.columns:
        df['gameDate'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Apply cutoff for backtesting
    if cutoff_date:
        cutoff = pd.Timestamp(cutoff_date)
        df = df[df['gameDate'] < cutoff].copy()
        reference_date = cutoff
    else:
        reference_date = pd.Timestamp.now()
    
    if verbose:
        print(f"[Strengths] Processing {len(df)} game records...")
    
    # Standardize team column
    team_col = None
    for col in ['teamName', 'TEAM_NAME', 'team']:
        if col in df.columns:
            team_col = col
            break
    
    if team_col is None:
        if verbose:
            print("❌ Cannot find team column")
        return pd.DataFrame()
    
    # Get unique teams
    teams = df[team_col].unique()
    
    results = []
    
    for team in teams:
        team_games = df[df[team_col] == team].copy()
        
        if len(team_games) < MIN_GAMES_REQUIRED:
            continue
        
        # Sort by date (newest first for recency)
        team_games = team_games.sort_values('gameDate', ascending=False)
        
        # Calculate recency weights
        weights = _apply_recency_weights(team_games['gameDate'], reference_date)
        
        # Get score columns
        score_col = None
        for col in ['teamScore', 'PTS', 'pts']:
            if col in team_games.columns:
                score_col = col
                break
        
        opp_score_col = None
        for col in ['opponentScore', 'OPP_PTS', 'opp_pts']:
            if col in team_games.columns:
                opp_score_col = col
                break
        
        if score_col is None:
            continue
        
        # Calculate weighted stats
        avg_pts = _weighted_mean(team_games[score_col], weights)
        
        # REAL VOLATILITY: Weighted standard deviation of scores
        volatility = _weighted_std(team_games[score_col], weights)
        if pd.isna(volatility):
            volatility = 10.0  # Default
        
        # Offensive rating (simplified)
        ortg = avg_pts  # Points scored
        
        # Defensive rating
        if opp_score_col:
            drtg = _weighted_mean(team_games[opp_score_col], weights)
        else:
            drtg = 110.0  # League average fallback
        
        # Pace (total points in game / 2)
        if opp_score_col:
            total_pts = team_games[score_col] + team_games[opp_score_col]
            pace = _weighted_mean(total_pts / 2, weights)
        else:
            pace = avg_pts
        
        # Calculate SOS (strength of schedule) if opponent ratings available
        # For now, use 1.0 (neutral)
        sos = 1.0
        
        results.append({
            'Team': team,
            'ORtg': round(avg_pts, 1),
            'DRtg': round(drtg, 1),
            'Pace': round(pace, 1),
            'Volatility': round(volatility, 2),
            'SOS': round(sos, 3),
            'Games': len(team_games),
        })
    
    result_df = pd.DataFrame(results)
    
    if verbose:
        print(f"[Strengths] Calculated ratings for {len(result_df)} teams")
    
    return result_df


def _load_default_game_data() -> pd.DataFrame:
    """Try to load game data from common paths."""
    possible_paths = [
        Path("data/raw/TeamStatistics.csv"),
        Path("data/Games.csv"),
        Path("data/GameResults_2025.csv"),
        Path("../data/raw/TeamStatistics.csv"),
    ]
    
    for path in possible_paths:
        if path.exists():
            try:
                return pd.read_csv(path)
            except Exception:
                continue
    
    return None


# =============================================================================
# ALTERNATIVE: FROM NBA API (Live Data)
# =============================================================================

def calculate_strengths_from_api(season: str = CURRENT_SEASON, verbose: bool = True) -> pd.DataFrame:
    """
    Calculate team strengths from NBA API.
    
    Note: This doesn't have recency weighting since API returns season aggregates.
    Use calculate_advanced_strengths() with game log data for better accuracy.
    """
    if not HAS_NBA_API:
        if verbose:
            print("❌ nba_api not installed")
        return pd.DataFrame()
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(season=season)
        df = stats.get_data_frames()[0]
        
        # Get opponent stats for SOS
        opp_stats = leaguedashopponentstats.LeagueDashOpponentStats(season=season)
        opp_df = opp_stats.get_data_frames()[0]
        
        df = pd.merge(df, opp_df[['TEAM_NAME', 'OPP_OFF_RATING']], on='TEAM_NAME', how='left')
        
        # Calculate SOS
        league_avg_ortg = df['OFF_RATING'].mean()
        df['SOS'] = df['OPP_OFF_RATING'] / league_avg_ortg
        
        # Estimate volatility from points std if available
        if 'PTS_STD' in df.columns:
            df['Volatility'] = df['PTS_STD']
        else:
            df['Volatility'] = 10.0  # Default
        
        # Rename columns
        result = df[['TEAM_NAME', 'OFF_RATING', 'DEF_RATING', 'PACE', 'Volatility', 'SOS']].copy()
        result.columns = ['Team', 'ORtg', 'DRtg', 'Pace', 'Volatility', 'SOS']
        
        if verbose:
            print(f"[Strengths] Loaded {len(result)} teams from NBA API")
        
        return result
        
    except Exception as e:
        if verbose:
            print(f"❌ API error: {e}")
        return pd.DataFrame()


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================

def get_team_strengths(
    source: str = 'auto',
    cutoff_date: str = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Get team strengths from best available source.
    
    Parameters
    ----------
    source : str
        'auto', 'csv', or 'api'
    cutoff_date : str
        For backtesting (only works with CSV source)
    verbose : bool
        Print progress
    
    Returns
    -------
    DataFrame with team strength ratings
    """
    if source == 'api':
        return calculate_strengths_from_api(verbose=verbose)
    
    if source == 'csv':
        return calculate_advanced_strengths(cutoff_date=cutoff_date, verbose=verbose)
    
    # Auto mode: Try CSV first (more accurate), fall back to API
    result = calculate_advanced_strengths(cutoff_date=cutoff_date, verbose=verbose)
    
    if result.empty and HAS_NBA_API:
        if verbose:
            print("[Strengths] Falling back to NBA API...")
        result = calculate_strengths_from_api(verbose=verbose)
    
    return result
