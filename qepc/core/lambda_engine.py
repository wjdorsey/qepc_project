"""
QEPC Module: lambda_engine.py - IMPROVED
=========================================

Key improvements:
1. Dynamic HCA (home court advantage varies by team)
2. Rest days advantage
3. Back-to-back penalty
4. Better formula documentation

"""
import pandas as pd
import numpy as np
from typing import Dict, Optional

# =============================================================================
# CONFIGURATION
# =============================================================================

LEAGUE_AVG_POINTS = 114.0  # 2024-25 NBA average (updated from 110)

# Base home court advantage: ~2.8% boost to home team scoring
BASE_HCA = 1.028

# Team-specific HCA adjustments (some venues are tougher)
# Values are multipliers on top of BASE_HCA
# Data-driven: could be calculated from historical home/away splits
TEAM_HCA_BOOST = {
    # High-altitude / tough venues
    "Denver Nuggets": 1.02,        # Altitude advantage
    "Utah Jazz": 1.015,            # Altitude + loud crowd
    "Miami Heat": 1.01,            # Kaseya Center is tough
    
    # Average venues
    # Most teams default to 1.0 (no adjustment)
    
    # Weaker home advantages
    "Brooklyn Nets": 0.99,         # Shared arena, less home feel
    "Los Angeles Clippers": 0.99,  # New arena, establishing home court
}

# Rest advantage: Points boost per extra rest day (diminishing returns)
REST_ADVANTAGE_PER_DAY = 1.5  # ~1.5 points per rest day advantage
MAX_REST_ADVANTAGE = 4.0      # Cap at 4 points

# Back-to-back penalty
B2B_PENALTY = 0.97  # 3% scoring reduction on second night of B2B


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_team_hca(team: str) -> float:
    """Get team-specific HCA multiplier."""
    base_boost = TEAM_HCA_BOOST.get(team, 1.0)
    return BASE_HCA * base_boost


def _calculate_rest_factor(
    rest_days_home: Optional[int],
    rest_days_away: Optional[int]
) -> tuple:
    """
    Calculate rest-based adjustments for both teams.
    
    Returns (home_factor, away_factor)
    """
    if rest_days_home is None or rest_days_away is None:
        return (1.0, 1.0)
    
    # Rest advantage is relative
    rest_diff = rest_days_home - rest_days_away
    
    # Convert to point advantage (capped)
    point_swing = np.clip(
        rest_diff * REST_ADVANTAGE_PER_DAY,
        -MAX_REST_ADVANTAGE,
        MAX_REST_ADVANTAGE
    )
    
    # Convert points to multiplicative factor
    # +4 points ≈ 3.5% boost
    home_factor = 1 + (point_swing / LEAGUE_AVG_POINTS / 2)
    away_factor = 1 - (point_swing / LEAGUE_AVG_POINTS / 2)
    
    return (home_factor, away_factor)


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def compute_lambda(
    schedule_df: pd.DataFrame, 
    team_stats_df: pd.DataFrame,
    include_situational: bool = True
) -> pd.DataFrame:
    """
    Compute expected scores (lambda) for each game.
    
    The formula:
    λ_home = LEAGUE_AVG × (home_off / league_avg_off) × (away_def / league_avg_def) 
             × pace_factor × HCA × situational_factors
    
    Parameters
    ----------
    schedule_df : DataFrame
        Must have 'Home Team' and 'Away Team' columns
    team_stats_df : DataFrame
        Must have 'Team', 'ORtg', 'DRtg', 'Pace', 'Volatility' columns
    include_situational : bool
        Whether to include rest/B2B adjustments (requires additional columns)
    
    Returns
    -------
    DataFrame with lambda_home, lambda_away, vol_home, vol_away columns
    """
    if team_stats_df.empty: 
        print("[QEPC Lambda] ERROR: Cannot compute without team strengths.")
        return schedule_df

    # Calculate league averages for normalization
    league_avg_ortg = team_stats_df['ORtg'].mean()
    league_avg_drtg = team_stats_df['DRtg'].mean()
    league_avg_pace = team_stats_df['Pace'].mean()

    # Build team lookup dictionary
    strengths = {}
    for _, row in team_stats_df.iterrows():
        team = row['Team']
        strengths[team] = {
            'off_ratio': row['ORtg'] / league_avg_ortg,  # >1 = better offense
            'def_ratio': row['DRtg'] / league_avg_drtg,  # >1 = worse defense (allows more)
            'pace_ratio': row['Pace'] / league_avg_pace,
            'volatility': row.get('Volatility', 11.0),
        }
    
    # Prepare output
    df = schedule_df.copy()
    df['lambda_home'] = 0.0
    df['lambda_away'] = 0.0
    df['vol_home'] = 0.0
    df['vol_away'] = 0.0

    missing_teams = set()

    for index, row in df.iterrows():
        home_team = row['Home Team']
        away_team = row['Away Team']

        if home_team not in strengths:
            missing_teams.add(home_team)
            continue
        if away_team not in strengths:
            missing_teams.add(away_team)
            continue
            
        home = strengths[home_team]
        away = strengths[away_team]

        # =================================================================
        # PACE FACTOR
        # =================================================================
        # Both teams' pace affects the game
        # Two fast teams = high-scoring game
        game_pace = home['pace_ratio'] * away['pace_ratio']

        # =================================================================
        # HOME COURT ADVANTAGE
        # =================================================================
        hca = _get_team_hca(home_team)

        # =================================================================
        # SITUATIONAL FACTORS (if available)
        # =================================================================
        home_situational = 1.0
        away_situational = 1.0
        
        if include_situational:
            # Rest days
            rest_home = row.get('rest_days_home')
            rest_away = row.get('rest_days_away')
            if rest_home is not None and rest_away is not None:
                h_rest, a_rest = _calculate_rest_factor(rest_home, rest_away)
                home_situational *= h_rest
                away_situational *= a_rest
            
            # Back-to-back
            if row.get('home_b2b', False):
                home_situational *= B2B_PENALTY
            if row.get('away_b2b', False):
                away_situational *= B2B_PENALTY

        # =================================================================
        # LAMBDA CALCULATION
        # =================================================================
        # Home team expected score:
        # = League average
        # × Home team offensive strength
        # × Away team defensive weakness (higher = allows more points)
        # × Pace factor
        # × Home court advantage
        # × Situational factors
        
        lambda_home = (
            LEAGUE_AVG_POINTS 
            * home['off_ratio'] 
            * away['def_ratio'] 
            * game_pace 
            * hca
            * home_situational
        )
        
        # Away team expected score (no HCA)
        lambda_away = (
            LEAGUE_AVG_POINTS 
            * away['off_ratio'] 
            * home['def_ratio'] 
            * game_pace
            * away_situational
        )

        # Store results
        df.loc[index, 'lambda_home'] = lambda_home
        df.loc[index, 'lambda_away'] = lambda_away
        df.loc[index, 'vol_home'] = home['volatility']
        df.loc[index, 'vol_away'] = away['volatility']

    if missing_teams:
        print(f"[QEPC Lambda] WARNING: {len(missing_teams)} teams not found in strengths:")
        for t in sorted(missing_teams)[:5]:
            print(f"  - {t}")
        if len(missing_teams) > 5:
            print(f"  ... and {len(missing_teams) - 5} more")

    valid_games = (df['lambda_home'] > 0).sum()
    print(f"[QEPC Lambda] Computed λ for {valid_games}/{len(df)} games.")
    
    return df


def compute_lambda_simple(
    schedule_df: pd.DataFrame, 
    team_stats_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Simplified version without situational factors.
    Use this for basic predictions.
    """
    return compute_lambda(schedule_df, team_stats_df, include_situational=False)
