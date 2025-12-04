"""
QEPC Player Data Fetcher
========================
Fetches player statistics from NBA API for props predictions.

Data Sources:
- Player game logs (current season + historical)
- Player season averages
- Player recent form (L5, L10)
- Team defensive ratings by position
- Pace and usage data

Usage:
    python fetch_player_data.py --current    # Current season only
    python fetch_player_data.py --full       # Full historical
    python fetch_player_data.py --defense    # Team defense stats
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import time
import argparse

# NBA API - install with: pip install nba_api
try:
    from nba_api.stats.endpoints import (
        playergamelog,
        commonallplayers,
        leaguedashplayerstats,
        leaguedashteamstats,
        teamdashboardbygeneralsplits,
    )
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("⚠️  nba_api not installed. Run: pip install nba_api")


# Rate limiting
REQUEST_DELAY = 0.6  # Seconds between requests


def get_all_players(season: str = "2024-25") -> pd.DataFrame:
    """Get list of all active players."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    print(f"📋 Fetching player list for {season}...")
    
    all_players = commonallplayers.CommonAllPlayers(
        is_only_current_season=1,
        season=season,
    )
    
    df = all_players.get_data_frames()[0]
    
    # Filter to active players
    df = df[df['ROSTERSTATUS'] == 1]
    
    print(f"   Found {len(df)} active players")
    return df


def get_player_season_stats(season: str = "2024-25") -> pd.DataFrame:
    """
    Get season averages for all players.
    
    Includes: PPG, RPG, APG, FG3M, STL, BLK, TOV, MIN, USG%
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    print(f"📊 Fetching player season stats for {season}...")
    
    stats = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Base',
    )
    
    time.sleep(REQUEST_DELAY)
    
    df = stats.get_data_frames()[0]
    
    # Select relevant columns
    cols = [
        'PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION',
        'GP', 'MIN', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV',
        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT',
        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB',
    ]
    
    df = df[[c for c in cols if c in df.columns]]
    
    print(f"   Got stats for {len(df)} players")
    return df


def get_player_game_logs(
    player_id: int,
    season: str = "2024-25",
    season_type: str = "Regular Season"
) -> pd.DataFrame:
    """Get game-by-game logs for a single player."""
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    try:
        logs = playergamelog.PlayerGameLog(
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        )
        
        time.sleep(REQUEST_DELAY)
        
        return logs.get_data_frames()[0]
    except Exception as e:
        print(f"   ⚠️  Error fetching logs for player {player_id}: {e}")
        return pd.DataFrame()


def get_all_player_game_logs(
    season: str = "2024-25",
    min_games: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Get game logs for all players with minimum games played.
    
    WARNING: This makes many API calls. Use sparingly.
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    # First get player list with stats to filter
    season_stats = get_player_season_stats(season)
    
    if season_stats.empty:
        return pd.DataFrame()
    
    # Filter to players with enough games
    qualified = season_stats[season_stats['GP'] >= min_games]
    
    if verbose:
        print(f"📚 Fetching game logs for {len(qualified)} players...")
    
    all_logs = []
    
    for i, (_, player) in enumerate(qualified.iterrows()):
        if verbose and (i + 1) % 25 == 0:
            print(f"   Progress: {i+1}/{len(qualified)}")
        
        logs = get_player_game_logs(player['PLAYER_ID'], season)
        
        if not logs.empty:
            logs['PLAYER_NAME'] = player['PLAYER_NAME']
            logs['TEAM_ABBREVIATION'] = player['TEAM_ABBREVIATION']
            all_logs.append(logs)
    
    if all_logs:
        result = pd.concat(all_logs, ignore_index=True)
        if verbose:
            print(f"   ✅ Got {len(result)} game logs total")
        return result
    
    return pd.DataFrame()


def calculate_player_volatility(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate volatility (standard deviation) for each player's stats.
    """
    if game_logs.empty:
        return pd.DataFrame()
    
    print("📈 Calculating player volatility...")
    
    # Group by player
    grouped = game_logs.groupby(['PLAYER_ID', 'PLAYER_NAME'])
    
    # Calculate stats
    volatility = grouped.agg({
        'PTS': ['mean', 'std', 'count'],
        'REB': ['mean', 'std'],
        'AST': ['mean', 'std'],
        'FG3M': ['mean', 'std'],
        'MIN': ['mean', 'std'],
        'STL': ['mean', 'std'],
        'BLK': ['mean', 'std'],
    }).reset_index()
    
    # Flatten column names
    volatility.columns = ['_'.join(col).strip('_') for col in volatility.columns]
    
    # Calculate coefficient of variation (consistency metric)
    for stat in ['PTS', 'REB', 'AST', 'FG3M']:
        mean_col = f'{stat}_mean'
        std_col = f'{stat}_std'
        cv_col = f'{stat}_cv'
        
        if mean_col in volatility.columns and std_col in volatility.columns:
            volatility[cv_col] = volatility[std_col] / volatility[mean_col].replace(0, np.nan)
    
    print(f"   Calculated volatility for {len(volatility)} players")
    return volatility


def calculate_recent_form(game_logs: pd.DataFrame, n_games: int = 5) -> pd.DataFrame:
    """
    Calculate recent form (last N games) vs season average.
    """
    if game_logs.empty:
        return pd.DataFrame()
    
    print(f"🔥 Calculating recent form (L{n_games})...")
    
    # Sort by date
    game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'])
    game_logs = game_logs.sort_values(['PLAYER_ID', 'GAME_DATE'], ascending=[True, False])
    
    # Get last N games for each player
    recent = game_logs.groupby('PLAYER_ID').head(n_games)
    
    # Calculate recent averages
    recent_stats = recent.groupby(['PLAYER_ID', 'PLAYER_NAME']).agg({
        'PTS': 'mean',
        'REB': 'mean',
        'AST': 'mean',
        'FG3M': 'mean',
        'MIN': 'mean',
    }).reset_index()
    
    # Rename columns
    recent_stats.columns = ['PLAYER_ID', 'PLAYER_NAME', 
                           f'PTS_L{n_games}', f'REB_L{n_games}', 
                           f'AST_L{n_games}', f'FG3M_L{n_games}',
                           f'MIN_L{n_games}']
    
    print(f"   Calculated form for {len(recent_stats)} players")
    return recent_stats


def get_team_defense_stats(season: str = "2024-25") -> pd.DataFrame:
    """
    Get team defensive statistics.
    
    Returns points/rebounds/assists allowed per game,
    and opponent shooting percentages.
    """
    if not NBA_API_AVAILABLE:
        return pd.DataFrame()
    
    print(f"🛡️ Fetching team defense stats for {season}...")
    
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        per_mode_detailed='PerGame',
        measure_type_detailed_defense='Opponent',  # Get opponent stats
    )
    
    time.sleep(REQUEST_DELAY)
    
    df = stats.get_data_frames()[0]
    
    # Calculate defensive rating relative to league average
    league_avg_pts = df['OPP_PTS'].mean() if 'OPP_PTS' in df.columns else df['PTS'].mean()
    
    if 'OPP_PTS' in df.columns:
        df['DEF_PTS_FACTOR'] = df['OPP_PTS'] / league_avg_pts
    
    print(f"   Got defense stats for {len(df)} teams")
    return df


def build_player_profiles(
    season_stats: pd.DataFrame,
    volatility: pd.DataFrame,
    recent_form: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine all player data into comprehensive profiles.
    """
    print("🏗️ Building player profiles...")
    
    # Start with season stats
    profiles = season_stats.copy()
    
    # Merge volatility
    if not volatility.empty:
        vol_cols = [c for c in volatility.columns if c not in ['PLAYER_NAME']]
        profiles = profiles.merge(
            volatility[vol_cols],
            on='PLAYER_ID',
            how='left',
            suffixes=('', '_vol')
        )
    
    # Merge recent form
    if not recent_form.empty:
        form_cols = [c for c in recent_form.columns if c not in ['PLAYER_NAME']]
        profiles = profiles.merge(
            recent_form[form_cols],
            on='PLAYER_ID',
            how='left'
        )
    
    # Calculate momentum (recent vs season)
    if 'PTS_L5' in profiles.columns:
        profiles['PTS_MOMENTUM'] = (profiles['PTS_L5'] - profiles['PTS']) / profiles['PTS'].replace(0, np.nan)
        profiles['REB_MOMENTUM'] = (profiles['REB_L5'] - profiles['REB']) / profiles['REB'].replace(0, np.nan)
        profiles['AST_MOMENTUM'] = (profiles['AST_L5'] - profiles['AST']) / profiles['AST'].replace(0, np.nan)
    
    print(f"   Built profiles for {len(profiles)} players")
    return profiles


def save_data(data_dir: Path, **datasets):
    """Save datasets to CSV files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for name, df in datasets.items():
        if df is not None and not df.empty:
            path = data_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            print(f"💾 Saved {name}: {len(df)} rows -> {path.name}")


def refresh_player_data(
    data_dir: Path,
    season: str = "2024-25",
    full_logs: bool = False,
    verbose: bool = True,
):
    """
    Main refresh function for player data.
    
    Parameters
    ----------
    data_dir : Path
        Directory to save data
    season : str
        NBA season (e.g., "2024-25")
    full_logs : bool
        Whether to fetch full game logs (slow)
    verbose : bool
        Print progress
    """
    if not NBA_API_AVAILABLE:
        print("❌ nba_api not available. Install with: pip install nba_api")
        return
    
    print(f"\n🏀 QEPC Player Data Refresh")
    print(f"   Season: {season}")
    print("=" * 50)
    
    # Get season stats (fast)
    season_stats = get_player_season_stats(season)
    
    # Get team defense (fast)
    team_defense = get_team_defense_stats(season)
    
    # Initialize volatility and form
    volatility = pd.DataFrame()
    recent_form = pd.DataFrame()
    game_logs = pd.DataFrame()
    
    if full_logs:
        # Get full game logs (slow - many API calls)
        game_logs = get_all_player_game_logs(season, min_games=5, verbose=verbose)
        
        if not game_logs.empty:
            volatility = calculate_player_volatility(game_logs)
            recent_form = calculate_recent_form(game_logs, n_games=5)
    
    # Build profiles
    profiles = build_player_profiles(season_stats, volatility, recent_form)
    
    # Save everything
    props_dir = data_dir / "props"
    
    save_data(
        props_dir,
        Player_Season_Averages=profiles,
        Player_Game_Logs=game_logs if not game_logs.empty else None,
        Player_Volatility=volatility if not volatility.empty else None,
        Player_Recent_Form=recent_form if not recent_form.empty else None,
        Team_Defense_Stats=team_defense,
    )
    
    print("\n✅ Player data refresh complete!")
    
    return profiles


def main():
    parser = argparse.ArgumentParser(description='QEPC Player Data Fetcher')
    parser.add_argument('--season', default='2024-25', help='NBA season')
    parser.add_argument('--full', action='store_true', help='Fetch full game logs (slow)')
    parser.add_argument('--data-dir', default=None, help='Data directory path')
    
    args = parser.parse_args()
    
    # Find data directory
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Try to find project root
        current = Path.cwd()
        for p in [current] + list(current.parents)[:5]:
            if (p / "data").exists():
                data_dir = p / "data"
                break
        else:
            data_dir = current / "data"
    
    refresh_player_data(
        data_dir=data_dir,
        season=args.season,
        full_logs=args.full,
    )


if __name__ == "__main__":
    main()
