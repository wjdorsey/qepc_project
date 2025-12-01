"""
QEPC NBA Team Strengths v2 - with live overlay
----------------------------------------------

Core ideas:
- Build team ORtg/DRtg/Pace from game logs (TeamStatistics.csv) with recency weighting.
- Compute real volatility from the distribution of scores.
- Optionally overlay live ORtg/DRtg/Pace from NBA API
  (data/live/team_stats_live_nba_api.csv) to keep the current season sharp.

This file is designed to run from anywhere as long as qepc_autoload is set up.
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Try to import from NBA API for pure-API strengths (optional)
try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguedashopponentstats
    HAS_NBA_API = True
except ImportError:
    HAS_NBA_API = False

# =============================================================================
# CONFIGURATION
# =============================================================================

RECENCY_HALF_LIFE_DAYS = 30      # 50% weight decay every 30 days
MIN_GAMES_REQUIRED = 5           # Minimum games for stable team stats
CURRENT_SEASON = "2025-26"       # Only used for API-based strengths, not critical

# Default live overlay settings
DEFAULT_USE_LIVE_OVERLAY = True
DEFAULT_LIVE_WEIGHT = 0.30       # 30% live, 70% historical


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _apply_recency_weights(dates: pd.Series,
                           reference_date: pd.Timestamp = None) -> pd.Series:
    """
    Apply exponential decay weights based on game age.

    weights ~ 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)
    normalized to sum to 1.
    """
    if reference_date is None:
        reference_date = pd.Timestamp.now()

    if dates is None or dates.empty:
        return pd.Series(dtype=float)

    dates = pd.to_datetime(dates, errors="coerce")

    # Drop timezone info so we don't get tz-aware vs tz-naive errors
    if hasattr(dates.dt, "tz") and dates.dt.tz is not None:
        dates = dates.dt.tz_localize(None)

    # Coerce reference_date to match (naive)
    if getattr(reference_date, "tzinfo", None) is not None:
        reference_date = reference_date.tz_convert(None)

    days_ago = (reference_date - dates).dt.days
    # In case of NaNs
    valid = days_ago.notna()
    if not valid.any():
        return pd.Series(dtype=float)

    days_ago = days_ago[valid]
    weights = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)

    # Normalize to sum 1
    weights = weights / weights.sum()
    # Reindex to original index, fill missing with 0
    full_weights = pd.Series(0.0, index=dates.index)
    full_weights.loc[days_ago.index] = weights
    return full_weights


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean, ignoring NaNs."""
    if values is None or weights is None:
        return np.nan
    mask = values.notna() & weights.notna()
    if not mask.any():
        return np.nan
    return float(np.average(values[mask], weights=weights[mask]))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Weighted standard deviation for volatility."""
    if values is None or weights is None:
        return np.nan
    mask = values.notna() & weights.notna()
    if mask.sum() < 2:
        return np.nan

    v = values[mask].values
    w = weights[mask].values

    mean = np.average(v, weights=w)
    variance = np.average((v - mean) ** 2, weights=w)
    return float(np.sqrt(variance))


def _load_default_game_data(verbose: bool = True) -> pd.DataFrame:
    """
    Try to load game data from common QEPC paths.

    Priority:
      1) data/raw/TeamStatistics.csv  (your big team game log)
      2) data/raw/Team_Stats.csv
      3) data/GameResults_2025.csv
      4) data/Games.csv  (less detailed)
    """

    # Anchor via qepc.autoload.paths if available
    try:
        from qepc.autoload.paths import get_project_root, get_data_dir
        root = get_project_root()
        data_dir = get_data_dir()
    except Exception:
        root = Path.cwd()
        data_dir = root / "data"

    candidates = [
        data_dir / "raw" / "TeamStatistics.csv",
        data_dir / "raw" / "Team_Stats.csv",
        data_dir / "GameResults_2025.csv",
        data_dir / "Games.csv",
    ]

    for path in candidates:
        if path.exists():
            try:
                if verbose:
                    print(f"[Strengths] Loading game data from {path}")
                df = pd.read_csv(path)
                return df
            except Exception as exc:
                if verbose:
                    print(f"[Strengths] Failed to read {path}: {exc}")

    if verbose:
        print("[Strengths] Could not find any local game data CSV.")
        print("           Checked:", [str(p) for p in candidates])
    return pd.DataFrame()


def _load_live_overlay_df(verbose: bool = True) -> pd.DataFrame:
    """
    Load the live NBA API team stats overlay from:
        data/live/team_stats_live_nba_api.csv

    Expected columns:
        TEAM_ID, Team, ORtg_live, DRtg_live, NetRtg_live, Pace_live, ...
    """
    try:
        from qepc.autoload.paths import get_data_dir
        data_dir = get_data_dir()
    except Exception:
        data_dir = Path.cwd() / "data"

    live_path = data_dir / "live" / "team_stats_live_nba_api.csv"

    if not live_path.exists():
        if verbose:
            print(f"[Strengths Live] No live team stats found at {live_path}")
        return pd.DataFrame()

    try:
        live_df = pd.read_csv(live_path)
        if verbose:
            print(f"[Strengths Live] Loaded live overlay from {live_path} "
                  f"({len(live_df)} rows)")
        # Ensure 'Team' column exists and is string for merge
        if "Team" not in live_df.columns:
            if verbose:
                print("[Strengths Live] Live file missing 'Team' column; "
                      "cannot overlay.")
            return pd.DataFrame()
        live_df["Team"] = live_df["Team"].astype(str)
        return live_df
    except Exception as exc:
        if verbose:
            print(f"[Strengths Live] Error reading {live_path}: {exc}")
        return pd.DataFrame()


def _apply_live_overlay_to_strengths(strengths_df: pd.DataFrame,
                                     live_weight: float = DEFAULT_LIVE_WEIGHT,
                                     verbose: bool = True) -> pd.DataFrame:
    """
    Blend live ORtg/DRtg/Pace from NBA API into the historical strengths table.

    ORtg_final = (1 - live_weight) * ORtg_hist + live_weight * ORtg_live
    similarly for DRtg and Pace.

    Only applies for teams that exist in the live overlay.
    """
    if strengths_df is None or strengths_df.empty:
        return strengths_df

    live_df = _load_live_overlay_df(verbose=verbose)
    if live_df.empty:
        return strengths_df

    merged = strengths_df.merge(
        live_df[["Team", "ORtg_live", "DRtg_live", "Pace_live"]],
        on="Team",
        how="left",
        suffixes=("", "_live"),
    )

    # Keep original columns for debugging / sanity checks
    for col in ["ORtg", "DRtg", "Pace"]:
        hist_col = f"{col}_hist"
        if hist_col not in merged.columns:
            merged[hist_col] = merged[col]

    mask = merged["ORtg_live"].notna()

    if verbose:
        n_applied = int(mask.sum())
        print(f"[Strengths Live] Applying live overlay to {n_applied} teams "
              f"with weight={live_weight:.2f}")

    # Blend only where live stats exist
    merged.loc[mask, "ORtg"] = (
        (1.0 - live_weight) * merged.loc[mask, "ORtg_hist"]
        + live_weight * merged.loc[mask, "ORtg_live"]
    )
    merged.loc[mask, "DRtg"] = (
        (1.0 - live_weight) * merged.loc[mask, "DRtg_hist"]
        + live_weight * merged.loc[mask, "DRtg_live"]
    )
    merged.loc[mask, "Pace"] = (
        (1.0 - live_weight) * merged.loc[mask, "Pace_hist"]
        + live_weight * merged.loc[mask, "Pace_live"]
    )

    return merged


# =============================================================================
# MAIN FUNCTION - FROM GAME LOG DATA
# =============================================================================

def calculate_advanced_strengths(
    game_data: pd.DataFrame = None,
    cutoff_date: str = None,
    verbose: bool = True,
    use_live_overlay: bool = DEFAULT_USE_LIVE_OVERLAY,
    live_weight: float = DEFAULT_LIVE_WEIGHT,
) -> pd.DataFrame:
    """
    Calculate team strengths from game data with recency weighting and real volatility.

    Parameters
    ----------
    game_data : DataFrame, optional
        Game-by-game data. If None, tries to load from default path.
        Expected columns (at minimum):
            - teamName or TEAM_NAME or team
            - teamScore (or PTS)
            - opponentScore (or OPP_PTS)
            - gameDate or date
    cutoff_date : str, optional
        Only use games before this date (YYYY-MM-DD) for backtesting.
    verbose : bool
        Print progress messages.
    use_live_overlay : bool
        If True, blend in live NBA API team stats when available.
    live_weight : float
        Weight of live stats vs historical (0.0 - 1.0).

    Returns
    -------
    DataFrame with columns:
        Team, ORtg, DRtg, Pace, Volatility, SOS, Games,
        (and optionally ORtg_hist/DRtg_hist/Pace_hist if overlay applied)
    """
    # Load data if not provided
    if game_data is None:
        game_data = _load_default_game_data(verbose=verbose)

    if game_data is None or game_data.empty:
        if verbose:
            print("❌ No game data available for strengths.")
        return pd.DataFrame()

    df = game_data.copy()

    # Parse / normalize date column
    date_col = None
    for col in ["gameDate", "GAME_DATE", "Date", "date"]:
        if col in df.columns:
            date_col = col
            break

    if date_col is None:
        if verbose:
            print("❌ Cannot find a date column in game data.")
        return pd.DataFrame()

    # Parse dates with utc=True to handle ISO 8601 format with timezone
    df["gameDate"] = pd.to_datetime(df[date_col], utc=True, errors="coerce")

    # Drop truly invalid dates
    valid_dates = df["gameDate"].notna()
    invalid_count = (~valid_dates).sum()
    if invalid_count > 0 and verbose:
        print(f"[Strengths] Dropped {invalid_count} rows with invalid dates.")

    df = df[valid_dates].copy()

    # Strip timezone info to avoid tz-aware vs tz-naive issues
    if hasattr(df["gameDate"].dt, "tz") and df["gameDate"].dt.tz is not None:
        df["gameDate"] = df["gameDate"].dt.tz_localize(None)

    # Apply cutoff for backtesting
    if cutoff_date:
        cutoff = pd.to_datetime(cutoff_date)
        df = df[df["gameDate"] < cutoff].copy()
        reference_date = cutoff
    else:
        reference_date = pd.Timestamp.now()

    if df.empty:
        if verbose:
            print("[Strengths] No games remain after date filtering.")
        return pd.DataFrame()

    if verbose:
        print(f"[Strengths] Processing {len(df)} game records...")
        print(
            f"[Strengths] Data range: {df['gameDate'].min().date()} "
            f"to {df['gameDate'].max().date()}"
        )

    # Standardize team column
    team_col = None
    for col in ["teamName", "TEAM_NAME", "team"]:
        if col in df.columns:
            team_col = col
            break

    if team_col is None:
        if verbose:
            print("❌ Cannot find team column in game data.")
        return pd.DataFrame()

    df["Team"] = df[team_col].astype(str)

    # Identify scoring columns
    score_col = None
    for col in ["teamScore", "PTS", "pts"]:
        if col in df.columns:
            score_col = col
            break

    opp_score_col = None
    for col in ["opponentScore", "OPP_PTS", "opp_pts"]:
        if col in df.columns:
            opp_score_col = col
            break

    if score_col is None:
        if verbose:
            print("❌ No team score column found in game data.")
        return pd.DataFrame()

    results = []
    teams = sorted(df["Team"].unique())

    for team in teams:
        team_games = df[df["Team"] == team].copy()
        if len(team_games) < MIN_GAMES_REQUIRED:
            # Skip very tiny samples
            continue

        team_games = team_games.sort_values("gameDate", ascending=False)

        weights = _apply_recency_weights(team_games["gameDate"], reference_date)
        if weights.empty or weights.sum() == 0:
            continue

        # Offensive output
        avg_pts = _weighted_mean(team_games[score_col], weights)
        volatility = _weighted_std(team_games[score_col], weights)
        if pd.isna(volatility):
            volatility = 10.0  # Default fallback

        # Simplified ORtg/DRtg as points scored/allowed per game for now
        if opp_score_col:
            avg_opp_pts = _weighted_mean(team_games[opp_score_col], weights)
        else:
            avg_opp_pts = 110.0  # league-like fallback

        # Pace proxy: total points / 2
        if opp_score_col:
            total_pts = team_games[score_col] + team_games[opp_score_col]
            pace = _weighted_mean(total_pts / 2.0, weights)
        else:
            pace = avg_pts

        ortg = avg_pts
        drtg = avg_opp_pts

        # SOS is placeholder here; can be wired later when opp strengths available
        sos = 1.0

        results.append(
            {
                "Team": team,
                "ORtg": round(ortg, 2),
                "DRtg": round(drtg, 2),
                "Pace": round(pace, 2),
                "Volatility": round(volatility, 2),
                "SOS": round(sos, 3),
                "Games": int(len(team_games)),
            }
        )

    strengths_df = pd.DataFrame(results)

    if verbose:
        print(f"[Strengths] Calculated ratings for {len(strengths_df)} teams.")

    # Optionally overlay live NBA API stats
    if use_live_overlay:
        strengths_df = _apply_live_overlay_to_strengths(
            strengths_df, live_weight=live_weight, verbose=verbose
        )

    return strengths_df


# =============================================================================
# ALTERNATIVE: DIRECTLY FROM NBA API (SEASON AGGREGATES)
# =============================================================================

def calculate_strengths_from_api(
    season: str = CURRENT_SEASON, verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate team strengths from NBA API only.

    Uses LeagueDashTeamStats + LeagueDashOpponentStats.
    No recency weighting here; just season aggregates.
    """
    if not HAS_NBA_API:
        if verbose:
            print("❌ nba_api not installed; cannot use API-only strengths.")
        return pd.DataFrame()

    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season, measure_type_detailed_defense="Advanced"
        )
        df = stats.get_data_frames()[0]

        opp_stats = leaguedashopponentstats.LeagueDashOpponentStats(season=season)
        opp_df = opp_stats.get_data_frames()[0]

        df = df.merge(
            opp_df[["TEAM_NAME", "OPP_OFF_RATING"]],
            on="TEAM_NAME",
            how="left",
        )

        league_avg_ortg = df["OFF_RATING"].mean()
        df["SOS"] = df["OPP_OFF_RATING"] / league_avg_ortg

        if "PTS_STD" in df.columns:
            df["Volatility"] = df["PTS_STD"]
        else:
            df["Volatility"] = 10.0

        result = df[
            ["TEAM_NAME", "OFF_RATING", "DEF_RATING", "PACE", "Volatility", "SOS"]
        ].copy()
        result.columns = ["Team", "ORtg", "DRtg", "Pace", "Volatility", "SOS"]

        if verbose:
            print(f"[Strengths API] Loaded {len(result)} teams from NBA API for {season}")

        return result

    except Exception as e:
        if verbose:
            print(f"❌ API error while computing strengths: {e}")
        return pd.DataFrame()


# =============================================================================
# CONVENIENCE WRAPPER
# =============================================================================

def get_team_strengths(
    source: str = "auto",
    cutoff_date: str = None,
    verbose: bool = True,
    use_live_overlay: bool = DEFAULT_USE_LIVE_OVERLAY,
    live_weight: float = DEFAULT_LIVE_WEIGHT,
) -> pd.DataFrame:
    """
    Get team strengths from the best available source.

    Parameters
    ----------
    source : str
        'auto' (default), 'csv', or 'api'
    cutoff_date : str
        For backtesting (only works with CSV source).
    verbose : bool
        Print progress.
    use_live_overlay : bool
        If True (and source uses CSV), overlay live NBA API stats from
        data/live/team_stats_live_nba_api.csv when available.
    live_weight : float
        Blend weight for live overlay.

    Returns
    -------
    DataFrame with team strength ratings.
    """
    source = (source or "auto").lower()

    if source == "api":
        return calculate_strengths_from_api(verbose=verbose)

    if source == "csv":
        return calculate_advanced_strengths(
            cutoff_date=cutoff_date,
            verbose=verbose,
            use_live_overlay=use_live_overlay,
            live_weight=live_weight,
        )

    # auto: prefer CSV (recency-weighted) + live overlay, fall back to pure API
    result = calculate_advanced_strengths(
        cutoff_date=cutoff_date,
        verbose=verbose,
        use_live_overlay=use_live_overlay,
        live_weight=live_weight,
    )

    if (result is None or result.empty) and HAS_NBA_API:
        if verbose:
            print("[Strengths] CSV-based strengths empty; falling back to NBA API.")
        result = calculate_strengths_from_api(verbose=verbose)

    return result