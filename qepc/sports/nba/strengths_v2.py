"""
QEPC NBA Team Strengths v2
==========================

Key features
------------
1. REAL volatility (std dev of game scores).
2. Recency weighting (recent games matter more).
3. Simple pace proxy.
4. Optional cutoff_date for backtesting "time travel".
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Tuple, Optional

import numpy as pd
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

RECENCY_HALF_LIFE_DAYS = 30  # Games from 30 days ago count ~50% as much
MIN_GAMES_REQUIRED = 5       # Minimum games for reliable stats
DEFAULT_VOLATILITY = 11.0    # NBA score std dev (roughly)
DEFAULT_PACE = 98.0          # League average possessions per game


# =============================================================================
# PATH HELPERS
# =============================================================================

def _get_project_root() -> Path:
    """
    Try to locate the QEPC project root.

    Priority:
    1. qepc.autoload.paths.get_project_root()
    2. Walk upwards looking for data/Team_Stats.csv
    3. Current working directory
    """
    try:
        from qepc.autoload.paths import get_project_root

        root = get_project_root()
        return Path(root) if not isinstance(root, Path) else root
    except Exception:
        pass

    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        candidate = parent / "data" / "Team_Stats.csv"
        if candidate.exists():
            return parent

    return Path.cwd()


def _detect_rating_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """Detect offensive and defensive rating column names by common patterns."""
    off_candidates = ["ORtg", "OffRtg", "Off_Rtg", "OffensiveRating"]
    def_candidates = ["DRtg", "DefRtg", "Def_Rtg", "DefensiveRating"]

    off_col = next((c for c in off_candidates if c in df.columns), None)
    def_col = next((c for c in def_candidates if c in df.columns), None)

    return off_col, def_col


# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

def _apply_recency_weights(
    df: pd.DataFrame,
    reference_date: pd.Timestamp,
    date_col: str = "gameDate",
) -> pd.DataFrame:
    """
    Apply exponential decay weights based on game recency.

    Weight = 0.5 ** (days_ago / RECENCY_HALF_LIFE_DAYS)

    Only games *on or before* reference_date are kept.
    """
    df = df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])

    if df.empty:
        return df

    # Normalize timezones
    if reference_date.tzinfo is not None:
        reference_date = reference_date.tz_convert(None)

    if pd.api.types.is_datetime64tz_dtype(df[date_col]):
        df[date_col] = df[date_col].dt.tz_convert(None)

    df["_days_ago"] = (reference_date - df[date_col]).dt.days
    df = df[df["_days_ago"] >= 0].copy()

    df["_weight"] = 0.5 ** (df["_days_ago"] / RECENCY_HALF_LIFE_DAYS)
    return df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted mean, handling empty/invalid cases."""
    valid_mask = values.notna() & weights.notna() & (weights > 0)
    if not valid_mask.any():
        return np.nan
    return np.average(values[valid_mask], weights=weights[valid_mask])


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    """Calculate weighted standard deviation."""
    valid_mask = values.notna() & weights.notna() & (weights > 0)
    if valid_mask.sum() < 2:
        return np.nan

    v = values[valid_mask].values
    w = weights[valid_mask].values

    weighted_mean = np.average(v, weights=w)
    weighted_var = np.average((v - weighted_mean) ** 2, weights=w)
    return np.sqrt(weighted_var)


# =============================================================================
# MAIN PUBLIC FUNCTION
# =============================================================================

def calculate_advanced_strengths(
    cutoff_date: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build team strengths with recency weighting and real volatility.

    Parameters
    ----------
    cutoff_date : str, optional
        Date string (YYYY-MM-DD) to use as "today" for backtesting.
        If None, uses the actual current date.
    verbose : bool
        Whether to print summary messages.

    Returns
    -------
    DataFrame
        Columns: Team, ORtg, DRtg, Pace, Volatility
    """
    project_root = _get_project_root()

    possible_paths = [
        project_root / "data" / "raw" / "Team_Stats.csv",
        project_root / "data" / "Team_Stats.csv",
        project_root / "data" / "raw" / "TeamStatistics.csv",
    ]

    team_stats_path: Optional[Path] = None
    for path in possible_paths:
        if path.exists():
            team_stats_path = path
            break

    if team_stats_path is None:
        raise FileNotFoundError(
            "Team stats file not found. Searched:\n"
            + "\n".join(f"  - {p}" for p in possible_paths)
        )

    team_stats = pd.read_csv(team_stats_path)
    cols = set(team_stats.columns)

    reference_date = pd.Timestamp(cutoff_date) if cutoff_date else pd.Timestamp.now()

    # -------------------------------------------------------------------------
    # CASE 1: Summary-style file (already aggregated by team)
    # -------------------------------------------------------------------------
    if "Team" in cols:
        off_col, def_col = _detect_rating_columns(team_stats)

        if off_col is None or def_col is None:
            raise ValueError(
                "Team_Stats.csv has a 'Team' column but is missing ORtg/DRtg.\n"
                f"Available columns: {list(team_stats.columns)}"
            )

        strengths = pd.DataFrame()
        strengths["Team"] = team_stats["Team"].astype(str).str.strip()
        strengths["ORtg"] = pd.to_numeric(team_stats[off_col], errors="coerce")
        strengths["DRtg"] = pd.to_numeric(team_stats[def_col], errors="coerce")

        strengths["ORtg"] = strengths["ORtg"].fillna(strengths["ORtg"].mean())
        strengths["DRtg"] = strengths["DRtg"].fillna(strengths["DRtg"].mean())

        if "Pace" in team_stats.columns:
            strengths["Pace"] = pd.to_numeric(team_stats["Pace"], errors="coerce")
            strengths["Pace"] = strengths["Pace"].fillna(DEFAULT_PACE)
        else:
            strengths["Pace"] = DEFAULT_PACE

        strengths["Volatility"] = DEFAULT_VOLATILITY

    # -------------------------------------------------------------------------
    # CASE 2: Game-level file (calculate everything from per-game results)
    # -------------------------------------------------------------------------
    elif {"teamCity", "teamName", "teamScore", "opponentScore"}.issubset(cols):
        team_stats["teamCity"] = team_stats["teamCity"].astype(str).str.strip()
        team_stats["teamName"] = team_stats["teamName"].astype(str).str.strip()
        team_stats["Team"] = team_stats["teamCity"] + " " + team_stats["teamName"]

        team_stats["teamScore"] = pd.to_numeric(team_stats["teamScore"], errors="coerce")
        team_stats["opponentScore"] = pd.to_numeric(
            team_stats["opponentScore"], errors="coerce"
        )

        if "gameDate" in team_stats.columns:
            team_stats["gameDate"] = pd.to_datetime(
                team_stats["gameDate"],
                errors="coerce",
            )
        else:
            warnings.warn(
                "No 'gameDate' column found; using equal weights for all games."
            )
            team_stats["gameDate"] = reference_date

        team_stats = _apply_recency_weights(team_stats, reference_date)

        if team_stats.empty:
            raise ValueError("No games found before the cutoff date.")

        results = []

        for team in team_stats["Team"].unique():
            team_games = team_stats[team_stats["Team"] == team]

            if len(team_games) < MIN_GAMES_REQUIRED:
                continue

            weights = team_games["_weight"]

            ortg = _weighted_mean(team_games["teamScore"], weights)
            drtg = _weighted_mean(team_games["opponentScore"], weights)

            volatility = _weighted_std(team_games["teamScore"], weights)
            if pd.isna(volatility) or volatility < 5:
                volatility = DEFAULT_VOLATILITY

            total_points = team_games["teamScore"] + team_games["opponentScore"]
            pace = _weighted_mean(total_points, weights) / 2.0

            results.append(
                {
                    "Team": team,
                    "ORtg": ortg,
                    "DRtg": drtg,
                    "Pace": pace,
                    "Volatility": volatility,
                    "Games": len(team_games),
                    "RecentGames": (team_games["_days_ago"] <= 14).sum(),
                }
            )

        strengths = pd.DataFrame(results)

        if strengths.empty:
            raise ValueError(
                f"No teams have {MIN_GAMES_REQUIRED}+ games before cutoff!"
            )

    else:
        raise ValueError(
            "Unrecognized Team_Stats.csv format.\n"
            f"Columns: {list(team_stats.columns)}"
        )

    strengths = strengths.sort_values("Team").reset_index(drop=True)

    if verbose:
        print(f"Built team strengths (cutoff: {reference_date.date()})")
        print(f"  Teams: {len(strengths)}")
        print(f"  ORtg range: {strengths['ORtg'].min():.1f} – {strengths['ORtg'].max():.1f}")
        print(f"  DRtg range: {strengths['DRtg'].min():.1f} – {strengths['DRtg'].max():.1f}")
        print(f"  Pace range: {strengths['Pace'].min():.1f} – {strengths['Pace'].max():.1f}")
        print(
            f"  Volatility range: "
            f"{strengths['Volatility'].min():.1f} – {strengths['Volatility'].max():.1f}"
        )

    return strengths[["Team", "ORtg", "DRtg", "Pace", "Volatility"]]


__all__ = ["calculate_advanced_strengths"]
