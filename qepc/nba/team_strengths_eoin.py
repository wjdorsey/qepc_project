"""
QEPC NBA: Team strength metrics built from Eoin-based team stats.

This takes the aggregated team_stats from eoin_team_stats.py and
builds a simple "advanced strengths" table QEPC can consume.

You can extend this later with pace, schedule-adjusted ratings, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qepc.utils.paths import get_project_root

from .eoin_team_stats import build_team_stats_from_eoin


def _safe_zscore(series: pd.Series) -> pd.Series:
    """
    Standard z-score with protection against zero std.
    """
    s = series.astype(float)
    mean = s.mean()
    std = s.std(ddof=0)
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index)
    return (s - mean) / std


def calculate_advanced_strengths_from_eoin(
    team_stats: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    cutoff_date: Optional[pd.Timestamp] = None,
    start_date: Optional[pd.Timestamp] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a QEPC-friendly team strengths table from Eoin data.

    Input: team_stats from build_team_stats_from_eoin(...)
      Columns expected:
        - team_id
        - games_played
        - wins
        - losses
        - win_pct
        - pts_for
        - pts_against
        - pts_diff
        - off_ppg
        - def_ppg

    Output: DataFrame with additional strength metrics:
        - pts_diff_per_game
        - z_win_pct
        - z_off_ppg
        - z_def_ppg (inverted so lower points allowed = higher z)
        - z_pts_diff_pg
        - strength_score (combined rating)
        - strength_rank (1 = strongest)
    """
    if team_stats is None:
        team_stats = build_team_stats_from_eoin(
            project_root=project_root,
            cutoff_date=cutoff_date,
            start_date=start_date,
        )

    df = team_stats.copy()

    required_cols = [
        "team_id",
        "games_played",
        "wins",
        "losses",
        "win_pct",
        "pts_for",
        "pts_against",
        "pts_diff",
        "off_ppg",
        "def_ppg",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"team_stats is missing required columns: {missing}. "
            "Check build_team_stats_from_eoin and your Eoin schema."
        )

    # Basic per-game differential
    df["pts_diff_per_game"] = df["pts_diff"] / df["games_played"]

    # Z-scores for key components
    df["z_win_pct"] = _safe_zscore(df["win_pct"])
    df["z_off_ppg"] = _safe_zscore(df["off_ppg"])
    # For defense, lower points allowed is better, so negate before z-scoring
    df["z_def_ppg"] = _safe_zscore(-df["def_ppg"])
    df["z_pts_diff_pg"] = _safe_zscore(df["pts_diff_per_game"])

    # Simple combined rating (you can tune these weights later)
    df["strength_score"] = (
        0.4 * df["z_win_pct"]
        + 0.3 * df["z_pts_diff_pg"]
        + 0.2 * df["z_off_ppg"]
        + 0.1 * df["z_def_ppg"]
    )

    # Rank: 1 = strongest
    df["strength_rank"] = df["strength_score"].rank(
        method="min", ascending=False
    ).astype(int)

    # Sort by rank for convenience
    df = df.sort_values("strength_rank").reset_index(drop=True)

    if verbose:
        print("Built advanced strengths from Eoin team_stats:")
        print(
            df[
                [
                    "team_id",
                    "games_played",
                    "win_pct",
                    "off_ppg",
                    "def_ppg",
                    "pts_diff_per_game",
                    "strength_score",
                    "strength_rank",
                ]
            ].head(10)
        )

    return df


def save_advanced_strengths_to_cache(
    strengths: Optional[pd.DataFrame] = None,
    project_root: Optional[Path] = None,
    filename: str = "eoin_team_strengths.parquet",
) -> Path:
    """
    Save the advanced team strengths to cache/imports as parquet.
    """
    if project_root is None:
        project_root = get_project_root()

    cache_dir = project_root / "cache" / "imports"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if strengths is None:
        strengths = calculate_advanced_strengths_from_eoin(
            project_root=project_root, verbose=False
        )

    out_path = cache_dir / filename
    strengths.to_parquet(out_path, index=False)
    print(f"Saved advanced strengths to: {out_path}")
    return out_path
