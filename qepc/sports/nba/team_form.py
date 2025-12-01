"""
QEPC NBA Team Form Adjustments
------------------------------

Use TeamForm.csv to gently warp team offensive strength (ORtg)
based on recent performance, while keeping the quantum core
(lambdas + multiverse) as the main engine.

Inputs:
    - strengths_df from strengths_v2.get_team_strengths()
      (must have columns: ['Team', 'ORtg', 'DRtg', 'Pace', ...])
    - data/TeamForm.csv

Outputs:
    - strengths_df with ORtg adjusted by a "form boost"
      and extra columns:
        * form_boost
        * form_ppg
        * form_last_game_date
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qepc.core.model_config import LEAGUE_AVG_POINTS


def _get_data_dir() -> Path:
    """
    Resolve QEPC data directory using autoload helpers if available,
    otherwise fall back to project_root/data.
    """
    try:
        from qepc.autoload.paths import get_data_dir  # type: ignore

        return get_data_dir()
    except Exception:
        cwd = Path.cwd()
        if cwd.name == "notebooks":
            project_root = cwd.parent
        else:
            project_root = cwd
        return project_root / "data"


def load_team_form(verbose: bool = True) -> pd.DataFrame:
    """
    Load TeamForm.csv and normalize columns.

    Expected columns:
        - Team
        - Last_Game_Date
        - Last_N_PPG
        - Last_N_OPPG
        - Last_N_Wins
        - Last_N_Win_Pct
        - Games_Count
    """
    data_dir = _get_data_dir()
    path = data_dir / "TeamForm.csv"

    if verbose:
        print(f"[TeamForm] Loading TeamForm from: {path}")

    if not path.exists():
        raise FileNotFoundError(f"TeamForm.csv not found at {path}")

    tf = pd.read_csv(path)

    # Robust date parsing
    if "Last_Game_Date" in tf.columns:
        tf["Last_Game_Date"] = pd.to_datetime(tf["Last_Game_Date"], errors="coerce")
    else:
        raise RuntimeError("TeamForm.csv missing 'Last_Game_Date' column")

    required = [
        "Team",
        "Last_N_PPG",
        "Last_N_OPPG",
        "Last_N_Wins",
        "Last_N_Win_Pct",
        "Games_Count",
    ]
    missing = [c for c in required if c not in tf.columns]
    if missing:
        raise RuntimeError(f"TeamForm.csv missing required columns: {missing}")

    # For each team, keep the most recent row (latest Last_Game_Date)
    tf = (
        tf.sort_values("Last_Game_Date")
        .dropna(subset=["Last_Game_Date"])
        .drop_duplicates(subset=["Team"], keep="last")
        .reset_index(drop=True)
    )

    if verbose and not tf.empty:
        print(f"[TeamForm] Loaded recent form for {len(tf)} teams.")
        print(
            "[TeamForm] Date range:",
            tf["Last_Game_Date"].min().date(),
            "to",
            tf["Last_Game_Date"].max().date(),
        )

    return tf


def apply_team_form_boost(
    strengths_df: pd.DataFrame,
    team_form_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.35,
    max_boost_pct: float = 0.10,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Warp team ORtg based on recent scoring form.

    Concept:
        form_ppg = Last_N_PPG
        delta = (form_ppg - LEAGUE_AVG_POINTS) / LEAGUE_AVG_POINTS
        raw_boost = 1 + alpha * delta

        Clamp boost to [1 - max_boost_pct, 1 + max_boost_pct]
        e.g. max_boost_pct = 0.10 => between 0.90x and 1.10x

    Parameters
    ----------
    strengths_df : DataFrame
        Output from get_team_strengths(...). Must include 'Team' and 'ORtg'.
    team_form_df : DataFrame, optional
        Preloaded TeamForm; if None, load via load_team_form().
    alpha : float
        Sensitivity of ORtg to recent scoring vs league average.
    max_boost_pct : float
        Maximum up/down adjustment (e.g. 0.10 => ±10%).
    verbose : bool

    Returns
    -------
    DataFrame
        strengths_df with adjusted ORtg and columns:
            - form_boost
            - form_ppg
            - form_last_game_date
    """
    if strengths_df is None or strengths_df.empty:
        if verbose:
            print("[TeamForm] strengths_df is empty; no adjustments applied.")
        return strengths_df

    if "Team" not in strengths_df.columns or "ORtg" not in strengths_df.columns:
        raise RuntimeError("strengths_df must have 'Team' and 'ORtg' columns")

    if team_form_df is None:
        team_form_df = load_team_form(verbose=verbose)

    if team_form_df is None or team_form_df.empty:
        if verbose:
            print("[TeamForm] No team_form data; returning strengths_df unchanged.")
        strengths_df = strengths_df.copy()
        strengths_df["form_boost"] = 1.0
        strengths_df["form_ppg"] = np.nan
        strengths_df["form_last_game_date"] = pd.NaT
        return strengths_df

    df = strengths_df.copy()

    # Merge on Team name
    merged = df.merge(
        team_form_df[
            [
                "Team",
                "Last_Game_Date",
                "Last_N_PPG",
                "Last_N_OPPG",
                "Last_N_Wins",
                "Last_N_Win_Pct",
                "Games_Count",
            ]
        ],
        on="Team",
        how="left",
    )

    # Compute form-based scoring delta
    form_ppg = merged["Last_N_PPG"]
    delta = (form_ppg - LEAGUE_AVG_POINTS) / LEAGUE_AVG_POINTS

    # For teams without form data, treat delta as 0
    delta = delta.fillna(0.0)

    # Raw boost and clamp
    raw_boost = 1.0 + alpha * delta
    lower = 1.0 - max_boost_pct
    upper = 1.0 + max_boost_pct
    boost = raw_boost.clip(lower, upper)

    # Apply to ORtg only (offense heats/cools faster than defense)
    merged["ORtg"] = merged["ORtg"] * boost
    merged["form_boost"] = boost
    merged["form_ppg"] = form_ppg
    merged["form_last_game_date"] = merged["Last_Game_Date"]

    if verbose:
        print("[TeamForm] Applied form boost to ORtg.")
        print(
            f"[TeamForm] boost range: {boost.min():.3f} to {boost.max():.3f} "
            f"(alpha={alpha}, max_boost_pct={max_boost_pct:.0%})"
        )
        print(
            "[TeamForm] Example teams:\n",
            merged[["Team", "ORtg", "form_boost", "form_ppg"]]
            .head(8)
            .to_string(index=False),
        )

    # Drop helper columns we don't need downstream
    merged = merged.drop(columns=["Last_Game_Date"], errors="ignore")

    return merged
