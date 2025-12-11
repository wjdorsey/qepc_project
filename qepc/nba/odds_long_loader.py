"""
QEPC NBA Module: odds_long_loader.py

Loads and normalizes long-horizon NBA betting data (2007–2024+)
from the Kaggle dataset "NBA Betting Data | October 2007 to June 2024".

Output is a tidy DataFrame with:
- game_date
- home_code / away_code (3–4 letter codes from the dataset)
- score_home / score_away
- spread_home / spread_away (from home POV)
- total_points (closing total)
- moneyline_home / moneyline_away
- p_home / p_away (vig-stripped implied win probs)
- regular / playoffs flags
- game_key (string key useful for joining)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------


def get_project_root() -> Path:
    """
    Resolve the QEPC project root assuming this file lives at:
        <PROJECT_ROOT>/qepc/nba/odds_long_loader.py
    """
    here = Path(__file__).resolve()
    # ... /qepc/nba/odds_long_loader.py
    # parents[0] -> nba
    # parents[1] -> qepc
    # parents[2] -> PROJECT_ROOT
    project_root = here.parents[2]
    return project_root


PROJECT_ROOT: Path = get_project_root()

# Default location where you should place the Kaggle CSV.
# Feel free to change the filename if yours is different.
DEFAULT_ODDS_CSV: Path = (
    PROJECT_ROOT / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def american_to_prob(odds_american: float | int | None) -> float:
    """
    Convert American odds to implied probability (pre-vig).

    Example:
        -140 -> ~0.58
        +200 -> ~0.333

    Returns np.nan for missing odds.
    """
    if odds_american is None or pd.isna(odds_american):
        return np.nan

    o = float(odds_american)
    if o < 0:
        # Negative odds: risk |o| to win 100
        return (-o) / ((-o) + 100.0)
    else:
        # Positive odds: risk 100 to win o
        return 100.0 / (o + 100.0)


def compute_home_away_spreads(row: pd.Series) -> tuple[float, float]:
    """
    From whos_favored + spread, compute spread_home / spread_away.

    Convention:
    - spread_home < 0 when home is favored (e.g. -6.5)
    - spread_home > 0 when home is an underdog
    - spread_away is the opposite sign.

    If data is missing or ambiguous, returns (np.nan, np.nan).
    """
    fav = row.get("whos_favored")
    s = row.get("spread")

    if pd.isna(s) or fav not in ("home", "away"):
        return np.nan, np.nan

    s = float(s)
    if fav == "home":
        # Home is favored by s points → spread_home = -s
        return -s, s
    else:
        # Away is favored by s points → spread_home = +s
        return s, -s


# ---------------------------------------------------------------------
# Core load + normalize functions
# ---------------------------------------------------------------------


def load_raw_odds(
    raw_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load the raw odds CSV from Kaggle.

    Parameters
    ----------
    raw_csv_path : str or Path, optional
        If provided, read from this path.
        If None, uses DEFAULT_ODDS_CSV under the project root.

    Returns
    -------
    pd.DataFrame
        Raw odds DataFrame with columns like:
        season, date, away, home, score_away, score_home, whos_favored,
        spread, total, moneyline_away, moneyline_home, ...
    """
    if raw_csv_path is None:
        raw_csv = DEFAULT_ODDS_CSV
    else:
        raw_csv = Path(raw_csv_path)

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"NBA odds CSV not found at: {raw_csv}\n"
            f"Either place the Kaggle file there, or pass raw_csv_path explicitly."
        )

    df = pd.read_csv(raw_csv)
    return df


def normalize_odds(odds_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the raw odds DataFrame into a tidy, QEPC-friendly table.

    Returns
    -------
    pd.DataFrame with columns:
        - season
        - game_date (python date)
        - game_key (str: date_awaycode_homecode)
        - away_code, home_code (uppercased team codes)
        - score_away, score_home
        - spread_home, spread_away
        - total_points
        - moneyline_away, moneyline_home
        - p_away, p_home (vig-stripped implied probabilities)
        - regular, playoffs
    """
    odds = odds_raw.copy()

    # 1) Normalize date + team codes
    odds["game_date"] = pd.to_datetime(odds["date"]).dt.date
    odds["home_code"] = odds["home"].astype(str).str.upper()
    odds["away_code"] = odds["away"].astype(str).str.upper()

    # 2) Compute spreads from whos_favored + spread
    home_spreads: list[float] = []
    away_spreads: list[float] = []

    for _, r in odds.iterrows():
        h_s, a_s = compute_home_away_spreads(r)
        home_spreads.append(h_s)
        away_spreads.append(a_s)

    odds["spread_home"] = home_spreads
    odds["spread_away"] = away_spreads

    # 3) Closing total
    odds["total_points"] = odds["total"]

    # 4) Moneyline → implied probabilities (pre-vig)
    odds["p_home_raw"] = odds["moneyline_home"].apply(american_to_prob)
    odds["p_away_raw"] = odds["moneyline_away"].apply(american_to_prob)

    # Remove vig via normalization
    sum_raw = odds["p_home_raw"] + odds["p_away_raw"]
    # Protect against division by zero
    sum_raw = sum_raw.replace({0: np.nan})
    odds["p_home"] = odds["p_home_raw"] / sum_raw
    odds["p_away"] = odds["p_away_raw"] / sum_raw

    # 5) Game key (useful for joining, though you'll likely map codes → names/ids)
    odds["game_key"] = (
        odds["game_date"].astype(str)
        + "_"
        + odds["away_code"]
        + "_"
        + odds["home_code"]
    )

    # 6) Select tidy subset
    cols = [
        "season",
        "game_date",
        "game_key",
        "away_code",
        "home_code",
        "score_away",
        "score_home",
        "spread_home",
        "spread_away",
        "total_points",
        "moneyline_away",
        "moneyline_home",
        "p_away",
        "p_home",
        "regular",
        "playoffs",
    ]

    missing = [c for c in cols if c not in odds.columns]
    if missing:
        raise ValueError(f"normalize_odds: missing expected columns: {missing}")

    odds_tidy = odds[cols].copy()
    return odds_tidy


def load_long_odds(
    raw_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Convenience function: load + normalize the long-horizon odds dataset.

    Parameters
    ----------
    raw_csv_path : str or Path, optional
        Custom path to the Kaggle CSV. If None, uses DEFAULT_ODDS_CSV.

    Returns
    -------
    pd.DataFrame
        The normalized odds_tidy table.
    """
    odds_raw = load_raw_odds(raw_csv_path=raw_csv_path)
    odds_tidy = normalize_odds(odds_raw)
    return odds_tidy


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEFAULT_ODDS_CSV:", DEFAULT_ODDS_CSV)

    df = load_long_odds()
    print("Loaded odds_tidy:", df.shape)
    print(df.head())
