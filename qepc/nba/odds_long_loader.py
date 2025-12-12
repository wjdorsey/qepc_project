"""Long-horizon NBA odds loader with resilient team-code mapping.

Outputs a tidy odds table with columns like:
- season
- game_date
- game_key
- home_code / away_code
- score_home / score_away
- spread_home / spread_away
- total_points
- moneyline_home / moneyline_away
- p_home / p_away (vig-stripped implied win probs)
- regular / playoffs flags
- join diagnostics helpers
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

from qepc.utils.paths import get_project_root

# ---------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------

PROJECT_ROOT: Path = get_project_root(Path(__file__).resolve())
DEFAULT_ODDS_CSV: Path = (
    PROJECT_ROOT / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
)

# ---------------------------------------------------------------------
# Team code normalization
# ---------------------------------------------------------------------

TEAM_CODE_ALIASES: Dict[str, str] = {
    "NJN": "BRK",
    "NOH": "NOP",
    "NOK": "NOP",
    "CHA": "CHO",
    "CHH": "CHO",
    "SEA": "OKC",
    "VAN": "MEM",
    "WSB": "WAS",
    "UTH": "UTA",
}

TEAM_NAME_TO_CODE: Dict[str, str] = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BRK",
    "NEW JERSEY NETS": "BRK",
    "CHARLOTTE HORNETS": "CHO",
    "CHARLOTTE BOBCATS": "CHO",
    "CHICAGO BULLS": "CHI",
    "CLEVELAND CAVALIERS": "CLE",
    "DALLAS MAVERICKS": "DAL",
    "DENVER NUGGETS": "DEN",
    "DETROIT PISTONS": "DET",
    "GOLDEN STATE WARRIORS": "GSW",
    "HOUSTON ROCKETS": "HOU",
    "INDIANA PACERS": "IND",
    "LOS ANGELES CLIPPERS": "LAC",
    "LA CLIPPERS": "LAC",
    "LOS ANGELES LAKERS": "LAL",
    "MEMPHIS GRIZZLIES": "MEM",
    "MIAMI HEAT": "MIA",
    "MILWAUKEE BUCKS": "MIL",
    "MINNESOTA TIMBERWOLVES": "MIN",
    "NEW ORLEANS HORNETS": "NOP",
    "NEW ORLEANS PELICANS": "NOP",
    "NEW YORK KNICKS": "NYK",
    "OKLAHOMA CITY THUNDER": "OKC",
    "ORLANDO MAGIC": "ORL",
    "PHILADELPHIA 76ERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}


def normalize_team_code(code: str | float | int | None) -> str | None:
    if code is None or pd.isna(code):
        return None
    cleaned = str(code).strip().upper()
    cleaned = TEAM_CODE_ALIASES.get(cleaned, cleaned)
    return cleaned


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def american_to_prob(odds_american: float | int | None) -> float:
    """Convert American odds to implied probability (pre-vig)."""

    if odds_american is None or pd.isna(odds_american):
        return np.nan

    o = float(odds_american)
    if o < 0:
        return (-o) / ((-o) + 100.0)
    return 100.0 / (o + 100.0)


def compute_home_away_spreads(row: pd.Series) -> tuple[float, float]:
    """From whos_favored + spread, compute spread_home / spread_away."""

    fav = row.get("whos_favored")
    s = row.get("spread")

    if pd.isna(s) or fav not in ("home", "away"):
        return np.nan, np.nan

    s = float(s)
    if fav == "home":
        return -s, s
    return s, -s


# ---------------------------------------------------------------------
# Core load + normalize functions
# ---------------------------------------------------------------------


def load_raw_odds(raw_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load the raw odds CSV from Kaggle."""

    raw_csv = Path(raw_csv_path) if raw_csv_path is not None else DEFAULT_ODDS_CSV

    if not raw_csv.exists():
        raise FileNotFoundError(
            f"NBA odds CSV not found at: {raw_csv}\n"
            "Either place the Kaggle file there, or pass raw_csv_path explicitly."
        )

    df = pd.read_csv(raw_csv)
    return df


def normalize_odds(odds_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize the raw odds DataFrame into a tidy, QEPC-friendly table.
    """

    odds = odds_raw.copy()

    # 1) Normalize date + team codes
    odds["game_date"] = pd.to_datetime(odds["date"]).dt.date
    odds["home_code"] = odds["home"].apply(normalize_team_code)
    odds["away_code"] = odds["away"].apply(normalize_team_code)

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

    sum_raw = odds["p_home_raw"] + odds["p_away_raw"]
    sum_raw = sum_raw.replace({0: np.nan})
    odds["p_home"] = odds["p_home_raw"] / sum_raw
    odds["p_away"] = odds["p_away_raw"] / sum_raw

    # 5) Game key (useful for diagnostics)
    odds["game_key"] = (
        odds["game_date"].astype(str)
        + "_"
        + odds["away_code"].astype(str)
        + "_"
        + odds["home_code"].astype(str)
    )

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


def load_long_odds(raw_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convenience function: load + normalize the long-horizon odds dataset."""

    odds_raw = load_raw_odds(raw_csv_path=raw_csv_path)
    odds_tidy = normalize_odds(odds_raw)
    return odds_tidy


# ---------------------------------------------------------------------
# Mapping odds to games
# ---------------------------------------------------------------------


def _detect_code_column(df: pd.DataFrame, prefix: str) -> str | None:
    candidates = [
        f"{prefix}_team_tricode",
        f"{prefix}teamtricode",
        f"{prefix}_team_abbrev",
        f"{prefix}teamabbrev",
        f"{prefix}_team_code",
        f"{prefix}teamcode",
    ]
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _detect_name_column(df: pd.DataFrame, prefix: str) -> str | None:
    candidates = [
        f"{prefix}_team_name",
        f"{prefix}teamname",
        f"{prefix}_team",
    ]
    for cand in candidates:
        if cand in df.columns:
            return cand
    return None


def _normalize_game_codes(games_df: pd.DataFrame, prefix: str) -> pd.Series:
    code_col = _detect_code_column(games_df, prefix)
    name_col = _detect_name_column(games_df, prefix)

    if code_col is not None:
        return games_df[code_col].apply(normalize_team_code)

    if name_col is not None:
        return (
            games_df[name_col]
            .astype(str)
            .str.upper()
            .map(TEAM_NAME_TO_CODE)
        )

    return pd.Series([None] * len(games_df), index=games_df.index)


def build_team_lookup_from_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build a mapping of team_code → team_id from games."""

    frames = []
    for prefix in ("home", "away"):
        id_col = f"{prefix}_team_id"
        if id_col not in games_df.columns:
            continue
        codes = _normalize_game_codes(games_df, prefix)
        chunk = pd.DataFrame(
            {
                "team_id": games_df[id_col],
                "team_code": codes,
            }
        )
        frames.append(chunk)

    if not frames:
        return pd.DataFrame(columns=["team_id", "team_code"])

    lookup = pd.concat(frames, ignore_index=True)
    lookup = lookup.dropna(subset=["team_id", "team_code"])
    lookup["team_code"] = lookup["team_code"].apply(normalize_team_code)
    lookup = lookup.drop_duplicates(subset=["team_id", "team_code"])
    return lookup


@dataclass
class OddsDiagnostics:
    matched_rows: int
    total_games: int
    unmatched_odds: int
    unmatched_games: int
    sample_unmatched_odds: pd.DataFrame
    sample_unmatched_games: pd.DataFrame


def attach_odds_to_games(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, OddsDiagnostics]:
    """Attach normalized odds to games with diagnostics."""

    games = games_df.copy()
    games["game_date"] = pd.to_datetime(games["game_date"]).dt.date

    odds = odds_df.copy()
    odds["game_date"] = pd.to_datetime(odds["game_date"]).dt.date
    odds["home_code"] = odds["home_code"].apply(normalize_team_code)
    odds["away_code"] = odds["away_code"].apply(normalize_team_code)

    lookup = build_team_lookup_from_games(games)
    code_to_team_id = dict(zip(lookup["team_code"], lookup["team_id"]))

    odds["home_team_id"] = odds["home_code"].map(code_to_team_id)
    odds["away_team_id"] = odds["away_code"].map(code_to_team_id)

    merged = games.merge(
        odds,
        on=["game_date", "home_team_id", "away_team_id"],
        how="left",
        suffixes=("", "_odds"),
    )

    matched_rows = merged[~merged["game_key"].isna()].shape[0]
    unmatched_odds = odds[odds[["home_team_id", "away_team_id"]].isna().any(axis=1)].shape[0]
    unmatched_games = merged[merged["game_key"].isna()].shape[0]

    diag = OddsDiagnostics(
        matched_rows=matched_rows,
        total_games=len(games),
        unmatched_odds=unmatched_odds,
        unmatched_games=unmatched_games,
        sample_unmatched_odds=odds[odds[["home_team_id", "away_team_id"]].isna()].head(5),
        sample_unmatched_games=merged[merged["game_key"].isna()].head(5),
    )

    return merged, diag


# ---------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEFAULT_ODDS_CSV:", DEFAULT_ODDS_CSV)

    try:
        df = load_long_odds()
        print("Loaded odds_tidy:", df.shape)
        print(df.head())
    except FileNotFoundError as exc:
        print(exc)
