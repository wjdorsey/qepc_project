"""
Long-horizon NBA odds loader with resilient team-code mapping.

Loads the Kaggle odds dataset (nba_2008-2025.csv) and normalizes it into a QEPC-friendly table,
then attaches odds to the Eoin games parquet.

Key goals:
- No hardcoded machine/user paths (PROJECT_ROOT auto-detect)
- Robust team normalization (abbr, legacy abbr, and full team names)
- Stable join keys:
    game_date_join (Eastern-local calendar date, tz-naive) + home_team_id + away_team_id
- Diagnostics to understand coverage

Raw odds columns (your Kaggle file):
- season, date, regular, playoffs, away, home,
  score_away, score_home, whos_favored, spread, total,
  moneyline_away, moneyline_home, ...

Normalized output columns:
- season
- game_date (datetime64[ns] normalized midnight; derived from raw "date")
- game_key (YYYY-MM-DD_AWAY_HOME)
- away_code, home_code (canonical tricode)
- away_team_id, home_team_id
- score_away, score_home
- spread_home, spread_away
- total_points
- moneyline_away, moneyline_home
- p_away, p_home (vig-stripped implied win probabilities)
- regular, playoffs
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

def get_default_project_root() -> Path:
    return get_project_root(Path(__file__).resolve())


def get_default_odds_csv(project_root: Optional[Path] = None) -> Path:
    root = project_root or get_default_project_root()
    return root / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"


PROJECT_ROOT: Path = get_default_project_root()
DEFAULT_ODDS_CSV: Path = get_default_odds_csv()

# ---------------------------------------------------------------------
# Team normalization
# ---------------------------------------------------------------------

# Canonical tri-codes -> NBA franchise team_id (matches nba_api / Eoin IDs)
TEAM_CODE_TO_ID: Dict[str, int] = {
    "ATL": 1610612737,
    "BOS": 1610612738,
    "BKN": 1610612751,
    "CHA": 1610612766,
    "CHI": 1610612741,
    "CLE": 1610612739,
    "DAL": 1610612742,
    "DEN": 1610612743,
    "DET": 1610612765,
    "GSW": 1610612744,
    "HOU": 1610612745,
    "IND": 1610612754,
    "LAC": 1610612746,
    "LAL": 1610612747,
    "MEM": 1610612763,
    "MIA": 1610612748,
    "MIL": 1610612749,
    "MIN": 1610612750,
    "NOP": 1610612740,
    "NYK": 1610612752,
    "OKC": 1610612760,
    "ORL": 1610612753,
    "PHI": 1610612755,
    "PHX": 1610612756,
    "POR": 1610612757,
    "SAC": 1610612758,
    "SAS": 1610612759,
    "TOR": 1610612761,
    "UTA": 1610612762,
    "WAS": 1610612764,
}

# Common legacy/alternate abbreviations -> canonical tricode
TEAM_CODE_ALIASES: Dict[str, str] = {
    # Brooklyn / New Jersey
    "BRK": "BKN",
    "BKN": "BKN",
    "NJN": "BKN",
    "BK": "BKN",

    # New Orleans franchise
    "NO": "NOP",
    "NOP": "NOP",
    "NOH": "NOP",
    "NOK": "NOP",
    "NOLA": "NOP",

    # Golden State
    "GS": "GSW",
    "GSW": "GSW",

    # San Antonio
    "SA": "SAS",
    "SAS": "SAS",

    # New York
    "NY": "NYK",
    "NYK": "NYK",

    # Washington
    "WSH": "WAS",
    "WAS": "WAS",
    "WSB": "WAS",

    # Utah / Phoenix common alternates
    "UTAH": "UTA",
    "UTA": "UTA",
    "UTH": "UTA",
    "PHO": "PHX",
    "PHX": "PHX",

    # Charlotte / OKC legacy
    "CHA": "CHA",
    "CHO": "CHA",
    "CHH": "CHA",
    "SEA": "OKC",
    "OKC": "OKC",

    # Memphis legacy
    "VAN": "MEM",
    "MEM": "MEM",
}

TEAM_NAME_TO_CODE: Dict[str, str] = {
    "ATLANTA HAWKS": "ATL",
    "BOSTON CELTICS": "BOS",
    "BROOKLYN NETS": "BKN",
    "NEW JERSEY NETS": "BKN",
    "CHARLOTTE HORNETS": "CHA",
    "CHARLOTTE BOBCATS": "CHA",
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
    "LA LAKERS": "LAL",
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
    "PHILADELPHIA SIXERS": "PHI",
    "PHOENIX SUNS": "PHX",
    "PORTLAND TRAIL BLAZERS": "POR",
    "SACRAMENTO KINGS": "SAC",
    "SAN ANTONIO SPURS": "SAS",
    "TORONTO RAPTORS": "TOR",
    "UTAH JAZZ": "UTA",
    "WASHINGTON WIZARDS": "WAS",
}


def normalize_team_code(x: object) -> str | None:
    """
    Normalize a team token from the odds CSV into a canonical NBA tricode (e.g., SAS, GSW, NOP).

    Handles:
    - Abbrev-like tokens: "SA", "SAS", "GS", "GSW", "NO", "NOP", "UTAH", "WSH", etc.
    - Full names: "LOS ANGELES LAKERS", "NEW JERSEY NETS", etc. (via TEAM_NAME_TO_CODE)
    """
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None

    s = str(x).strip().upper()

    # Normalize punctuation/spaces
    s = s.replace(".", " ").replace(",", " ").replace("_", " ").replace("-", " ")
    s = " ".join(s.split())

    # Abbreviation-like path (most odds datasets)
    compact = s.replace(" ", "")
    if 2 <= len(compact) <= 5 and compact.isalpha():
        return TEAM_CODE_ALIASES.get(compact, compact)

    # Full-name path
    canon = TEAM_NAME_TO_CODE.get(s)
    if canon:
        return canon

    return None


# ---------------------------------------------------------------------
# Odds helpers
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
    return pd.read_csv(raw_csv)


def normalize_odds(odds_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw odds into a QEPC-friendly tidy table."""
    odds = odds_raw.copy()

    required = ["season", "date", "away", "home", "spread", "total", "moneyline_away", "moneyline_home"]
    missing = [c for c in required if c not in odds.columns]
    if missing:
        raise ValueError(f"Odds CSV missing required columns {missing}. Found: {list(odds.columns)}")

    # Dates: normalize to midnight (tz-naive). Odds "date" is already local-date style.
    odds["game_date"] = pd.to_datetime(odds["date"], errors="coerce").dt.normalize()
    if odds["game_date"].isna().all():
        raise ValueError("Odds date parsing failed: all game_date values are NaT.")

    # Team codes + IDs
    odds["home_code"] = odds["home"].apply(normalize_team_code)
    odds["away_code"] = odds["away"].apply(normalize_team_code)

    odds["home_team_id"] = odds["home_code"].map(TEAM_CODE_TO_ID).astype("Int64")
    odds["away_team_id"] = odds["away_code"].map(TEAM_CODE_TO_ID).astype("Int64")

    # Spreads + totals
    spreads = odds.apply(compute_home_away_spreads, axis=1, result_type="expand")
    odds["spread_home"] = spreads[0]
    odds["spread_away"] = spreads[1]
    odds["total_points"] = odds["total"]

    # Moneyline → implied probabilities (pre-vig), then vig-strip
    odds["p_home_raw"] = odds["moneyline_home"].apply(american_to_prob)
    odds["p_away_raw"] = odds["moneyline_away"].apply(american_to_prob)

    sum_raw = (odds["p_home_raw"] + odds["p_away_raw"]).replace({0: np.nan})
    odds["p_home"] = odds["p_home_raw"] / sum_raw
    odds["p_away"] = odds["p_away_raw"] / sum_raw

    # Game key for diagnostics
    odds["game_key"] = (
        odds["game_date"].dt.strftime("%Y-%m-%d")
        + "_"
        + odds["away_code"].fillna("UNK")
        + "_"
        + odds["home_code"].fillna("UNK")
    )

    # Optional flags if present
    if "regular" not in odds.columns:
        odds["regular"] = np.nan
    if "playoffs" not in odds.columns:
        odds["playoffs"] = np.nan
    if "score_away" not in odds.columns:
        odds["score_away"] = np.nan
    if "score_home" not in odds.columns:
        odds["score_home"] = np.nan

    cols = [
        "season",
        "game_date",
        "game_key",
        "away_code",
        "home_code",
        "away_team_id",
        "home_team_id",
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
    return odds[cols].copy()


def load_long_odds(raw_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Convenience function: load + normalize the long-horizon odds dataset."""
    return normalize_odds(load_raw_odds(raw_csv_path=raw_csv_path))


# Back-compat alias (useful because names drift between notebooks/scripts)
def load_odds_long(raw_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    return load_long_odds(raw_csv_path=raw_csv_path)


# ---------------------------------------------------------------------
# Mapping odds to games + diagnostics
# ---------------------------------------------------------------------


@dataclass
class OddsDiagnostics:
    matched_rows: int
    total_games: int
    unmatched_odds: int
    unmatched_games: int
    sample_unmatched_odds: pd.DataFrame
    sample_unmatched_games: pd.DataFrame
    # Extra (nice-to-have) fields for debugging:
    overlap_games: int
    overlap_matched: int


def _games_join_date_eastern_tznaive(games: pd.DataFrame) -> pd.Series:
    """
    Build the join date as Eastern-local calendar date (tz-naive, midnight).
    This avoids the pandas merge error (tz-aware vs tz-naive) and matches odds conventions.
    """
    if "game_datetime" in games.columns:
        dt = pd.to_datetime(games["game_datetime"], errors="coerce", utc=True)
        return dt.dt.tz_convert("US/Eastern").dt.normalize().dt.tz_localize(None)
    if "game_date" in games.columns:
        return pd.to_datetime(games["game_date"], errors="coerce").dt.normalize()
    raise ValueError("Games df missing game_datetime/game_date; cannot join odds.")


def attach_odds_to_games(
    games_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> tuple[pd.DataFrame, OddsDiagnostics]:
    """
    Attach normalized odds to games with diagnostics.

    Join keys:
        game_date_join (Eastern-local, tz-naive) + home_team_id + away_team_id

    Also includes a small ±1 day rescue fill to catch lingering calendar/date edge cases.
    """
    games = games_df.copy()
    odds = odds_df.copy()

    # --- Build join date keys ---
    games["game_date_join"] = _games_join_date_eastern_tznaive(games)
    odds["game_date_join"] = pd.to_datetime(odds["game_date"], errors="coerce").dt.normalize()

    # Align ID dtypes
    for c in ("home_team_id", "away_team_id"):
        if c not in games.columns:
            raise ValueError(f"Games df missing required column: {c}")
        games[c] = pd.to_numeric(games[c], errors="coerce").astype("Int64")
        odds[c] = pd.to_numeric(odds[c], errors="coerce").astype("Int64")

    key_cols = ["game_date_join", "home_team_id", "away_team_id"]

    # Deduplicate odds on join keys (keep last row)
    if odds.duplicated(subset=key_cols).any():
        odds = odds.sort_values(key_cols).drop_duplicates(subset=key_cols, keep="last")

    # Primary merge: keep odds game_date separately for clarity
    odds_for_merge = odds.rename(columns={"game_date": "odds_game_date"})
    merged = games.merge(
        odds_for_merge,
        on=key_cols,
        how="left",
        suffixes=("", "_odds"),
    )

    odds_fields = [c for c in ["total_points", "spread_home", "moneyline_home", "moneyline_away", "p_home", "p_away"] if c in merged.columns]
    matched_mask = merged[odds_fields].notna().any(axis=1) if odds_fields else pd.Series(False, index=merged.index)

    # --- Rescue pass: try ±1 day for rows that missed ---
    if (~matched_mask).any():
        missing_idx = merged.index[~matched_mask]

        # Build fast lookup table for odds fields only (prevents column chaos)
        odds_lookup = odds_for_merge.set_index(key_cols)

        def pull_shift(days: int) -> pd.DataFrame:
            tmp_keys = merged.loc[missing_idx, key_cols].copy()
            tmp_keys["game_date_join"] = tmp_keys["game_date_join"] + pd.Timedelta(days=days)
            pulled = odds_lookup.reindex(list(zip(tmp_keys["game_date_join"], tmp_keys["home_team_id"], tmp_keys["away_team_id"])))
            pulled.index = missing_idx
            return pulled

        pulled_plus = pull_shift(+1)
        pulled_minus = pull_shift(-1)

        # Fill only when the target field is missing
        for col in odds_for_merge.columns:
            if col in key_cols:
                continue
            if col not in merged.columns:
                merged[col] = np.nan

            base = merged.loc[missing_idx, col]
            merged.loc[missing_idx, col] = base.where(base.notna(), pulled_plus[col])
            base2 = merged.loc[missing_idx, col]
            merged.loc[missing_idx, col] = base2.where(base2.notna(), pulled_minus[col])

        # Recompute match mask after rescue
        matched_mask = merged[odds_fields].notna().any(axis=1) if odds_fields else pd.Series(False, index=merged.index)

    matched_rows = int(matched_mask.sum())
    unmatched_games = int((~matched_mask).sum())

    # Unmatched odds: keys not found in games keys
    game_keys = set(
        zip(
            games["game_date_join"].astype("datetime64[ns]").tolist(),
            games["home_team_id"].tolist(),
            games["away_team_id"].tolist(),
        )
    )
    odds_keys = list(
        zip(
            odds["game_date_join"].astype("datetime64[ns]").tolist(),
            odds["home_team_id"].tolist(),
            odds["away_team_id"].tolist(),
        )
    )
    odds_not_in_games_mask = pd.Series([k not in game_keys for k in odds_keys], index=odds.index)
    unmatched_odds_df = odds.loc[odds_not_in_games_mask].copy()

    # Overlap-focused metrics (more “honest” than counting 1946 games with no odds file)
    odds_min = odds["game_date_join"].min()
    odds_max = odds["game_date_join"].max()
    in_overlap = games["game_date_join"].between(odds_min, odds_max, inclusive="both")
    overlap_games = int(in_overlap.sum())
    overlap_matched = int((matched_mask & in_overlap).sum())

    diag = OddsDiagnostics(
        matched_rows=matched_rows,
        total_games=int(len(games)),
        unmatched_odds=int(len(unmatched_odds_df)),
        unmatched_games=unmatched_games,
        sample_unmatched_odds=unmatched_odds_df.head(5),
        sample_unmatched_games=merged.loc[~matched_mask].head(5),
        overlap_games=overlap_games,
        overlap_matched=overlap_matched,
    )

    return merged, diag


__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_ODDS_CSV",
    "TEAM_CODE_TO_ID",
    "TEAM_CODE_ALIASES",
    "TEAM_NAME_TO_CODE",
    "normalize_team_code",
    "load_raw_odds",
    "normalize_odds",
    "load_long_odds",
    "load_odds_long",
    "attach_odds_to_games",
    "OddsDiagnostics",
]


if __name__ == "__main__":
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("DEFAULT_ODDS_CSV:", DEFAULT_ODDS_CSV)
    o = load_long_odds()
    print("Loaded odds:", o.shape)
    print(o.head())

    # Optional quick merge smoke check if games parquet exists in your pipeline
    games_path = PROJECT_ROOT / "cache" / "imports" / "eoin_games_qepc.parquet"
    if games_path.exists():
        g = pd.read_parquet(games_path)
        merged, d = attach_odds_to_games(g, o)
        print(f"Matched {d.matched_rows} of {d.total_games} games.")
        print(f"Overlap matched {d.overlap_matched} of {d.overlap_games} games in odds date range.")
        print("Unmatched odds sample:")
        print(d.sample_unmatched_odds.head())
