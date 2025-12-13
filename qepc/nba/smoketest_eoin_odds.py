"""
QEPC NBA Smoketest — Eoin Games + Kaggle Odds (Long)

Goals:
- Verify project-root auto-detect (no hardcoded user paths)
- Load Eoin games parquet + odds CSV (normalized)
- Attach odds to games using stable join keys:
    (Eastern-local game_date_join, home_team_id, away_team_id)
- Report *overall* coverage AND *overlap-range* coverage (no fake panic)
- Diagnose overlap-missing odds (often preseason/exhibitions)

Run:
    python -m qepc.nba.smoketest_eoin_odds
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from qepc.utils.paths import get_project_root
from qepc.nba.odds_long_loader import load_long_odds, attach_odds_to_games


def get_default_project_root() -> Path:
    return get_project_root(Path(__file__).resolve())


PROJECT_ROOT: Path = get_default_project_root()

GAMES_PARQUET_DEFAULT = PROJECT_ROOT / "cache" / "imports" / "eoin_games_qepc.parquet"
ODDS_CSV_DEFAULT = PROJECT_ROOT / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"


@dataclass
class OverlapDiagnostics:
    overlap_games: int
    overlap_matched: int
    overlap_missing: int
    preseason_like_missing: int
    sample_overlap_missing: pd.DataFrame
    missing_by_date_head: pd.DataFrame


def _print_range(label: str, s: pd.Series) -> None:
    s = pd.to_datetime(s, errors="coerce")
    print(f"[smoketest] {label}: {s.min()}  {s.max()}")


def _compute_overlap_diagnostics(merged: pd.DataFrame, odds_min: pd.Timestamp, odds_max: pd.Timestamp) -> OverlapDiagnostics:
    # matched mask: odds_long_loader attaches 'game_key' from odds; missing odds => NaN
    if "game_key" in merged.columns:
        matched_mask = merged["game_key"].notna()
    else:
        # fallback: any of these being present counts as "matched"
        odds_fields = [c for c in ["total_points", "spread_home", "moneyline_home", "moneyline_away", "p_home", "p_away"] if c in merged.columns]
        matched_mask = merged[odds_fields].notna().any(axis=1) if odds_fields else pd.Series(False, index=merged.index)

    if "game_date_join" not in merged.columns:
        raise ValueError("merged is missing 'game_date_join'. Did attach_odds_to_games change?")

    in_overlap = merged["game_date_join"].between(odds_min, odds_max, inclusive="both")
    overlap_df = merged.loc[in_overlap].copy()

    overlap_games = int(len(overlap_df))
    overlap_matched = int(matched_mask.loc[overlap_df.index].sum())
    overlap_missing = overlap_games - overlap_matched

    missing_df = overlap_df.loc[~matched_mask.loc[overlap_df.index]].copy()

    # Heuristic: preseason/exhibition game_ids often look like 1240xxxx / 1230xxxx etc.
    # This isn't perfect, but it’s a very strong signal for the dates you showed (Oct 17/18 preseason window).
    gid = missing_df.get("game_id")
    if gid is not None:
        gid_str = gid.astype(str)
        preseason_like = gid_str.str.match(r"^1\d{7}$")  # 8 digits starting with 1
        preseason_like_missing = int(preseason_like.sum())
    else:
        preseason_like_missing = 0

    sample_cols = [c for c in ["game_id", "game_date_join", "away_team_id", "home_team_id", "away_team_name", "home_team_name"] if c in missing_df.columns]
    sample_overlap_missing = missing_df[sample_cols].head(10)

    missing_by_date = (
        missing_df.groupby("game_date_join", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(15)
        .reset_index(name="missing_games")
    )

    return OverlapDiagnostics(
        overlap_games=overlap_games,
        overlap_matched=overlap_matched,
        overlap_missing=overlap_missing,
        preseason_like_missing=preseason_like_missing,
        sample_overlap_missing=sample_overlap_missing,
        missing_by_date_head=missing_by_date,
    )


def main() -> None:
    print(f"[smoketest] PROJECT_ROOT = {PROJECT_ROOT}")

    games_path = GAMES_PARQUET_DEFAULT
    odds_path = ODDS_CSV_DEFAULT

    if not games_path.exists():
        raise FileNotFoundError(
            f"Eoin games parquet not found:\n  {games_path}\n\n"
            "Expected you to have run the Eoin fetch/build notebook that writes:\n"
            "  cache/imports/eoin_games_qepc.parquet"
        )

    if not odds_path.exists():
        raise FileNotFoundError(
            f"Odds CSV not found:\n  {odds_path}\n\n"
            "Expected:\n"
            "  data/raw/nba/odds_long/nba_2008-2025.csv"
        )

    games = pd.read_parquet(games_path)
    odds = load_long_odds(odds_path)

    # Ranges (for sanity)
    if "game_date" in games.columns:
        _print_range("games", games["game_date"])
    elif "game_datetime" in games.columns:
        _print_range("games", pd.to_datetime(games["game_datetime"], errors="coerce", utc=True).dt.date)
    _print_range("odds ", odds["game_date"])

    merged, diag = attach_odds_to_games(games, odds)

    print(
        f"[smoketest] Odds coverage: matched {diag.matched_rows} of {diag.total_games} games; "
        f"{diag.unmatched_games} games missing odds; {diag.unmatched_odds} odds rows unmatched."
    )

    # Show a few merged rows (most recent rows often have NaNs if odds dataset ends earlier)
    print("[smoketest] Sample merged rows:")
    print(merged.head(5))

    print("[smoketest] Unmatched games sample:")
    print(diag.sample_unmatched_games.head(5))

    print("[smoketest] Unmatched odds sample:")
    print(diag.sample_unmatched_odds.head(5))

    # Overlap diagnostics: only judge join quality where odds *could* exist
    odds_min = pd.to_datetime(odds["game_date"], errors="coerce").min()
    odds_max = pd.to_datetime(odds["game_date"], errors="coerce").max()
    overlap = _compute_overlap_diagnostics(merged, odds_min, odds_max)

    print(
        f"[smoketest] Overlap truth-serum: matched {overlap.overlap_matched} of {overlap.overlap_games} "
        f"(missing {overlap.overlap_missing})."
    )
    if overlap.overlap_missing > 0:
        print(
            f"[smoketest] Overlap-missing preseason-like (heuristic game_id ^1xxxxxxx): "
            f"{overlap.preseason_like_missing} of {overlap.overlap_missing}"
        )
        print("[smoketest] Overlap missing odds sample:")
        print(overlap.sample_overlap_missing)

        print("[smoketest] Overlap missing odds — most common dates (top 15):")
        print(overlap.missing_by_date_head)


if __name__ == "__main__":
    main()
