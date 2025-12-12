"""Quick smoke test for Eoin parquets + odds merge."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from qepc.utils.paths import get_project_root
from .eoin_data_source import load_eoin_games
from .odds_long_loader import attach_odds_to_games, load_long_odds


def _safe_load_games() -> pd.DataFrame | None:
    try:
        return load_eoin_games()
    except FileNotFoundError as exc:
        print(f"[smoketest] Games parquet missing: {exc}")
    except ImportError as exc:
        print(f"[smoketest] Cannot read parquet (install pyarrow/fastparquet): {exc}")
    except Exception as exc:  # pragma: no cover - guardrail only
        print(f"[smoketest] Unexpected error loading games: {exc}")
    return None


def _safe_load_odds() -> pd.DataFrame | None:
    try:
        return load_long_odds()
    except FileNotFoundError as exc:
        print(f"[smoketest] Odds CSV missing: {exc}")
    except Exception as exc:  # pragma: no cover - guardrail only
        print(f"[smoketest] Unexpected error loading odds: {exc}")
    return None


def main() -> int:
    project_root = get_project_root(Path(__file__).resolve())
    print(f"[smoketest] PROJECT_ROOT = {project_root}")

    games = _safe_load_games()
    if games is None:
        print("[smoketest] Skipping merge because games could not be loaded.")
        return 0

    odds = _safe_load_odds()
    if odds is None:
        print("[smoketest] Skipping merge because odds could not be loaded.")
        return 0

    merged, diag = attach_odds_to_games(games, odds)
    print(
        f"[smoketest] Odds coverage: matched {diag.matched_rows} of {diag.total_games} games; "
        f"{diag.unmatched_games} games missing odds; {diag.unmatched_odds} odds rows unmatched."
    )

    print("[smoketest] Sample merged rows:")
    print(merged.head())

    if diag.sample_unmatched_games is not None:
        print("[smoketest] Unmatched games sample:")
        print(diag.sample_unmatched_games.head())

    if diag.sample_unmatched_odds is not None:
        print("[smoketest] Unmatched odds sample:")
        print(diag.sample_unmatched_odds.head())

    return 0


if __name__ == "__main__":  # pragma: no cover - script entrypoint
    sys.exit(main())
