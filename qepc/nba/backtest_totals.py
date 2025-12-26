"""CLI backtest runner for QEPC NBA totals.

Example:
    python -m qepc.nba.backtest_totals --start 2022-10-01 --with-odds
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from qepc.nba.eoin_data_source import load_eoin_games
from qepc.nba.league_field import apply_env_drift, compute_env_drift
from qepc.nba.odds_long_loader import attach_odds_to_games, load_long_odds
from qepc.nba.qpa_totals import (
    apply_script_superposition,
    collapse_total_with_odds,
    overdispersed_total_distribution,
)
from qepc.utils.paths import get_project_root


def _basic_expected_totals(games: pd.DataFrame, window: int = 10) -> pd.Series:
    """Very simple leakage-safe baseline: team rolling for/against means."""
    df = games.sort_values("game_datetime").copy()

    for side, opp in (("home", "away"), ("away", "home")):
        scored_col = f"{side}_score"
        opp_scored = f"{opp}_score"
        team_col = f"{side}_team_id"

        df[f"{side}_for"] = (
            df.groupby(team_col)[scored_col]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )
        df[f"{side}_against"] = (
            df.groupby(team_col)[opp_scored]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
        )

    df["exp_home_pts"] = (df["home_for"] + df["away_against"]) / 2
    df["exp_away_pts"] = (df["away_for"] + df["home_against"]) / 2
    return df["exp_home_pts"] + df["exp_away_pts"]


def _compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    actual = actual.astype(float)
    pred = pred.astype(float)
    mask = actual.notna() & pred.notna()
    if mask.sum() == 0:
        return {"mae": float("nan"), "bias": float("nan"), "n": 0}
    mae = float(np.mean(np.abs(actual[mask] - pred[mask])))
    bias = float(np.mean(pred[mask] - actual[mask]))
    return {"mae": mae, "bias": bias, "n": int(mask.sum())}


def _resolve_root(project_root: Optional[Path]) -> Path:
    return Path(project_root) if project_root is not None else get_project_root()


def _find_odds_csv(root: Path) -> Path:
    """Locate the Kaggle odds CSV under PROJECT_ROOT (portable across machines)."""
    candidate = root / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
    if candidate.exists():
        return candidate

    matches = list(root.rglob("nba_2008-2025.csv"))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"NBA odds CSV not found under: {root}\n"
        "Expected at: data/raw/nba/odds_long/nba_2008-2025.csv\n"
        "Either place the file there or rerun the odds download notebook on this machine."
    )


def run_backtest(
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_odds: bool = False,
    project_root: Optional[Path] = None,
) -> dict:
    root = _resolve_root(project_root)

    games = load_eoin_games(project_root=root).copy()

    # Standardize dates
    games["game_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    if start:
        games = games.loc[games["game_date"] >= pd.to_datetime(start)]
    if end:
        games = games.loc[games["game_date"] <= pd.to_datetime(end)]

    games = games.sort_values("game_datetime").reset_index(drop=True)
    games["total_actual"] = games["home_score"].astype(float) + games["away_score"].astype(float)

    # Baseline + QPA layers
    games["total_pred"] = _basic_expected_totals(games)
    games["env_drift"] = compute_env_drift(games)  # must be leakage-safe internally
    games["total_pred_env"] = apply_env_drift(games["total_pred"], games["env_drift"])
    games["total_mix"] = apply_script_superposition(games)

    metrics: dict = {
        "rows": int(len(games)),
        "date_min": str(games["game_date"].min().date()) if len(games) else None,
        "date_max": str(games["game_date"].max().date()) if len(games) else None,
        "qepc_raw": _compute_metrics(games["total_actual"], games["total_pred"]),
        "qepc_env": _compute_metrics(games["total_actual"], games["total_pred_env"]),
        "qepc_script": _compute_metrics(games["total_actual"], games["total_mix"]),
    }

    if with_odds:
        odds_csv = _find_odds_csv(root)
        odds = load_long_odds(odds_csv)

        merged, diag = attach_odds_to_games(games, odds)

        # Posterior collapse only meaningful where Vegas totals exist
        merged["total_posterior"] = collapse_total_with_odds(
            merged, qepc_col="total_mix", vegas_col="total_points", actual_col="total_actual"
        )
        overlap = merged[merged["total_points"].notna()].copy()

        metrics["odds_overlap_rows"] = int(len(overlap))
        metrics["vegas"] = _compute_metrics(overlap["total_actual"], overlap["total_points"])
        metrics["posterior"] = _compute_metrics(overlap["total_actual"], overlap["total_posterior"])

        # Coverage diagnostics if present
        metrics["coverage"] = {
            "matched_rows": int(getattr(diag, "matched_rows", np.nan)),
            "total_games": int(getattr(diag, "total_games", np.nan)),
            "overlap_games": int(getattr(diag, "overlap_games", np.nan)),
            "overlap_matched": int(getattr(diag, "overlap_matched", np.nan)),
        }

        # Optional distribution stats (may be slow; still deterministic)
        dist = overdispersed_total_distribution(merged, pred_col="total_mix", actual_col="total_actual")
        metrics["entropy_mean_bits"] = float(dist.entropy_bits.mean())

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QEPC NBA totals backtest")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--with-odds", action="store_true", dest="with_odds")
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser() if args.project_root else None
    metrics = run_backtest(args.start, args.end, args.with_odds, project_root=root)

    # Pretty print in a stable order
    print("[backtest_totals] rows:", metrics.get("rows"))
    print("[backtest_totals] date range:", metrics.get("date_min"), "→", metrics.get("date_max"))
    for k in ("qepc_raw", "qepc_env", "qepc_script", "vegas", "posterior"):
        if k in metrics:
            print(f"[{k}] {metrics[k]}")
    if "coverage" in metrics:
        print("[coverage]", metrics["coverage"])
    if "entropy_mean_bits" in metrics:
        print("[entropy_mean_bits]", round(metrics["entropy_mean_bits"], 4))


if __name__ == "__main__":
    main()
