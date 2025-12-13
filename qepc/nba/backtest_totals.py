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
    df = games.sort_values("game_datetime").copy()

    for side, opp in (("home", "away"), ("away", "home")):
        scored_col = f"{side}_score"
        opp_scored = f"{opp}_score"
        team_col = f"{side}_team_id"

        df[f"{side}_for"] = (
            df.groupby(team_col)[scored_col]
            .transform(lambda s: s.shift().rolling(window, min_periods=3).mean())
        )
        df[f"{side}_against"] = (
            df.groupby(team_col)[opp_scored]
            .transform(lambda s: s.shift().rolling(window, min_periods=3).mean())
        )

    df["exp_home_pts"] = (df["home_for"] + df["away_against"]) / 2
    df["exp_away_pts"] = (df["away_for"] + df["home_against"]) / 2
    return df["exp_home_pts"] + df["exp_away_pts"]


def _compute_metrics(actual: pd.Series, pred: pd.Series) -> dict:
    mae = np.mean(np.abs(actual - pred))
    bias = float(np.mean(pred - actual))
    return {"mae": float(mae), "bias": bias}


def run_backtest(
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_odds: bool = False,
    project_root: Optional[Path] = None,
) -> dict:
    games = load_eoin_games(project_root=project_root).copy()
    games["game_date"] = pd.to_datetime(games["game_date"])

    if start:
        games = games.loc[games["game_date"] >= pd.to_datetime(start)]
    if end:
        games = games.loc[games["game_date"] <= pd.to_datetime(end)]

    games = games.sort_values("game_datetime")
    games["total_actual"] = games["home_score"] + games["away_score"]

    games["total_pred"] = _basic_expected_totals(games)
    games["env_drift"] = compute_env_drift(games)
    games["total_pred_env"] = apply_env_drift(games["total_pred"], games["env_drift"])
    games["total_mix"] = apply_script_superposition(games)

    metrics = {
        "qepc_raw": _compute_metrics(games["total_actual"], games["total_pred"]),
        "qepc_env": _compute_metrics(games["total_actual"], games["total_pred_env"]),
        "qepc_script": _compute_metrics(games["total_actual"], games["total_mix"]),
    }

    if with_odds:
        odds = load_long_odds(project_root=project_root)
        merged, diag = attach_odds_to_games(games, odds)
        merged["total_posterior"] = collapse_total_with_odds(
            merged, qepc_col="total_mix", vegas_col="total_points"
        )
        overlap = merged[merged["total_points"].notna()]
        metrics["vegas"] = _compute_metrics(overlap["total_actual"], overlap["total_points"])
        metrics["posterior"] = _compute_metrics(
            overlap["total_actual"], overlap["total_posterior"]
        )
        metrics["coverage"] = {
            "matched_rows": int(diag.matched_rows),
            "overlap_matched": int(diag.overlap_matched),
            "overlap_games": int(diag.overlap_games),
        }
        dist = overdispersed_total_distribution(merged)
        metrics["entropy_mean"] = float(dist.entropy_bits.mean())

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="QEPC NBA totals backtest")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--with-odds", action="store_true", dest="with_odds")
    parser.add_argument("--project-root", type=str, default=None)
    args = parser.parse_args()

    root = Path(args.project_root).expanduser() if args.project_root else get_project_root()
    metrics = run_backtest(args.start, args.end, args.with_odds, project_root=root)

    for name, m in metrics.items():
        print(f"[{name}] {m}")


if __name__ == "__main__":
    main()
