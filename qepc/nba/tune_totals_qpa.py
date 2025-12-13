"""Tune quantum-inspired totals parameters via walk-forward search."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from qepc.nba.eoin_data_source import load_eoin_games, load_eoin_team_boxes
from qepc.nba.odds_long_loader import attach_odds_to_games, load_long_odds
from qepc.nba.qpa_totals import (
    TotalsConfig,
    enrich_games_with_config,
    evaluate_predictions,
    estimate_score_correlation,
    predict_totals,
    save_config,
)
from qepc.utils.paths import get_project_root


def _build_walkforward_indices(df: pd.DataFrame, folds: int) -> List[Tuple[int, int]]:
    n = len(df)
    fold_size = max(n // folds, 1)
    indices = []
    for i in range(folds):
        start = i * fold_size
        end = n if i == folds - 1 else min((i + 1) * fold_size, n)
        indices.append((start, end))
    return indices


def _score_config(
    games: pd.DataFrame,
    team_boxes: pd.DataFrame,
    config: TotalsConfig,
    folds: int,
    seed: int,
) -> Tuple[float, List[Dict[str, float]]]:
    team_state, enriched = enrich_games_with_config(games, team_boxes, config)
    enriched = enriched.sort_values(
        pd.to_datetime(enriched.get("game_date", enriched.get("game_datetime")))
    ).reset_index(drop=True)
    indices = _build_walkforward_indices(enriched, folds)

    fold_metrics: List[Dict[str, float]] = []
    scores: List[float] = []

    for fold_idx, (start, end) in enumerate(indices):
        val_df = enriched.iloc[start:end]
        train_df = enriched.iloc[:start]
        if train_df.empty:
            continue
        corr_stats = estimate_score_correlation(train_df, config.corr_shrink)
        rng = np.random.default_rng(seed + fold_idx)
        preds = predict_totals(val_df, config, corr_stats, rng)
        metrics = evaluate_predictions(preds)
        fold_metrics.append(metrics)
        score = metrics["mae"] + 0.25 * abs(metrics["bias"])
        scores.append(score)

    overall = float(np.mean(scores)) if scores else float("inf")
    return overall, fold_metrics


def _random_config(rng: np.random.Generator) -> TotalsConfig:
    return TotalsConfig(
        tau_offense=float(rng.uniform(6, 60)),
        tau_defense=float(rng.uniform(6, 60)),
        tau_pace=float(rng.uniform(6, 60)),
        offense_weight=float(rng.uniform(0.3, 0.8)),
        defense_weight=float(rng.uniform(0.2, 0.7)),
        pace_weight=float(rng.uniform(0.0, 0.4)),
        vegas_weight=float(rng.uniform(0.0, 0.7)),
        entropy_shrink=float(rng.uniform(0.05, 0.6)),
        corr_shrink=float(rng.uniform(0.05, 0.6)),
        sample_size=256,
    )


def _jitter_config(base: TotalsConfig, rng: np.random.Generator, temperature: float) -> TotalsConfig:
    def jitter(value: float, low: float, high: float) -> float:
        span = high - low
        proposal = value + rng.normal(scale=span * 0.1 * temperature)
        return float(np.clip(proposal, low, high))

    return TotalsConfig(
        tau_offense=jitter(base.tau_offense, 6, 60),
        tau_defense=jitter(base.tau_defense, 6, 60),
        tau_pace=jitter(base.tau_pace, 6, 60),
        offense_weight=jitter(base.offense_weight, 0.3, 0.8),
        defense_weight=jitter(base.defense_weight, 0.2, 0.7),
        pace_weight=jitter(base.pace_weight, 0.0, 0.4),
        vegas_weight=jitter(base.vegas_weight, 0.0, 0.7),
        entropy_shrink=jitter(base.entropy_shrink, 0.05, 0.6),
        corr_shrink=jitter(base.corr_shrink, 0.05, 0.6),
        sample_size=base.sample_size,
    )


def tune(
    games: pd.DataFrame,
    team_boxes: pd.DataFrame,
    folds: int,
    iterations: int,
    seed: int,
) -> Tuple[TotalsConfig, List[Dict[str, float]]]:
    rng = np.random.default_rng(seed)
    best_config = TotalsConfig()
    best_score, _ = _score_config(games, team_boxes, best_config, folds, seed)

    for i in range(iterations):
        temperature = max(0.1, 1.0 - i / max(iterations - 1, 1))
        if i % 3 == 0:
            candidate = _random_config(rng)
        else:
            candidate = _jitter_config(best_config, rng, temperature)
        score, _ = _score_config(games, team_boxes, candidate, folds, seed)
        if score < best_score:
            best_score = score
            best_config = candidate
        else:
            accept_prob = math.exp(-(score - best_score) / max(temperature, 1e-4))
            if rng.uniform() < accept_prob:
                best_score = score
                best_config = candidate
    final_score, fold_metrics = _score_config(games, team_boxes, best_config, folds, seed)
    return best_config, fold_metrics


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="Filter games on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Filter games on/before this date (YYYY-MM-DD)")
    parser.add_argument("--with-odds", action="store_true", help="Include odds if available")
    parser.add_argument("--folds", type=int, default=5, help="Number of walk-forward folds")
    parser.add_argument("--iterations", type=int, default=24, help="Number of tuning iterations")
    parser.add_argument("--seed", type=int, default=11, help="Random seed for reproducibility")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    project_root = get_project_root(Path(__file__).resolve())

    games = load_eoin_games(project_root)
    team_boxes = load_eoin_team_boxes(project_root)

    if args.start is not None:
        games = games.loc[
            pd.to_datetime(games.get("game_date", games.get("game_datetime")))
            >= pd.to_datetime(args.start)
        ]
        team_boxes = team_boxes.loc[
            pd.to_datetime(team_boxes.get("game_date", team_boxes.get("game_datetime")))
            >= pd.to_datetime(args.start)
        ]
    if args.end is not None:
        games = games.loc[
            pd.to_datetime(games.get("game_date", games.get("game_datetime")))
            <= pd.to_datetime(args.end)
        ]
        team_boxes = team_boxes.loc[
            pd.to_datetime(team_boxes.get("game_date", team_boxes.get("game_datetime")))
            <= pd.to_datetime(args.end)
        ]

    if args.with_odds:
        odds_path = project_root / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
        odds_df = load_long_odds(odds_path)
        games, _ = attach_odds_to_games(games, odds_df)

    best_config, fold_metrics = tune(
        games=games,
        team_boxes=team_boxes,
        folds=args.folds,
        iterations=args.iterations,
        seed=args.seed,
    )

    for idx, metrics in enumerate(fold_metrics):
        print(f"Fold {idx+1}: MAE={metrics['mae']:.3f} bias={metrics['bias']:.3f}")

    print("Best config:")
    print(json.dumps(best_config.to_dict(), indent=2))

    save_path = project_root / "cache" / "tuning" / "totals_qpa_best.json"
    save_config(best_config, save_path)
    print(f"Saved tuned config to {save_path}")


if __name__ == "__main__":
    main()
