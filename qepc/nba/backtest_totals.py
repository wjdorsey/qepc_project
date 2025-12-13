"""Backtest QEPC quantum-inspired totals predictor."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from qepc.nba.qpa_totals import (
    TotalsConfig,
    evaluate_predictions,
    estimate_score_correlation,
    load_config,
    load_games_with_features,
    predict_totals,
)
from qepc.utils.paths import get_project_root


def _run_backtest(config: TotalsConfig, args: argparse.Namespace, label: str) -> None:
    project_root = get_project_root(Path(__file__).resolve())
    _, _, games_features = load_games_with_features(
        start=args.start,
        end=args.end,
        with_odds=args.with_odds,
        config=config,
        project_root=project_root,
    )
    corr_stats = estimate_score_correlation(games_features, config.corr_shrink)
    rng = np.random.default_rng(args.seed)
    preds = predict_totals(games_features, config, corr_stats, rng)
    metrics = evaluate_predictions(preds)
    print(f"[{label}] rows={len(preds)} MAE={metrics['mae']:.3f} bias={metrics['bias']:.3f}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="Filter games on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Filter games on/before this date (YYYY-MM-DD)")
    parser.add_argument("--with-odds", action="store_true", help="Include odds features when available")
    parser.add_argument("--use-tuned-config", action="store_true", help="Load cache/tuning/totals_qpa_best.json if present")
    parser.add_argument(
        "--tuned-path",
        type=str,
        default=None,
        help="Override tuned config path (default cache/tuning/totals_qpa_best.json)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for correlated sampling")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    project_root = get_project_root(Path(__file__).resolve())

    default_config = TotalsConfig()
    tuned_config = None
    if args.use_tuned_config:
        tuned_path = Path(args.tuned_path) if args.tuned_path else project_root / "cache" / "tuning" / "totals_qpa_best.json"
        if tuned_path.exists():
            tuned_config = load_config(tuned_path)
            print(f"Loaded tuned config from {tuned_path}")
        else:
            print(f"Tuned config not found at {tuned_path}; falling back to defaults")

    _run_backtest(default_config, args, label="qepc_default")
    if tuned_config is not None:
        _run_backtest(tuned_config, args, label="qepc_tuned")


if __name__ == "__main__":
    main()
