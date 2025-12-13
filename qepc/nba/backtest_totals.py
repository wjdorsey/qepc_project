"""Backtest QEPC quantum-inspired totals predictor.

This runner is designed to be:
- simple to call from CLI
- reasonably leakage-aware (walk-forward by default)
- compatible with tuned configs saved to cache/tuning/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from qepc.nba.qpa_totals import (
    TotalsConfig,
    evaluate_predictions,
    estimate_score_correlation,
    load_config,
    load_games_with_features,
    predict_totals,
)
from qepc.utils.paths import get_project_root


def _to_game_date(df: pd.DataFrame) -> pd.Series:
    s = df.get("game_date")
    if s is None:
        s = df.get("game_datetime")
    if s is None:
        return pd.Series(pd.NaT, index=df.index)
    dt_series = pd.to_datetime(s, errors="coerce", utc=True).dt.tz_convert(None).dt.normalize()
    return dt_series


def _build_walkforward_indices(n: int, folds: int) -> List[Tuple[int, int]]:
    folds = max(int(folds), 1)
    fold_size = max(n // folds, 1)
    indices = []
    for i in range(folds):
        start = i * fold_size
        end = n if i == folds - 1 else min((i + 1) * fold_size, n)
        indices.append((start, end))
    return indices


def _walkforward_predict(enriched: pd.DataFrame, config: TotalsConfig, folds: int, seed: int) -> pd.DataFrame:
    """Predict using walk-forward correlation estimation (reduces leakage)."""
    df = enriched.copy()
    df["_sort_date"] = _to_game_date(df)
    df = df.sort_values(["_sort_date", "game_id"], kind="mergesort").reset_index(drop=False)
    df = df.rename(columns={"index": "__orig_idx"})

    indices = _build_walkforward_indices(len(df), folds)

    parts = []
    for fold_idx, (start, end) in enumerate(indices):
        val_df = df.iloc[start:end].copy()
        train_df = df.iloc[:start].copy()
        if train_df.empty:
            continue

        corr_stats = estimate_score_correlation(train_df, config.corr_shrink)
        rng = np.random.default_rng(seed + fold_idx)
        preds = predict_totals(val_df, config, corr_stats, rng)
        preds["__fold"] = fold_idx + 1
        parts.append(preds)

    if not parts:
        return df  # no preds

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values("__orig_idx").drop(columns=["__orig_idx", "_sort_date"], errors="ignore")
    return out


def _print_line(label: str, metrics: dict) -> None:
    mae = metrics.get("mae", float("nan"))
    bias = metrics.get("bias", float("nan"))
    n = metrics.get("n", 0)
    print(f"[{label}] rows={n} MAE={mae:.3f} bias={bias:.3f}")


def _run_backtest(config: TotalsConfig, args: argparse.Namespace, label: str) -> None:
    project_root = get_project_root(Path(__file__).resolve())

    _, _, games_features = load_games_with_features(
        start=args.start,
        end=args.end,
        with_odds=args.with_odds,
        config=config,
        project_root=project_root,
    )

    preds = _walkforward_predict(games_features, config=config, folds=args.folds, seed=args.seed)

    # QEPC metrics
    metrics = evaluate_predictions(preds)
    _print_line(label, metrics)

    # Optional overlap-only reporting when odds are present
    if args.with_odds and "total_points" in preds.columns:
        overlap = preds["total_points"].notna() & preds.get("total_score").notna()
        if overlap.any():
            overlap_metrics = evaluate_predictions(preds.loc[overlap].copy())
            _print_line(f"{label}_overlap", overlap_metrics)

            vegas_mae = float((preds.loc[overlap, "total_points"] - preds.loc[overlap, "total_score"]).abs().mean())
            vegas_bias = float((preds.loc[overlap, "total_points"] - preds.loc[overlap, "total_score"]).mean())
            print(f"[vegas] rows={int(overlap.sum())} MAE={vegas_mae:.3f} bias={vegas_bias:.3f}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=str, default=None, help="Filter games on/after this date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default=None, help="Filter games on/before this date (YYYY-MM-DD)")
    parser.add_argument("--with-odds", action="store_true", help="Include odds features when available")
    parser.add_argument("--folds", type=int, default=6, help="Walk-forward folds (default 6)")
    parser.add_argument("--use-tuned-config", action="store_true", help="Load cache/tuning/totals_qpa_best.json if present")
    parser.add_argument(
        "--tuned-path",
        type=str,
        default=None,
        help="Override tuned config path (default cache/tuning/totals_qpa_best.json)",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for correlated sampling")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
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
