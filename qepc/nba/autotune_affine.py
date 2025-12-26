"""Autotune affine calibration for player points predictions (leakage-safe).

This script is meant to *calibrate* an already leakage-safe model output.
It fits an affine transform:

    points_cal = intercept + slope * points_pred

on a TRAIN window, then evaluates on a VALID window.

Why this is safe:
- We never use any data from the VALID window to choose slope/intercept.
- Your underlying features are already shift(1) (no leakage) in player_points_model.py.

Typical workflow:
1) Run backtest once and save predictions:
   python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2024-06-22 --min-minutes 10 --progress --save-preds

2) Fit affine on 2022-10-01..2023-06-22, validate on 2023-10-01..2024-06-22:
   python -m qepc.nba.autotune_affine --preds-file logs\\backtest_player_points_preds_*.parquet \
       --train-start 2022-10-01 --train-end 2023-06-22 \
       --valid-start 2023-10-01 --valid-end 2024-06-22 \
       --objective mae --save-json

"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Project root detection
# -----------------------------
def _find_project_root(start: Optional[Path] = None) -> Path:
    env = os.environ.get("QEPC_PROJECT_ROOT") or os.environ.get("QEPC_PROJECTPATH")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / "qepc").is_dir():
            return parent
    return p


def _resolve_path(project_root: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


# -----------------------------
# Metrics + fitting
# -----------------------------
def _mae(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(np.abs(y_pred - y_true)))


def _bias(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(np.mean(y_pred - y_true))


def _ols_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Return (slope, intercept) minimizing MSE."""
    x = x.astype(float)
    y = y.astype(float)
    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xv = float(np.var(x))
    if xv < 1e-12:
        return 1.0, 0.0
    cov = float(np.mean((x - xm) * (y - ym)))
    slope = cov / xv
    intercept = ym - slope * xm
    return float(slope), float(intercept)


def _lad_affine_grid(
    x: np.ndarray,
    y: np.ndarray,
    *,
    slope_center: float,
    slope_span: float = 0.35,
    slope_steps: int = 121,
    sample_max: int = 200_000,
    seed: int = 7,
) -> Tuple[float, float]:
    """Approximate least-absolute-deviation (MAE) affine via grid search on slope.

    For each candidate slope b, the MAE-optimal intercept a is median(y - b*x).
    """
    n = x.size
    if n == 0:
        return 1.0, 0.0

    rng = np.random.default_rng(seed)
    if n > sample_max:
        idx = rng.choice(n, size=sample_max, replace=False)
        xs = x[idx]
        ys = y[idx]
    else:
        xs = x
        ys = y

    slopes = np.linspace(slope_center * (1.0 - slope_span), slope_center * (1.0 + slope_span), slope_steps)

    best_mae = float("inf")
    best_slope = float(slope_center)
    best_intercept = 0.0

    # vectorized-ish loop (intercept depends on slope via median residual)
    for b in slopes:
        a = float(np.median(ys - b * xs))
        pred = a + b * xs
        m = _mae(pred, ys)
        if m < best_mae:
            best_mae = m
            best_slope = float(b)
            best_intercept = float(a)

    return best_slope, best_intercept


@dataclass
class FitResult:
    objective: str
    slope: float
    intercept: float
    train_mae: float
    train_bias: float
    valid_mae: float
    valid_bias: float
    valid_corr: float


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    try:
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return float("nan")


def _evaluate(
    x_pred: np.ndarray,
    y_true: np.ndarray,
    *,
    slope: float,
    intercept: float,
    clip_nonnegative: bool = True,
) -> Dict[str, float]:
    y_hat = intercept + slope * x_pred
    if clip_nonnegative:
        y_hat = np.clip(y_hat, 0.0, None)
    return {
        "mae": _mae(y_hat, y_true),
        "bias": _bias(y_hat, y_true),
        "corr": _corr(y_hat, y_true),
    }


# -----------------------------
# Main
# -----------------------------
def _parse_date(s: str) -> pd.Timestamp:
    return pd.to_datetime(s, errors="raise")


def _fit_and_eval(
    df: pd.DataFrame,
    *,
    date_col: str,
    pred_col: str,
    actual_col: str,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    valid_start: pd.Timestamp,
    valid_end: pd.Timestamp,
    objective: str,
    clip_nonnegative: bool,
    slope_span: float,
    slope_steps: int,
    sample_max: int,
) -> Dict[str, Any]:
    train = df[(df[date_col] >= train_start) & (df[date_col] <= train_end)]
    valid = df[(df[date_col] >= valid_start) & (df[date_col] <= valid_end)]
    if train.empty or valid.empty:
        return {"ok": False, "reason": "empty_split", "n_train": int(len(train)), "n_valid": int(len(valid))}

    x_tr = train[pred_col].to_numpy(dtype="float64")
    y_tr = train[actual_col].to_numpy(dtype="float64")
    x_va = valid[pred_col].to_numpy(dtype="float64")
    y_va = valid[actual_col].to_numpy(dtype="float64")

    ols_slope, ols_intercept = _ols_affine(x_tr, y_tr)

    if objective == "mse":
        slope, intercept = ols_slope, ols_intercept
    else:
        slope, intercept = _lad_affine_grid(
            x_tr,
            y_tr,
            slope_center=ols_slope,
            slope_span=float(slope_span),
            slope_steps=int(slope_steps),
            sample_max=int(sample_max),
            seed=7,
        )

    base_valid = {"mae": _mae(x_va, y_va), "bias": _bias(x_va, y_va), "corr": _corr(x_va, y_va)}
    train_eval = _evaluate(x_tr, y_tr, slope=slope, intercept=intercept, clip_nonnegative=clip_nonnegative)
    valid_eval = _evaluate(x_va, y_va, slope=slope, intercept=intercept, clip_nonnegative=clip_nonnegative)

    return {
        "ok": True,
        "train": {"start": str(train_start.date()), "end": str(train_end.date()), "n": int(len(train))},
        "valid": {"start": str(valid_start.date()), "end": str(valid_end.date()), "n": int(len(valid))},
        "baseline_valid": base_valid,
        "fit": {
            "objective": objective,
            "slope": float(slope),
            "intercept": float(intercept),
            "clip_nonneg": bool(clip_nonnegative),
            "train_mae": float(train_eval["mae"]),
            "train_bias": float(train_eval["bias"]),
            "valid_mae": float(valid_eval["mae"]),
            "valid_bias": float(valid_eval["bias"]),
            "valid_corr": float(valid_eval["corr"]),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Autotune affine calibration for player points predictions")
    p.add_argument("--preds-file", required=True, help="Predictions table (.parquet or .csv) saved from backtest_player_points --save-preds")
    p.add_argument("--pred-col", default="predicted_points_raw", help="Prediction column to calibrate (default: predicted_points_raw)")
    p.add_argument("--actual-col", default="actual_points", help="Actual points column (default: actual_points)")
    p.add_argument("--date-col", default="game_date", help="Date column (default: game_date)")

    # Mode A: single train->valid split (explicit dates)
    p.add_argument("--train-start", help="Train start date (YYYY-MM-DD)")
    p.add_argument("--train-end", help="Train end date (YYYY-MM-DD)")
    p.add_argument("--valid-start", help="Validation start date (YYYY-MM-DD)")
    p.add_argument("--valid-end", help="Validation end date (YYYY-MM-DD)")

    # Mode B: rolling walk-forward loop (autonomous-ish)
    p.add_argument("--rolling", action="store_true", help="Run a rolling walk-forward loop (re-fits affine each step)")
    p.add_argument("--rolling-start", help="First anchor date (YYYY-MM-DD) for rolling VALID end")
    p.add_argument("--rolling-end", help="Last anchor date (YYYY-MM-DD) for rolling VALID end")
    p.add_argument("--train-days", type=int, default=180, help="Train window size in days for rolling mode (default 180)")
    p.add_argument("--valid-days", type=int, default=30, help="Validation window size in days for rolling mode (default 30)")
    p.add_argument("--step-days", type=int, default=7, help="Step size (days) between rolling anchors (default 7)")
    p.add_argument("--out-csv", default=None, help="Where to write rolling results CSV (relative to project root if not absolute)")

    p.add_argument("--objective", choices=["mse", "mae"], default="mae", help="Objective for fitting affine (default: mae)")
    p.add_argument("--clip-nonnegative", action="store_true", help="Clip calibrated predictions at 0 (recommended)")
    p.add_argument("--no-clip-nonnegative", action="store_true", help="Do NOT clip calibrated predictions at 0")

    # MAE grid search knobs
    p.add_argument("--slope-span", type=float, default=0.35, help="MAE grid slope span around OLS slope (fraction). Default 0.35")
    p.add_argument("--slope-steps", type=int, default=121, help="MAE grid slope steps. Default 121")
    p.add_argument("--sample-max", type=int, default=200000, help="Max rows sampled for MAE grid fit. Default 200000")

    # Output
    p.add_argument("--save-json", action="store_true", help="Save a JSON summary under ./logs/")
    p.add_argument("--json-file", default=None, help="Explicit JSON output path (relative to project root if not absolute)")

    args = p.parse_args()

    project_root = _find_project_root()
    preds_path = _resolve_path(project_root, args.preds_file)

    if not preds_path.exists():
        raise SystemExit(f"[ERROR] preds-file not found: {preds_path}")

    # Load predictions
    if preds_path.suffix.lower() == ".csv":
        df = pd.read_csv(preds_path)
    else:
        df = pd.read_parquet(preds_path)

    if args.date_col not in df.columns:
        raise SystemExit(f"[ERROR] date column '{args.date_col}' not found in preds file.")
    df[args.date_col] = pd.to_datetime(df[args.date_col], errors="coerce")

    for col in (args.pred_col, args.actual_col):
        if col not in df.columns:
            raise SystemExit(f"[ERROR] column '{col}' not found in preds file.")
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[args.date_col, args.pred_col, args.actual_col]).copy()

    clip_nonneg = True
    if args.no_clip_nonnegative:
        clip_nonneg = False
    if args.clip_nonnegative:
        clip_nonneg = True

    if args.rolling:
        if not args.rolling_start or not args.rolling_end:
            raise SystemExit("[ERROR] --rolling requires --rolling-start and --rolling-end (YYYY-MM-DD).")

        anchor = _parse_date(args.rolling_start)
        anchor_end = _parse_date(args.rolling_end)
        step = timedelta(days=int(args.step_days))
        train_days = int(args.train_days)
        valid_days = int(args.valid_days)

        rows = []
        while anchor <= anchor_end:
            valid_end = anchor
            valid_start = valid_end - timedelta(days=valid_days - 1)
            train_end = valid_start - timedelta(days=1)
            train_start = train_end - timedelta(days=train_days - 1)

            res = _fit_and_eval(
                df,
                date_col=args.date_col,
                pred_col=args.pred_col,
                actual_col=args.actual_col,
                train_start=train_start,
                train_end=train_end,
                valid_start=valid_start,
                valid_end=valid_end,
                objective=args.objective,
                clip_nonnegative=clip_nonneg,
                slope_span=float(args.slope_span),
                slope_steps=int(args.slope_steps),
                sample_max=int(args.sample_max),
            )

            row = {
                "anchor_valid_end": str(valid_end.date()),
                "train_start": str(train_start.date()),
                "train_end": str(train_end.date()),
                "valid_start": str(valid_start.date()),
                "valid_end": str(valid_end.date()),
                "ok": bool(res.get("ok")),
            }
            if res.get("ok"):
                fit = res["fit"]
                base = res["baseline_valid"]
                row.update(
                    {
                        "n_train": res["train"]["n"],
                        "n_valid": res["valid"]["n"],
                        "slope": fit["slope"],
                        "intercept": fit["intercept"],
                        "base_valid_mae": base["mae"],
                        "cal_valid_mae": fit["valid_mae"],
                        "base_valid_bias": base["bias"],
                        "cal_valid_bias": fit["valid_bias"],
                        "cal_valid_corr": fit["valid_corr"],
                    }
                )
            else:
                row.update({"n_train": res.get("n_train", 0), "n_valid": res.get("n_valid", 0), "reason": res.get("reason")})
            rows.append(row)
            anchor = anchor + step

        out_df = pd.DataFrame(rows)
        print("=== Affine Autotune (rolling walk-forward) ===")
        print(f"Preds file: {preds_path}")
        print(f"Rows: {len(out_df)} | step_days={args.step_days} | train_days={train_days} | valid_days={valid_days}")
        print(out_df.tail(10).to_string(index=False))

        if args.out_csv:
            out_path = _resolve_path(project_root, args.out_csv)
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = project_root / "logs" / f"affine_autotune_rolling_{args.rolling_start}_{args.rolling_end}_{ts}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        print(f"[csv] Wrote: {out_path}")
        return

    # Single split mode
    if not (args.train_start and args.train_end and args.valid_start and args.valid_end):
        raise SystemExit("[ERROR] Provide --train-start/--train-end/--valid-start/--valid-end, or use --rolling.")

    train_start = _parse_date(args.train_start)
    train_end = _parse_date(args.train_end)
    valid_start = _parse_date(args.valid_start)
    valid_end = _parse_date(args.valid_end)

    res = _fit_and_eval(
        df,
        date_col=args.date_col,
        pred_col=args.pred_col,
        actual_col=args.actual_col,
        train_start=train_start,
        train_end=train_end,
        valid_start=valid_start,
        valid_end=valid_end,
        objective=args.objective,
        clip_nonnegative=clip_nonneg,
        slope_span=float(args.slope_span),
        slope_steps=int(args.slope_steps),
        sample_max=int(args.sample_max),
    )
    if not res.get("ok"):
        raise SystemExit(f"[ERROR] Split failed: {res}")

    fit = res["fit"]
    base_valid = res["baseline_valid"]

    print("=== Affine Autotune (train -> valid) ===")
    print(f"Preds file: {preds_path}")
    print(f"Pred col:   {args.pred_col}")
    print("")
    print(f"Train: {args.train_start} → {args.train_end} (n={res['train']['n']})")
    print(f"Valid: {args.valid_start} → {args.valid_end} (n={res['valid']['n']})")
    print("")
    print("Baseline on VALID (no affine):")
    print(f"  MAE={base_valid['mae']:.3f} | bias={base_valid['bias']:.3f} | corr={base_valid['corr']:.3f}")
    print("")
    print(f"Best affine ({args.objective.upper()}-fit on TRAIN):")
    print(f"  slope={fit['slope']:.6f} | intercept={fit['intercept']:.6f} | clip_nonneg={fit['clip_nonneg']}")
    print("")
    print("To apply these numbers in QEPC:")
    print(f"  python -m qepc.nba.backtest_player_points --start {args.valid_start} --end {args.valid_end} --min-minutes 10 --affine-slope {fit['slope']:.6f} --affine-intercept {fit['intercept']:.6f}")

    print("")
    print("TRAIN performance (after affine):")
    print(f"  MAE={fit['train_mae']:.3f} | bias={fit['train_bias']:.3f}")
    print("VALID performance (after affine):")
    print(f"  MAE={fit['valid_mae']:.3f} | bias={fit['valid_bias']:.3f} | corr={fit['valid_corr']:.3f}")

    if args.save_json or args.json_file:
        out_path = _resolve_path(project_root, args.json_file) if args.json_file else None
        if out_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = project_root / "logs" / f"affine_autotune_{args.train_start}_{args.valid_end}_{ts}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "preds_file": str(preds_path),
            "pred_col": args.pred_col,
            "actual_col": args.actual_col,
            "date_col": args.date_col,
            "train": res["train"],
            "valid": res["valid"],
            "baseline_valid": base_valid,
            "fit": fit,
            "notes": "Leakage-safe iff TRAIN window strictly precedes VALID window.",
        }
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[json] Wrote: {out_path}")


if __name__ == "__main__":
    main()
