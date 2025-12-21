"""CLI backtest for player points predictions.

Example
-------
python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2024-06-22 --min-minutes 10
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, Optional

from .player_points_model import backtest_player_points


def _parse_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _print_calib(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    for row in rows:
        print(
            f"  {row['bucket']:>12} | n={row['n']:>5} | MAE={row['mae']:.3f} | "
            f"bias={row['bias']:.3f} | pred={row['pred_mean']:.2f} | actual={row['actual_mean']:.2f}"
        )


def _isfinite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest player points projections")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-minutes", type=int, default=0, help="Filter out games where player minutes < this")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config for tau/weights")
    parser.add_argument("--affine-slope", type=float, default=None, help="Optional affine calibration slope (apply to predicted points)")
    parser.add_argument("--affine-intercept", type=float, default=None, help="Optional affine calibration intercept (apply to predicted points)")
    parser.add_argument("--apply-insample-affine", action="store_true", help="Fit affine calibration on this backtest range (diagnostic / not leakage-safe)")
    parser.add_argument("--minutes-affine-slope", type=float, default=None, help="Optional affine calibration slope (apply to predicted minutes)")
    parser.add_argument("--minutes-affine-intercept", type=float, default=None, help="Optional affine calibration intercept (apply to predicted minutes)")
    parser.add_argument("--apply-insample-minutes-affine", action="store_true", help="Fit minutes affine calibration on this backtest range (diagnostic / not leakage-safe)")

    args = parser.parse_args()
    cfg_dict = _parse_config(args.config)

    result = backtest_player_points(
        start_date=args.start,
        end_date=args.end,
        min_minutes=args.min_minutes,
        config=cfg_dict,
        affine_slope=args.affine_slope,
        affine_intercept=args.affine_intercept,
        apply_insample_affine=bool(args.apply_insample_affine),
        minutes_affine_slope=args.minutes_affine_slope,
        minutes_affine_intercept=args.minutes_affine_intercept,
        apply_insample_minutes_affine=bool(args.apply_insample_minutes_affine),
    )

    print("=== Player Points Backtest ===")
    print(f"Date range: {args.start} → {args.end}")
    print(f"Min minutes filter: {args.min_minutes}")
    print(f"MAE:  {result['mae']:.3f}")
    print(f"Bias: {result['bias']:.3f} (positive = over-predict)")

    if result.get("applied_minutes_affine") and _isfinite(result.get("mae_minutes_calibrated")):
        m_aff = result["applied_minutes_affine"]
        print("")
        print("Minutes affine applied (min_cal = intercept + slope * minutes_coherent) and points recomputed:")
        print(f"  slope={m_aff.get('slope', float('nan')):.3f} | intercept={m_aff.get('intercept', float('nan')):.3f}")
        print(f"Minutes-calibrated points MAE:  {float(result['mae_minutes_calibrated']):.3f}")
        print(f"Minutes-calibrated points Bias: {float(result['bias_minutes_calibrated']):.3f} (positive = over-predict)")

    if result.get("applied_affine") and _isfinite(result.get("mae_calibrated")):
        aff = result["applied_affine"]
        print("")
        print("Affine calibration applied (points_cal = intercept + slope * points_pred):")
        print(f"  slope={aff.get('slope', float('nan')):.3f} | intercept={aff.get('intercept', float('nan')):.3f}")
        print(f"Calibrated MAE:  {float(result['mae_calibrated']):.3f}")
        print(f"Calibrated Bias: {float(result['bias_calibrated']):.3f} (positive = over-predict)")

    diag = result.get("diagnostics", {})
    if diag:
        minutes = diag.get("minutes", {})
        ppm = diag.get("ppm", {})
        scale = diag.get("scale", {})

        minutes_scale = diag.get("minutes_scale", {})
        minutes_cal = diag.get("minutes_calibrated", {})

        if minutes:
            print(f"\nMinutes diagnostic (coherent minutes vs actual): MAE={minutes.get('mae', float('nan')):.3f} | bias={minutes.get('bias', float('nan')):.3f}")
        if minutes_scale:
            print(
                "Minutes scale diagnostic (actual_min ≈ intercept + slope * pred_min):\n"
                f"  slope={minutes_scale.get('slope', float('nan')):.3f} | intercept={minutes_scale.get('intercept', float('nan')):.3f} | "
                f"corr={minutes_scale.get('corr', float('nan')):.3f} | std_ratio(pred/actual)={minutes_scale.get('std_ratio', float('nan')):.3f}"
            )
        if minutes_cal:
            print(f"Minutes diagnostic (CALIBRATED): MAE={minutes_cal.get('mae', float('nan')):.3f} | bias={minutes_cal.get('bias', float('nan')):.3f}")
        if ppm:
            print(f"PPM diagnostic (implied pred PPM vs actual PPM):    MAE={ppm.get('mae', float('nan')):.3f} | bias={ppm.get('bias', float('nan')):.3f}")
        if scale:
            print(
                "\nPrediction scale diagnostic (actual ≈ intercept + slope * pred):\n"
                f"  slope={scale.get('slope', float('nan')):.3f} | intercept={scale.get('intercept', float('nan')):.3f} | "
                f"corr={scale.get('corr', float('nan')):.3f} | std_ratio(pred/actual)={scale.get('std_ratio', float('nan')):.3f}"
            )

    _print_calib("Calibration by ACTUAL minutes bucket:", result.get("calibration_actual_minutes", result.get("calibration", [])))
    _print_calib("Calibration by PREDICTED minutes bucket (minutes_coherent):", result.get("calibration_pred_minutes", []))

    if result.get("applied_minutes_affine") and result.get("calibration_pred_minutes_mincal"):
        _print_calib("Calibration by ACTUAL minutes bucket (MINUTES-AFFINE points):", result.get("calibration_actual_minutes_mincal", []))
        _print_calib("Calibration by PREDICTED minutes bucket (MINUTES-AFFINE points):", result.get("calibration_pred_minutes_mincal", []))

    if result.get("applied_affine") and result.get("calibration_actual_minutes_calibrated"):
        _print_calib("Calibration by ACTUAL minutes bucket (CALIBRATED):", result.get("calibration_actual_minutes_calibrated", []))
        _print_calib("Calibration by PREDICTED minutes bucket (CALIBRATED):", result.get("calibration_pred_minutes_calibrated", []))


if __name__ == "__main__":
    main()
