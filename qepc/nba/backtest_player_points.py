"""CLI backtest for player points projections (with progress + log + JSON).

Example
-------
python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2024-06-22 --min-minutes 10 --progress --save-log --save-json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .player_points_model import backtest_player_points


# -----------------------------
# Utilities
# -----------------------------
def _isfinite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except Exception:
        return False


def _parse_config(config_json: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_json:
        return None
    try:
        return json.loads(config_json)
    except Exception as e:
        raise SystemExit(f"[ERROR] --config must be valid JSON. Got: {e}") from e


def _find_project_root(start: Optional[Path] = None) -> Path:
    """
    Cross-computer safe project root auto-detect.
    - Respects env QEPC_PROJECT_ROOT if set.
    - Otherwise walks upward looking for pyproject.toml, .git, or qepc/ package folder.
    """
    env = os.environ.get("QEPC_PROJECT_ROOT")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    p = (start or Path.cwd()).resolve()
    for parent in [p] + list(p.parents):
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists() or (parent / "qepc").is_dir():
            return parent
    return p



def _fill_bucket_labels(rows, labels):
    """
    Ensure each row dict has a human-readable 'label'. If missing, fill from labels by index.
    """
    if not rows:
        return rows
    fixed = []
    for i, r in enumerate(rows):
        rr = dict(r) if isinstance(r, dict) else {"value": r}
        lab = rr.get("label")
        if (lab is None) or (str(lab).strip() == "") or (lab == "?"):
            if i < len(labels):
                rr["label"] = labels[i]
        fixed.append(rr)
    return fixed

def _fix_result_bucket_labels(result):
    """
    Mutate result dict so printed tables and JSON have bucket labels even if the model omitted them.
    """
    if not isinstance(result, dict):
        return result
    default_labels = ["0-20 mins", "20-28 mins", "28-34 mins", "34-inf mins"]
    for k in [
        "calibration_actual_minutes",
        "calibration_pred_minutes",
        "calibration_actual_minutes_cal",
        "calibration_pred_minutes_cal",
        "calibration_actual_minutes_minutes_affine",
        "calibration_pred_minutes_minutes_affine",
    ]:
        if k in result:
            result[k] = _fill_bucket_labels(result.get(k), default_labels)
    return result


class _TeeStdout:
    """Write stdout to both console and a file (progress bars usually stay on stderr)."""

    def __init__(self, file_path: Path):
        self.file_path = file_path
        self._f = None
        self._stdout = sys.stdout

    def __enter__(self):
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = self.file_path.open("w", encoding="utf-8")
        sys.stdout = self  # type: ignore[assignment]
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            sys.stdout = self._stdout  # type: ignore[assignment]
        finally:
            if self._f:
                self._f.flush()
                self._f.close()
        return False

    def write(self, s: str):
        # Console
        self._stdout.write(s)
        self._stdout.flush()
        # File
        if self._f:
            self._f.write(s)
            self._f.flush()

    def flush(self):
        self._stdout.flush()
        if self._f:
            self._f.flush()


def _json_safe(obj: Any) -> Any:
    """Convert numpy/pandas-ish scalars and other non-serializables into JSON-safe primitives."""
    # Basic primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # dict-like
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # list/tuple-like
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]

    # dataclasses
    try:
        from dataclasses import is_dataclass

        if is_dataclass(obj):
            return _json_safe(asdict(obj))
    except Exception:
        pass

    # Path
    if isinstance(obj, Path):
        return str(obj)

    # Try float conversion
    try:
        f = float(obj)
        if math.isfinite(f):
            return f
        return None
    except Exception:
        pass

    # Fallback to string
    return str(obj)


def _default_log_path(project_root: Path, start: str, end: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "logs" / f"backtest_player_points_{start}_{end}_{ts}.log"


def _default_json_path(project_root: Path, start: str, end: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return project_root / "logs" / f"backtest_player_points_{start}_{end}_{ts}.json"


def _resolve_path(project_root: Path, p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    path = Path(p)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


# -----------------------------
# CLI
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest player points projections")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-minutes", type=int, default=0, help="Filter out games where player minutes < this")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config for tau/weights")

    # Progress bars (kept on stderr by tqdm; remains single-line in terminal)
    parser.add_argument("--progress", action="store_true", help="Show progress bars during feature building")

    # Points affine calibration
    parser.add_argument("--affine-slope", type=float, default=None, help="Optional affine calibration slope (apply to predicted points)")
    parser.add_argument("--affine-intercept", type=float, default=None, help="Optional affine calibration intercept (apply to predicted points)")
    parser.add_argument("--apply-insample-affine", action="store_true", help="Fit affine calibration on this backtest range (diagnostic / not leakage-safe)")

    # Minutes affine calibration
    parser.add_argument("--minutes-affine-slope", type=float, default=None, help="Optional affine calibration slope (apply to predicted minutes)")
    parser.add_argument("--minutes-affine-intercept", type=float, default=None, help="Optional affine calibration intercept (apply to predicted minutes)")
    parser.add_argument("--apply-insample-minutes-affine", action="store_true", help="Fit minutes affine calibration on this range (diagnostic / not leakage-safe)")

    # Logging / JSON
    parser.add_argument("--save-log", action="store_true", help="Save printed output to a log file under ./logs/")
    parser.add_argument("--log-file", type=str, default=None, help="Explicit log file path (relative to project root if not absolute)")
    parser.add_argument("--save-json", action="store_true", help="Save a JSON run summary under ./logs/")
    parser.add_argument("--json-file", type=str, default=None, help="Explicit JSON file path (relative to project root if not absolute)")

    args = parser.parse_args()
    cfg_dict = _parse_config(args.config)

    project_root = _find_project_root()
    log_path = _resolve_path(project_root, args.log_file) if (args.save_log or args.log_file) else None
    json_path = _resolve_path(project_root, args.json_file) if (args.save_json or args.json_file) else None

    if (args.save_log or args.log_file) and log_path is None:
        log_path = _default_log_path(project_root, args.start, args.end)
    if (args.save_json or args.json_file) and json_path is None:
        json_path = _default_json_path(project_root, args.start, args.end)

    def _run_and_print() -> Dict[str, Any]:
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
            progress=bool(args.progress),
        )
        result = _fix_result_bucket_labels(result)

        # ---- Human-readable report ----
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

        # Diagnostics blocks
        if _isfinite(result.get("minutes_mae")):
            print("")
            print(f"Minutes diagnostic (coherent minutes vs actual): MAE={float(result['minutes_mae']):.3f} | bias={float(result['minutes_bias']):.3f}")

        if result.get("minutes_scale_diagnostic"):
            msd = result["minutes_scale_diagnostic"]
            if _isfinite(msd.get("slope")) and _isfinite(msd.get("intercept")):
                print("Minutes scale diagnostic (actual_min ≈ intercept + slope * pred_min):")
                print(
                    f"  slope={float(msd['slope']):.3f} | intercept={float(msd['intercept']):.3f} | corr={float(msd.get('corr', float('nan'))):.3f} | std_ratio(pred/actual)={float(msd.get('std_ratio', float('nan'))):.3f}"
                )

        if _isfinite(result.get("minutes_mae_cal")):
            print(f"Minutes diagnostic (CALIBRATED): MAE={float(result['minutes_mae_cal']):.3f} | bias={float(result['minutes_bias_cal']):.3f}")

        if _isfinite(result.get("ppm_mae")):
            print(f"PPM diagnostic (implied pred PPM vs actual PPM):    MAE={float(result['ppm_mae']):.3f} | bias={float(result['ppm_bias']):.3f}")

        if result.get("scale_diagnostic"):
            sd = result["scale_diagnostic"]
            if _isfinite(sd.get("slope")) and _isfinite(sd.get("intercept")):
                print("")
                print("Prediction scale diagnostic (actual ≈ intercept + slope * pred):")
                print(
                    f"  slope={float(sd['slope']):.3f} | intercept={float(sd['intercept']):.3f} | corr={float(sd.get('corr', float('nan'))):.3f} | std_ratio(pred/actual)={float(sd.get('std_ratio', float('nan'))):.3f}"
                )

        def _print_bucket_table(title: str, rows: Any):
            if not rows:
                return
            print("")
            print(title)
            for r in rows:
                label = r.get("label", "?")
                n = int(r.get("n", 0))
                mae = float(r.get("mae", float("nan")))
                bias = float(r.get("bias", float("nan")))
                pred = float(r.get("pred_mean", float("nan")))
                actual = float(r.get("actual_mean", float("nan")))
                print(f"{label:>12} | n={n:5d} | MAE={mae:5.3f} | bias={bias:6.3f} | pred={pred:5.2f} | actual={actual:5.2f}")

        _print_bucket_table("Calibration by ACTUAL minutes bucket:", result.get("calibration_actual_minutes"))
        _print_bucket_table("Calibration by PREDICTED minutes bucket (minutes_coherent):", result.get("calibration_pred_minutes"))
        _print_bucket_table("Calibration by ACTUAL minutes bucket (CALIBRATED):", result.get("calibration_actual_minutes_cal"))
        _print_bucket_table("Calibration by PREDICTED minutes bucket (CALIBRATED):", result.get("calibration_pred_minutes_cal"))
        _print_bucket_table("Calibration by ACTUAL minutes bucket (MINUTES-AFFINE points):", result.get("calibration_actual_minutes_minutes_affine"))
        _print_bucket_table("Calibration by PREDICTED minutes bucket (MINUTES-AFFINE points):", result.get("calibration_pred_minutes_minutes_affine"))

        return result

    # Run with optional stdout tee to a file
    if log_path is not None:
        print(f"[log] Saving console output to: {log_path}")
        with _TeeStdout(log_path):
            result = _run_and_print()
    else:
        result = _run_and_print()

    # JSON summary at the end
    if json_path is not None:
        payload = {
            "run_utc": datetime.now(timezone.utc).isoformat().replace("+00:00","Z"),
            "start": args.start,
            "end": args.end,
            "min_minutes": args.min_minutes,
            "progress": bool(args.progress),
            "config": cfg_dict,
            "result": _json_safe(result),
        }
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[json] Wrote run summary: {json_path}")


if __name__ == "__main__":
    main()
