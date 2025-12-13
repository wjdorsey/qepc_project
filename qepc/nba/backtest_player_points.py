"""CLI backtest for player points predictions.

Example
-------
python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2024-06-22 --min-minutes 10
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, Optional

from .player_points_model import PlayerPointsConfig, backtest_player_points


def _parse_config(config_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not config_path:
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest player points projections")
    parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--min-minutes", type=int, default=0, help="Filter out games where player minutes < this")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config for tau/weights")

    args = parser.parse_args()

    cfg_dict = _parse_config(args.config)

    result = backtest_player_points(
        start_date=args.start,
        end_date=args.end,
        min_minutes=args.min_minutes,
        config=cfg_dict,
    )

    print("=== Player Points Backtest ===")
    print(f"Date range: {args.start} → {args.end}")
    print(f"Min minutes filter: {args.min_minutes}")
    print(f"MAE:  {result['mae']:.3f}")
    print(f"Bias: {result['bias']:.3f} (positive = over-predict)")

    print("\nCalibration by minutes bucket:")
    for row in result["calibration"]:
        print(
            f"  {row['bucket']:>12} | n={row['n']:>5} | MAE={row['mae']:.3f} | "
            f"bias={row['bias']:.3f} | pred={row['pred_mean']:.2f} | actual={row['actual_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
