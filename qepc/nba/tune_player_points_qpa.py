"""Grid tuner for the quantum-inspired player points model.

Writes the best-performing configuration to ``cache/tuning/player_points_best.json``.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from typing import Dict, List, Tuple

from qepc.utils.paths import get_project_root

from .player_points_model import PlayerPointsConfig, backtest_player_points


TAU_GRID = [12.0, 16.0, 20.0, 24.0]
WEIGHT_GRID: List[Tuple[float, float, float]] = [
    (0.5, 0.3, 0.2),
    (0.45, 0.35, 0.2),
    (0.4, 0.4, 0.2),
]
SHRINK_GRID = [0.3, 0.5, 0.7]
VAR_BOOST_GRID = [0.15, 0.25, 0.35]


def _config_from_tuple(minutes_tau: float, weights: Tuple[float, float, float], shrink: float, var_boost: float) -> Dict:
    return {
        "recent_window": 5,
        "decoherence": {
            "minutes_tau": minutes_tau,
            "usage_tau": minutes_tau - 2,
            "efficiency_tau": minutes_tau + 2,
            "variance_tau": 30.0,
        },
        "weights": {
            "season": weights[0],
            "recency": weights[1],
            "decoherence": weights[2],
        },
        "entanglement": {
            "enabled": True,
            "shrinkage": shrink,
            "variance_boost": var_boost,
            "min_games": 6,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune player points decoherence/weights")
    parser.add_argument("--start", required=True, help="Start date for tuning window (YYYY-MM-DD)")
    parser.add_argument("--end", required=True, help="End date for tuning window (YYYY-MM-DD)")
    parser.add_argument("--min-minutes", type=int, default=0, help="Minimum minutes filter for training rows")
    args = parser.parse_args()

    candidates = []
    for tau, wts, shrink, boost in product(TAU_GRID, WEIGHT_GRID, SHRINK_GRID, VAR_BOOST_GRID):
        cfg_dict = _config_from_tuple(tau, wts, shrink, boost)
        result = backtest_player_points(
            start_date=args.start,
            end_date=args.end,
            min_minutes=args.min_minutes,
            config=cfg_dict,
        )
        candidates.append({
            "config": cfg_dict,
            "mae": result["mae"],
            "bias": result["bias"],
        })
        print(
            f"τ={tau:4.1f} weights={wts} shrink={shrink:.2f} boost={boost:.2f} "
            f"→ MAE={result['mae']:.4f} bias={result['bias']:.4f}"
        )

    candidates = [c for c in candidates if c["mae"] == c["mae"]]
    if not candidates:
        raise RuntimeError("No valid candidates produced metrics")

    best = min(candidates, key=lambda x: x["mae"])
    best_cfg = PlayerPointsConfig.from_dict(best["config"]).to_dict()

    out_path = get_project_root() / "cache" / "tuning" / "player_points_best.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(best_cfg, f, indent=2)

    print("\n=== Best configuration ===")
    print(json.dumps(best_cfg, indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
