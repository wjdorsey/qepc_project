"""
QEPC Module: props_adjust.py
Adjusts with real opponent DRtg.
"""

import numpy as np

LEAGUE_AVG_DRTG = 110.5  # Real 2025 avg

def adjust_prop(projection: float, opponent_drtg: float, projected_min: float, avg_min: float = 21.0) -> float:
    min_scale = projected_min / avg_min if avg_min > 0 else 1.0
    drtg_factor = LEAGUE_AVG_DRTG / opponent_drtg if opponent_drtg > 0 else 1.0
    noise = np.random.normal(1.0, 0.02)
    return projection * min_scale * drtg_factor * noise