# qepc/nba/calibration_eoin.py

"""
Calibration constants for Eoin-based NBA team total model.

These were fit on ~1,800 games from the Eoin dataset
(2024-10-01 onward, modern-era strengths) using a linear
mapping: actual ≈ m * pred + b.
"""

HOME_M = 1.556
HOME_B = -56.346

AWAY_M = 1.664
AWAY_B = -70.100


def calibrate_team_totals(raw_home: float, raw_away: float) -> tuple[float, float]:
    """
    Apply linear calibration to raw team total predictions.
    """
    cal_home = HOME_M * raw_home + HOME_B
    cal_away = AWAY_M * raw_away + AWAY_B
    return cal_home, cal_away
