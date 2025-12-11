# qepc/nba/calibration_eoin.py

"""
Calibration constants for the Eoin-based NBA team total model.

These were fit on ~1,800 games from the Eoin dataset
(2024-10-04 onward, strengths built from 2022-10-01+ games)
using a linear mapping: actual ≈ m * pred + b.
"""

# Paste the numbers from your notebook output:
HOME_M = 1.916
HOME_B = -105.818

AWAY_M = 1.997
AWAY_B = -112.937


def calibrate_team_totals(raw_home: float, raw_away: float) -> tuple[float, float]:
    """
    Apply linear calibration to raw team total predictions.
    """
    cal_home = HOME_M * raw_home + HOME_B
    cal_away = AWAY_M * raw_away + AWAY_B
    return cal_home, cal_away
