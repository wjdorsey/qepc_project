# qepc/core/calibration.py

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class LinearCalibration:
    intercept: float
    slope: float

    def apply(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * x


def fit_linear_calibration(
    pred: pd.Series,
    actual: pd.Series,
) -> LinearCalibration:
    x = pred.values
    y = actual.values
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return LinearCalibration(intercept=float(intercept), slope=float(slope))
