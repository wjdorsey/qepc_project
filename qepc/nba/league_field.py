"""League field (environment drift) adjustments for totals.

The "league field" represents slow-moving shifts in scoring
environment (e.g., officiating emphasis, pace meta).  We track
residuals between observed totals and model predictions in a
leakage-safe way and feed a blended rolling mean back into the
predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class DriftConfig:
    W_fast: int = 50
    W_slow: int = 250
    weights: Sequence[float] = (0.6, 0.4)
    residual_col: str = "residual"
    actual_col: str = "total_actual"
    pred_col: str = "total_pred"


def compute_env_drift(
    backtest_df: pd.DataFrame,
    W_fast: int = 50,
    W_slow: int = 250,
    weights: Sequence[float] | None = None,
    actual_col: str = "total_actual",
    pred_col: str = "total_pred",
    sort_keys: Optional[Iterable[str]] = ("game_date", "game_id"),
) -> pd.Series:
    """Compute leakage-safe blended residual drift.

    Rolling windows are shifted by one game so the adjustment available
    for game *t* depends only on games strictly before *t*.
    """

    df = backtest_df.copy()
    if sort_keys:
        df = df.sort_values(list(sort_keys))

    if weights is None:
        weights = (0.6, 0.4)
    if len(weights) != 2:
        raise ValueError("weights must have length 2 for (fast, slow) components")

    if actual_col not in df.columns or pred_col not in df.columns:
        raise KeyError("backtest_df must contain actual and predicted total columns")

    df["_resid"] = df[actual_col] - df[pred_col]

    fast = (
        df["_resid"].rolling(window=W_fast, min_periods=1).mean().shift(1)
    )
    slow = (
        df["_resid"].rolling(window=W_slow, min_periods=1).mean().shift(1)
    )

    drift = weights[0] * fast + weights[1] * slow
    drift.name = "env_drift"
    return drift


def apply_env_drift(total_pred: pd.Series, env_drift: pd.Series) -> pd.Series:
    """Add the environment drift to a totals prediction."""

    aligned_pred, aligned_drift = total_pred.align(env_drift, join="left")
    return aligned_pred + aligned_drift
