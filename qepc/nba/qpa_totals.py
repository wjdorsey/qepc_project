"""Quantum-inspired totals helpers for QEPC NBA."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from qepc.quantum.entropy import discrete_entropy

DEFAULT_SCRIPTS = ("GRIND", "BALANCED", "CHAOS")


def collapse_total_with_odds(
    df: pd.DataFrame,
    qepc_col: str = "total_pred_env",
    vegas_col: str = "total_points",
    actual_col: str = "total_actual",
    alpha: float = 0.02,
) -> pd.Series:
    """Blend QEPC and Vegas totals using rolling EMA of errors (shifted)."""

    working = df.copy()
    for col in (qepc_col, vegas_col, actual_col):
        if col not in working.columns:
            raise KeyError(f"{col} missing from DataFrame")

    working = working.sort_values("game_date")
    working["err_qepc"] = (working[actual_col] - working[qepc_col]).abs()
    working["err_vegas"] = (working[actual_col] - working[vegas_col]).abs()

    ema_q = working["err_qepc"].ewm(alpha=alpha, adjust=False).mean().shift(1)
    ema_v = working["err_vegas"].ewm(alpha=alpha, adjust=False).mean().shift(1)

    denom = ema_q + ema_v
    weight_vegas = ema_q / denom
    weight_vegas = weight_vegas.fillna(0.5).clip(0, 1)

    collapsed = weight_vegas * working[vegas_col] + (1 - weight_vegas) * working[qepc_col]
    collapsed.name = "total_posterior"
    return collapsed


def _heuristic_script_probs(
    totals: pd.Series,
    window: int = 20,
    scripts: Sequence[str] = DEFAULT_SCRIPTS,
) -> pd.DataFrame:
    rolling_std = totals.rolling(window=window, min_periods=5).std().shift(1)
    rolling_mean = totals.rolling(window=window, min_periods=5).mean().shift(1)

    # map std to chaos probability via logistic
    chaos_p = 1 / (1 + np.exp(-(rolling_std - 8) / 2))
    grind_p = 1 / (1 + np.exp((rolling_mean - totals.median()) / 5))
    chaos_p = chaos_p.fillna(0.33)
    grind_p = grind_p.fillna(0.33)
    balanced_p = 1 - (chaos_p + grind_p)
    balanced_p = balanced_p.clip(lower=0)

    probs = pd.DataFrame({"CHAOS": chaos_p, "GRIND": grind_p, "BALANCED": balanced_p})
    probs = probs[scripts]
    probs = probs.div(probs.sum(axis=1), axis=0).fillna(1 / len(scripts))
    probs.columns = [f"p_{c.lower()}" for c in probs.columns]
    return probs


def apply_script_superposition(
    df: pd.DataFrame,
    base_total_col: str = "total_pred_env",
    scripts: Sequence[str] = DEFAULT_SCRIPTS,
    deltas: Optional[Sequence[float]] = None,
    window: int = 20,
) -> pd.Series:
    if deltas is None:
        deltas = (-6.0, 0.0, 6.0)
    if len(deltas) != len(scripts):
        raise ValueError("deltas length must match scripts length")

    totals = df[base_total_col]
    probs = _heuristic_script_probs(df.get("total_actual", totals), window=window, scripts=scripts)
    adjustments = {s: d for s, d in zip(scripts, deltas)}

    components = []
    for script in scripts:
        p_col = f"p_{script.lower()}"
        adj = adjustments[script]
        comp = probs[p_col].fillna(0) * (totals + adj)
        components.append(comp)

    mixture = sum(components)
    mixture.name = "total_mix"
    return mixture


@dataclass
class DistributionStats:
    mean: pd.Series
    variance: pd.Series
    overdispersion: pd.Series
    lower_pi: pd.Series
    upper_pi: pd.Series
    entropy_bits: pd.Series


def overdispersed_total_distribution(
    df: pd.DataFrame,
    pred_col: str = "total_mix",
    actual_col: str = "total_actual",
    alpha: float = 0.05,
    window: int = 60,
) -> DistributionStats:
    working = df.sort_values("game_date").copy()
    if pred_col not in working.columns:
        raise KeyError(f"{pred_col} missing from DataFrame")
    if actual_col not in working.columns:
        working[actual_col] = np.nan

    working["resid"] = working[actual_col] - working[pred_col]
    rolling_var = (working["resid"] ** 2).rolling(window=window, min_periods=10).mean().shift(1)

    mu = working[pred_col]
    alpha_param = ((rolling_var - mu).clip(lower=0)) / (mu ** 2 + 1e-9)
    alpha_param = alpha_param.fillna(alpha_param.median()).clip(lower=0)

    variance = mu + alpha_param * (mu ** 2)
    std = np.sqrt(variance)

    z = 1.96  # ~95%
    lower = mu - z * std
    upper = mu + z * std

    entropy_vals = []
    for m, v in zip(mu, variance):
        sigma = np.sqrt(max(v, 1e-6))
        samples = np.random.normal(loc=m, scale=sigma, size=2000)
        samples = np.clip(np.round(samples), 0, None).astype(int)
        counts = np.histogram(samples, bins=range(int(samples.min()), int(samples.max()) + 2))[0]
        probs = counts / counts.sum()
        entropy_vals.append(discrete_entropy(probs, base=2))

    stats = DistributionStats(
        mean=mu,
        variance=variance,
        overdispersion=alpha_param,
        lower_pi=lower,
        upper_pi=upper,
        entropy_bits=pd.Series(entropy_vals, index=working.index),
    )
    return stats
