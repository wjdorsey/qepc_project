"""
QEPC Quantum: Entropy and uncertainty measures.

These tools help you quantify how *sharp* or *fuzzy* a simulated
distribution is. Higher entropy ~ more uncertainty / spread.
Lower entropy ~ more concentrated / confident.

We work with discrete distributions (e.g. points, rebounds count).
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np


def discrete_entropy(
    probs: np.ndarray,
    base: float = 2.0,
) -> float:
    """
    Compute entropy of a discrete distribution.

    Parameters
    ----------
    probs : array-like
        Probabilities that sum to 1 (or very close).
    base : float
        Logarithm base. base=2 -> bits; base=e -> nats.

    Returns
    -------
    float entropy value.
    """
    p = np.asarray(probs, dtype=float)
    p = p[p > 0.0]  # ignore zero-probability bins
    if p.size == 0:
        return 0.0

    log_base = np.log(base) if base is not None else 1.0
    h = -np.sum(p * np.log(p)) / log_base
    return float(h)


def sample_entropy(
    samples: np.ndarray,
    base: float = 2.0,
    return_pmf: bool = False,
) -> Tuple[float, Optional[np.ndarray]]:
    """
    Estimate entropy from integer samples (e.g. simulated points).

    Parameters
    ----------
    samples : array-like of ints
        Discrete samples (e.g., simulated stat outcomes).
    base : float
        Logarithm base.
    return_pmf : bool
        If True, also return the normalized histogram (pmf).

    Returns
    -------
    (H, pmf) if return_pmf=True, else (H, None)
    """
    x = np.asarray(samples, dtype=int)
    if x.size == 0:
        return 0.0, None

    counts = np.bincount(x)
    total = counts.sum()
    if total == 0:
        return 0.0, None

    probs = counts / total
    h = discrete_entropy(probs, base=base)

    if return_pmf:
        return h, probs
    return h, None
