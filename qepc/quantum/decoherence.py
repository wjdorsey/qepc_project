"""
QEPC Quantum: Decoherence / recency weighting utilities.

Core idea:
    Older games "lose coherence" and should matter less than recent ones.

We implement this as exponential time decay:
    weight ~ exp(-Δdays / tau_days)

Where:
    - tau_days is a "coherence time" for a given stat.
    - Different stats can have different tau (points vs rebounds, etc.).
"""

from __future__ import annotations

from typing import Sequence, Union, Optional
from datetime import date, datetime

import numpy as np
import pandas as pd


DateLike = Union[str, date, datetime, pd.Timestamp]


def _to_datetime_series(dates: Union[pd.Series, Sequence[DateLike]]) -> pd.Series:
    """
    Normalize a sequence/Series of dates into a pandas datetime64[ns] Series.
    """
    if isinstance(dates, pd.Series):
        return pd.to_datetime(dates)
    return pd.to_datetime(pd.Series(dates))


def exponential_time_weights(
    dates: Union[pd.Series, Sequence[DateLike]],
    ref_date: Optional[DateLike] = None,
    tau_days: float = 30.0,
    clip_days: Optional[float] = None,
    normalize: bool = True,
) -> pd.Series:
    """
    Compute exponential recency weights for a vector of dates.

    Parameters
    ----------
    dates : sequence or Series of dates
        Game dates (or datetimes).
    ref_date : date-like, optional
        Reference date. If None, uses max(dates).
    tau_days : float
        Time constant in days. Larger tau = slower decay.
    clip_days : float, optional
        If provided, limit Δdays to at most this value (older games get
        essentially the same tiny weight).
    normalize : bool
        If True, rescale weights so they sum to 1.

    Returns
    -------
    pandas.Series of float weights aligned with `dates`.
    """
    dt = _to_datetime_series(dates)

    if ref_date is None:
        ref_dt = dt.max()
    else:
        ref_dt = pd.to_datetime(ref_date)

    # Δdays = (ref - game). If game is older, Δdays > 0.
    delta_days = (ref_dt - dt).dt.days.astype(float)

    if clip_days is not None:
        delta_days = np.minimum(delta_days, float(clip_days))

    # Exponential decay: w ~ exp(-Δdays / tau)
    tau = float(tau_days)
    if tau <= 0:
        raise ValueError(f"tau_days must be > 0, got {tau_days}")

    weights = np.exp(-delta_days / tau)

    if normalize:
        total = weights.sum()
        if total > 0:
            weights = weights / total

    return pd.Series(weights, index=dt.index)


def recency_weighted_groupby_mean(
    df: pd.DataFrame,
    date_col: str,
    group_cols: Union[str, Sequence[str]],
    value_cols: Sequence[str],
    tau_days: float,
    ref_date: Optional[DateLike] = None,
    clip_days: Optional[float] = None,
    weight_col_name: str = "recency_weight",
) -> pd.DataFrame:
    """
    Compute recency-weighted means of one or more value columns,
    grouped by group_cols, using exponential decay weights.

    This is a generic helper – you can use it for player-level stats,
    team-level stats, etc.

    Parameters
    ----------
    df : DataFrame
        Source data with at least date_col, group_cols, value_cols.
    date_col : str
        Name of the date/datetime column.
    group_cols : str or sequence of str
        Columns to group by (e.g. ["player_id", "team_name"]).
    value_cols : sequence of str
        Numeric columns to compute recency-weighted means for.
    tau_days : float
        Time constant in days for the exponential decay.
    ref_date : date-like, optional
        Reference date. If None, uses max(df[date_col]).
    clip_days : float, optional
        If provided, clip Δdays to at most this many days.
    weight_col_name : str
        Name for the temporary weight column.

    Returns
    -------
    DataFrame indexed by group_cols with columns:
        - recency-weighted means for each value_col
        - total_weight (sum of raw weights per group)
    """
    if isinstance(group_cols, str):
        group_cols = [group_cols]

    work = df.copy()

    # Compute row-level weights (not normalized within groups yet)
    w = exponential_time_weights(
        work[date_col],
        ref_date=ref_date,
        tau_days=tau_days,
        clip_days=clip_days,
        normalize=False,  # we'll normalize at group level
    )
    work[weight_col_name] = w

    # For each group, recency-weighted mean:
    #   mean = sum(w * x) / sum(w)
    agg_dict = {}
    for col in value_cols:
        agg_dict[col] = lambda x, col=col: np.nan  # placeholder, we override

    # We'll implement manually to keep it explicit
    grouped = work.groupby(group_cols)

    rows = []
    for keys, group in grouped:
        gw = group[weight_col_name].values
        total_w = gw.sum()

        result = {}
        if isinstance(keys, tuple):
            for kname, kval in zip(group_cols, keys):
                result[kname] = kval
        else:
            result[group_cols[0]] = keys

        result["total_weight"] = float(total_w)

        if total_w > 0:
            for col in value_cols:
                gx = group[col].astype(float).values
                result[col] = float((gw * gx).sum() / total_w)
        else:
            for col in value_cols:
                result[col] = np.nan

        rows.append(result)

    out = pd.DataFrame(rows)
    return out
