"""Player points projection utilities with quantum-inspired twists.

This module focuses on **points** but is intentionally structured so rebounds
and assists can be layered in later.  It keeps a leakage-safe per-game schema
that downstream CLI tools (backtest/tuner) rely on.

Key ideas
---------
- Leakage-safe rolling features via ``shift(1)`` before rolling/expanding.
- Decoherence-weighted recency for minutes / usage / scoring efficiency using
  :func:`qepc.quantum.decoherence.exponential_time_weights`.
- Optional entanglement variance boost from correlation with team scoring
  context (team points / pace proxies) with shrinkage.
- All file paths are resolved from the auto-detected project root.

Schema (per-game rows)
----------------------
Returned DataFrames always contain at least the following columns:
- ``game_id``
- ``player_id``
- ``team_name``
- ``game_date`` (datetime64[ns])
- ``actual_points``
- ``predicted_points`` (mean)
- ``predicted_variance``

Additional helpful diagnostics are attached (season_mean_pts, recency_mean_pts,
minutes_coherent, efficiency_coherent, etc.) and are safe for extension to
rebounds/assists later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

from qepc.utils.paths import get_project_root

from .eoin_data_source import load_eoin_player_boxes, load_eoin_team_boxes


# ---------------------------------------------------------------------------
# Dataclasses for configuration (extensible to rebounds/assists later)
# ---------------------------------------------------------------------------


@dataclass
class DecoherenceConfig:
    """Time constants (in days) for recency decay of player features."""

    minutes_tau: float = 18.0
    usage_tau: float = 16.0
    efficiency_tau: float = 20.0
    variance_tau: float = 30.0


@dataclass
class BlendWeights:
    """Weights for blending season / recency / decoherence components."""

    season: float = 0.45
    recency: float = 0.35
    decoherence: float = 0.20

    def normalized(self) -> "BlendWeights":
        total = self.season + self.recency + self.decoherence
        if total <= 0:
            return BlendWeights(0.45, 0.35, 0.20)
        return BlendWeights(
            season=self.season / total,
            recency=self.recency / total,
            decoherence=self.decoherence / total,
        )


@dataclass
class EntanglementConfig:
    """Correlation-based variance adjustment with shrinkage."""

    enabled: bool = True
    shrinkage: float = 0.5  # 0=no shrink, 1=fully shrink to zero
    variance_boost: float = 0.25
    min_games: int = 6


@dataclass
class PlayerPointsConfig:
    """Bundle of tunable knobs for the player points model."""

    recent_window: int = 5
    min_history_games: int = 4
    decoherence: DecoherenceConfig = DecoherenceConfig()
    weights: BlendWeights = BlendWeights()
    entanglement: EntanglementConfig = EntanglementConfig()
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerPointsConfig":
        deco = data.get("decoherence", {})
        weights = data.get("weights", {})
        ent = data.get("entanglement", {})
        return cls(
            recent_window=data.get("recent_window", cls().recent_window),
            min_history_games=data.get("min_history_games", cls().min_history_games),
            decoherence=DecoherenceConfig(**deco) if not isinstance(deco, DecoherenceConfig) else deco,
            weights=BlendWeights(**weights).normalized()
            if not isinstance(weights, BlendWeights)
            else weights.normalized(),
            entanglement=EntanglementConfig(**ent)
            if not isinstance(ent, EntanglementConfig)
            else ent,
            seed=data.get("seed", cls().seed),
        )

    def to_dict(self) -> Dict:
        return {
            "recent_window": self.recent_window,
            "min_history_games": self.min_history_games,
            "decoherence": asdict(self.decoherence),
            "weights": asdict(self.weights.normalized()),
            "entanglement": asdict(self.entanglement),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if "game_date" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"])
        return df
    if "game_datetime" in df.columns:
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_datetime"])
        return df
    raise KeyError("player_boxes must contain game_date or game_datetime")


def _detect_minutes_column(df: pd.DataFrame) -> Optional[str]:
    for candidate in ("minutes", "minsplayed", "minutesplayed", "mins"):
        if candidate in df.columns:
            return candidate
    return None


def _attach_team_totals(df: pd.DataFrame) -> pd.DataFrame:
    required = ["game_id", "team_name", "points", "reboundstotal", "assists"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_boxes is missing required columns: {missing}")

    team_totals = (
        df.groupby(["game_id", "team_name"], as_index=False)[["points", "reboundstotal", "assists"]]
        .sum()
        .rename(
            columns={
                "points": "team_points",
                "reboundstotal": "team_rebounds",
                "assists": "team_assists",
            }
        )
    )
    merged = df.merge(team_totals, on=["game_id", "team_name"], how="left")

    for col in ("team_points", "team_rebounds", "team_assists"):
        merged[col] = merged[col].replace(0, np.nan)
    merged["points_share"] = merged["points"] / merged["team_points"]
    return merged


def _decoherence_prior_mean(group: pd.DataFrame, value_col: str, tau_days: float) -> pd.Series:
    values = []
    dates: list[pd.Timestamp] = []
    results = []
    for _, row in group.iterrows():
        if dates:
            delta_days = (pd.to_datetime(row["game_date"]) - pd.to_datetime(pd.Series(dates))).dt.days.astype(float)
            weights = np.exp(-delta_days / float(tau_days))
            weight_sum = weights.sum()
            if weight_sum > 0:
                results.append(float(np.sum(weights * np.array(values, dtype=float)) / weight_sum))
            else:
                results.append(np.nan)
        else:
            results.append(np.nan)
        dates.append(pd.to_datetime(row["game_date"]))
        values.append(row[value_col])
    return pd.Series(results, index=group.index)


def _historical_variance(group: pd.DataFrame, value_col: str, tau_days: float) -> pd.Series:
    values: list[float] = []
    dates: list[pd.Timestamp] = []
    output = []
    for _, row in group.iterrows():
        if values:
            delta_days = (pd.to_datetime(row["game_date"]) - pd.to_datetime(pd.Series(dates))).dt.days.astype(float)
            weights = np.exp(-delta_days / float(tau_days))
            w_sum = weights.sum()
            centered = np.array(values, dtype=float)
            mean_val = np.sum(weights * centered) / w_sum
            var_val = np.sum(weights * np.square(centered - mean_val)) / w_sum if w_sum > 0 else np.nan
            output.append(float(var_val))
        else:
            output.append(np.nan)
        dates.append(pd.to_datetime(row["game_date"]))
        values.append(row[value_col])
    return pd.Series(output, index=group.index)


def _shrink_correlation(corr: float, n: int, shrinkage: float) -> float:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    if n <= 1:
        return 0.0
    return float((1.0 - shrinkage) * corr * (1.0 - np.exp(-n / 10.0)))


def _attach_team_context(player_games: pd.DataFrame, team_boxes: Optional[pd.DataFrame]) -> pd.DataFrame:
    if team_boxes is None:
        return player_games

    tb = team_boxes.copy()
    tb = _ensure_datetime(tb)
    team_points = tb[["game_id", "team_id", "teamscore"]].rename(columns={"teamscore": "team_points_game"})
    player_games = player_games.merge(
        team_points,
        left_on=["game_id", "team_id"],
        right_on=["game_id", "team_id"],
        how="left",
    )
    return player_games


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def build_player_points_expectations(
    player_boxes: Optional[pd.DataFrame] = None,
    team_boxes: Optional[pd.DataFrame] = None,
    *,
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    min_minutes: int = 0,
    config: Optional[PlayerPointsConfig | Dict] = None,
) -> pd.DataFrame:
    """Compute leakage-free per-game player points expectations.

    Parameters
    ----------
    player_boxes : DataFrame, optional
        If None, loads QEPC Eoin player boxes from cache.
    team_boxes : DataFrame, optional
        If provided, used for entanglement correlation with team totals.
    start_date, end_date : date-like, optional
        Filter window for games (inclusive). Applied after features are built.
    min_minutes : int
        Drop rows where the player's recorded minutes < min_minutes (if a
        minutes column exists).
    config : PlayerPointsConfig or dict, optional
        Tuning parameters. When a dict is passed, :class:`PlayerPointsConfig`
        is constructed from it.
    """

    if config is None:
        cfg = PlayerPointsConfig()
    elif isinstance(config, PlayerPointsConfig):
        cfg = config
    else:
        cfg = PlayerPointsConfig.from_dict(config)
    cfg = PlayerPointsConfig.from_dict(cfg.to_dict())  # normalize weights

    np.random.seed(cfg.seed)

    if player_boxes is None:
        player_boxes = load_eoin_player_boxes(get_project_root())
    if team_boxes is None:
        try:
            team_boxes = load_eoin_team_boxes(get_project_root())
        except Exception:
            team_boxes = None

    df = _ensure_datetime(player_boxes)
    df = _attach_team_totals(df)

    minutes_col = _detect_minutes_column(df)
    if minutes_col is not None:
        df = df[df[minutes_col] >= min_minutes].copy()
        df["minutes_played"] = df[minutes_col]
    else:
        df["minutes_played"] = np.nan

    if start_date is not None:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # Baseline cumulative season averages (shifted to avoid leakage)
    g = df.groupby("player_id")
    df["cum_points_sum"] = g["points"].cumsum() - df["points"]
    df["cum_games"] = g.cumcount()
    df["season_mean_pts"] = df["cum_points_sum"] / df["cum_games"].replace(0, np.nan)

    df["recency_mean_pts"] = g["points"].shift(1).rolling(
        window=cfg.recent_window, min_periods=1
    ).mean()

    df["points_per_min"] = df["points"] / df["minutes_played"]

    # Decoherence-weighted features per player
    df["minutes_coherent"] = (
        g.apply(_decoherence_prior_mean, value_col="minutes_played", tau_days=cfg.decoherence.minutes_tau)
        .reset_index(level=0, drop=True)
    )
    df["usage_coherent"] = (
        g.apply(_decoherence_prior_mean, value_col="points_share", tau_days=cfg.decoherence.usage_tau)
        .reset_index(level=0, drop=True)
    )
    df["efficiency_coherent"] = (
        g.apply(_decoherence_prior_mean, value_col="points_per_min", tau_days=cfg.decoherence.efficiency_tau)
        .reset_index(level=0, drop=True)
    )
    df["variance_prior"] = (
        g.apply(_historical_variance, value_col="points", tau_days=cfg.decoherence.variance_tau)
        .reset_index(level=0, drop=True)
    )

    # Expectation built from components
    weights = cfg.weights.normalized()
    comp_from_parts = df["minutes_coherent"] * df["efficiency_coherent"]

    df["predicted_points"] = (
        weights.season * df["season_mean_pts"]
        + weights.recency * df["recency_mean_pts"]
        + weights.decoherence * comp_from_parts
    )

    # Variance estimate with entanglement optional
    df["team_id"] = df.get("team_id", df.get("teamid"))
    df = _attach_team_context(df, team_boxes)

    df["team_points_prior_var"] = (
        df.groupby("team_id").apply(_historical_variance, value_col="team_points_game", tau_days=cfg.decoherence.variance_tau)
        .reset_index(level=0, drop=True)
    )

    def compute_corr(sub: pd.DataFrame) -> pd.Series:
        corr_vals = []
        points_hist: list[float] = []
        team_hist: list[float] = []
        dates: list[pd.Timestamp] = []
        for _, row in sub.iterrows():
            if len(points_hist) >= cfg.entanglement.min_games:
                pts_arr = np.array(points_hist, dtype=float)
                team_arr = np.array(team_hist, dtype=float)
                mask = ~np.isnan(pts_arr) & ~np.isnan(team_arr)
                if mask.sum() >= cfg.entanglement.min_games:
                    corr = np.corrcoef(pts_arr[mask], team_arr[mask])[0, 1]
                    corr_vals.append(_shrink_correlation(corr, mask.sum(), cfg.entanglement.shrinkage))
                else:
                    corr_vals.append(0.0)
            else:
                corr_vals.append(0.0)
            points_hist.append(row["points"])
            team_hist.append(row.get("team_points_game", np.nan))
            dates.append(pd.to_datetime(row["game_date"]))
        return pd.Series(corr_vals, index=sub.index)

    if cfg.entanglement.enabled and team_boxes is not None:
        df["entanglement_corr"] = df.groupby("player_id").apply(compute_corr).reset_index(level=0, drop=True)
    else:
        df["entanglement_corr"] = 0.0

    df["predicted_variance"] = df["variance_prior"]
    if cfg.entanglement.enabled:
        df["predicted_variance"] = df["predicted_variance"] + (
            np.square(df["entanglement_corr"]) * cfg.entanglement.variance_boost * df["team_points_prior_var"]
        )

    # Trim to useful outputs
    out_cols = [
        "game_id",
        "player_id",
        "team_name",
        "game_date",
        "points",
        "predicted_points",
        "predicted_variance",
        "season_mean_pts",
        "recency_mean_pts",
        "minutes_coherent",
        "usage_coherent",
        "efficiency_coherent",
        "entanglement_corr",
    ]
    out = df[out_cols].rename(columns={"points": "actual_points"}).copy()
    out = out[out["cum_games"] >= cfg.min_history_games].reset_index(drop=True)

    return out


def backtest_player_points(
    *,
    start_date: str,
    end_date: str,
    min_minutes: int = 0,
    config: Optional[PlayerPointsConfig | Dict] = None,
    buckets: Optional[Iterable[int]] = None,
) -> Dict:
    """Run a backtest over the specified date range and return metrics."""

    preds = build_player_points_expectations(
        start_date=start_date,
        end_date=end_date,
        min_minutes=min_minutes,
        config=config,
    )

    preds = preds.dropna(subset=["predicted_points", "actual_points"])
    preds["error"] = preds["predicted_points"] - preds["actual_points"]

    mae = float(np.abs(preds["error"]).mean()) if not preds.empty else np.nan
    bias = float(preds["error"].mean()) if not preds.empty else np.nan

    if buckets is None:
        buckets = [0, 20, 30, 40]

    minutes_col = "minutes_coherent"
    labels = []
    bin_edges = list(buckets) + [np.inf]
    for i in range(len(bin_edges) - 1):
        labels.append(f"{bin_edges[i]}-{bin_edges[i+1]} mins")

    preds["minutes_bucket"] = pd.cut(
        preds[minutes_col].fillna(0),
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    calib_rows = []
    for bucket_label, group in preds.groupby("minutes_bucket"):
        if group.empty:
            continue
        calib_rows.append(
            {
                "bucket": str(bucket_label),
                "n": int(group.shape[0]),
                "mae": float(np.abs(group["error"]).mean()),
                "bias": float(group["error"].mean()),
                "pred_mean": float(group["predicted_points"].mean()),
                "actual_mean": float(group["actual_points"].mean()),
            }
        )

    return {
        "mae": mae,
        "bias": bias,
        "calibration": calib_rows,
        "predictions": preds,
    }
