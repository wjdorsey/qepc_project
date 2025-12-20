"""Player points projection utilities with quantum-inspired twists.

This module focuses on **points** but is intentionally structured so rebounds
and assists can be layered in later. It aims to be:

- **Leakage-safe**: all rolling features use shift(1) before rolling/expanding.
- **Portable**: all disk access resolves from QEPC auto-detected PROJECT_ROOT.
- **Robust**: hardened dtype joins + NaN-safe weighted statistics.

Key ideas
---------
- "Coherent minutes" are modeled as *expected minutes* from prior games
  (rolling mean of shifted minutes), not the current game's boxscore minutes.
  This prevents a classic leakage trap.
- Decoherence-weighted recency is used for usage and efficiency priors.
- Optional entanglement variance boost from correlation with team scoring context.

Schema (per-game rows)
----------------------
Returned DataFrames contain at least:
- ``game_id``
- ``player_id``
- ``team_name``
- ``game_date`` (datetime64[ns])
- ``actual_points``
- ``predicted_points`` (mean)
- ``predicted_variance``

Plus diagnostics:
- ``cum_games`` (number of PRIOR games for that player)
- ``minutes_actual`` (from boxscore; used only for filtering/diagnostics)
- ``minutes_expected`` / ``minutes_coherent`` (leakage-safe minutes feature)
- ``season_mean_pts`` / ``recency_mean_pts`` / ``efficiency_coherent`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, Optional, Union

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
        total = float(self.season + self.recency + self.decoherence)
        if total <= 0:
            return BlendWeights(0.45, 0.35, 0.20)
        return BlendWeights(
            season=float(self.season) / total,
            recency=float(self.recency) / total,
            decoherence=float(self.decoherence) / total,
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
    decoherence: DecoherenceConfig = field(default_factory=DecoherenceConfig)
    weights: BlendWeights = field(default_factory=BlendWeights)
    entanglement: EntanglementConfig = field(default_factory=EntanglementConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerPointsConfig":
        deco = data.get("decoherence", {})
        weights = data.get("weights", {})
        ent = data.get("entanglement", {})
        base = cls()
        return cls(
            recent_window=int(data.get("recent_window", base.recent_window)),
            min_history_games=int(data.get("min_history_games", base.min_history_games)),
            decoherence=DecoherenceConfig(**deco) if not isinstance(deco, DecoherenceConfig) else deco,
            weights=(BlendWeights(**weights) if not isinstance(weights, BlendWeights) else weights).normalized(),
            entanglement=EntanglementConfig(**ent) if not isinstance(ent, EntanglementConfig) else ent,
            seed=int(data.get("seed", base.seed)),
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
    """Ensure df has a datetime64[ns] column named game_date."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a DataFrame, got {type(df)}")

    out = df.copy()
    if "game_date" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_date"], errors="coerce")
        return out
    if "game_datetime" in out.columns:
        out["game_date"] = pd.to_datetime(out["game_datetime"], errors="coerce")
        return out
    raise KeyError("DataFrame must contain game_date or game_datetime.")


def _as_1d_series(obj, index: pd.Index, name: str = "value") -> pd.Series:
    """Coerce groupby/apply outputs into a 1D Series aligned to `index`."""
    if isinstance(obj, pd.DataFrame):
        s = obj.iloc[:, 0] if obj.shape[1] else pd.Series(index=index, dtype="float64", name=name)
    else:
        s = obj

    if not isinstance(s, pd.Series):
        s = pd.Series(s)

    if isinstance(s.index, pd.MultiIndex) and s.index.nlevels >= 2:
        try:
            s = s.droplevel(0)
        except Exception:
            pass

    s = s.reindex(index)
    s.name = name
    return s


def _detect_minutes_column(df: pd.DataFrame) -> Optional[str]:
    """Best-effort minutes column detection (case-insensitive)."""
    cols = {c.lower(): c for c in df.columns}

    preferred = [
        "minutes", "min", "mp", "mins", "minutes_played", "minutesplayed",
        "minsplayed", "minutes_played_total",
    ]
    for key in preferred:
        if key in cols:
            return cols[key]

    for lc, orig in cols.items():
        if "minute" in lc or (lc.endswith("min") and len(lc) <= 6):
            return orig

    return None


def _safe_div(numer: pd.Series, denom: pd.Series) -> pd.Series:
    denom = pd.to_numeric(denom, errors="coerce")
    numer = pd.to_numeric(numer, errors="coerce")
    out = numer / denom.replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _attach_team_totals(df: pd.DataFrame) -> pd.DataFrame:
    required = ["game_id", "team_name", "points", "reboundstotal", "assists"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"player_boxes is missing required columns: {missing}")

    team_totals = (
        df.groupby(["game_id", "team_name"], as_index=False)[["points", "reboundstotal", "assists"]]
        .sum()
        .rename(columns={"points": "team_points", "reboundstotal": "team_rebounds", "assists": "team_assists"})
    )

    merged = df.merge(team_totals, on=["game_id", "team_name"], how="left")
    for col in ("team_points", "team_rebounds", "team_assists"):
        merged[col] = merged[col].replace(0, np.nan)

    merged["points_share"] = _safe_div(merged["points"], merged["team_points"])
    return merged


def _decoherence_prior_mean(group: pd.DataFrame, value_col: str, tau_days: float) -> pd.Series:
    """NaN-safe exponential decay prior mean using only previous games."""
    values: list[float] = []
    dates: list[pd.Timestamp] = []
    results: list[float] = []

    tau = float(tau_days) if tau_days and tau_days > 0 else 1.0

    for _, row in group.iterrows():
        if dates:
            dt0 = pd.to_datetime(row["game_date"])
            past_dates = pd.to_datetime(pd.Series(dates))
            delta_days = (dt0 - past_dates).dt.days.astype(float).to_numpy()

            v = np.array(values, dtype="float64")
            w = np.exp(-delta_days / tau)

            mask = np.isfinite(v) & np.isfinite(w)
            if mask.any():
                w_sum = float(w[mask].sum())
                results.append(float((w[mask] * v[mask]).sum() / w_sum) if w_sum > 0 else np.nan)
            else:
                results.append(np.nan)
        else:
            results.append(np.nan)

        dates.append(pd.to_datetime(row["game_date"]))
        values.append(pd.to_numeric(row.get(value_col), errors="coerce"))

    return pd.Series(results, index=group.index)


def _historical_variance(group: pd.DataFrame, value_col: str, tau_days: float) -> pd.Series:
    """NaN-safe exponential decay prior variance using only previous games."""
    values: list[float] = []
    dates: list[pd.Timestamp] = []
    out: list[float] = []

    tau = float(tau_days) if tau_days and tau_days > 0 else 1.0

    for _, row in group.iterrows():
        if values:
            dt0 = pd.to_datetime(row["game_date"])
            past_dates = pd.to_datetime(pd.Series(dates))
            delta_days = (dt0 - past_dates).dt.days.astype(float).to_numpy()

            v = np.array(values, dtype="float64")
            w = np.exp(-delta_days / tau)

            mask = np.isfinite(v) & np.isfinite(w)
            if mask.any():
                w_sum = float(w[mask].sum())
                mu = float((w[mask] * v[mask]).sum() / w_sum) if w_sum > 0 else np.nan
                var = float((w[mask] * (v[mask] - mu) ** 2).sum() / w_sum) if w_sum > 0 else np.nan
                out.append(var)
            else:
                out.append(np.nan)
        else:
            out.append(np.nan)

        dates.append(pd.to_datetime(row["game_date"]))
        values.append(pd.to_numeric(row.get(value_col), errors="coerce"))

    return pd.Series(out, index=group.index)


def _shrink_correlation(corr: float, n: int, shrinkage: float) -> float:
    shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
    if n <= 1 or not np.isfinite(corr):
        return 0.0
    return float((1.0 - shrinkage) * corr * (1.0 - np.exp(-n / 10.0)))


def _attach_team_context(player_games: pd.DataFrame, team_boxes: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Attach lightweight team context (team points in the game), dtype-hardened."""
    if team_boxes is None:
        return player_games

    pg = player_games.copy()
    tb = _ensure_datetime(team_boxes)

    for df in (pg, tb):
        if "team_id" in df.columns:
            df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
        if "game_id" in df.columns:
            df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")

    if "teamscore" not in tb.columns:
        for alt in ("team_points", "points", "score"):
            if alt in tb.columns:
                tb = tb.rename(columns={alt: "teamscore"})
                break

    if "teamscore" not in tb.columns:
        return pg

    team_points = tb[["game_id", "team_id", "teamscore"]].rename(columns={"teamscore": "team_points_game"})
    return pg.merge(team_points, on=["game_id", "team_id"], how="left")


def _prior_rolling_mean(s: pd.Series, window: int) -> pd.Series:
    """Leakage-safe rolling mean: shift(1) before rolling."""
    return s.shift(1).rolling(window=window, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Core public API
# ---------------------------------------------------------------------------


def build_player_points_expectations(
    player_boxes: Optional[pd.DataFrame] = None,
    team_boxes: Optional[pd.DataFrame] = None,
    *,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
    end_date: Optional[Union[str, pd.Timestamp]] = None,
    min_minutes: int = 0,
    config: Optional[Union[PlayerPointsConfig, Dict]] = None,
) -> pd.DataFrame:
    """Compute leakage-free per-game player points expectations."""
    if config is None:
        cfg = PlayerPointsConfig()
    elif isinstance(config, PlayerPointsConfig):
        cfg = config
    else:
        cfg = PlayerPointsConfig.from_dict(config)

    cfg = PlayerPointsConfig.from_dict(cfg.to_dict())  # normalize weights
    np.random.seed(cfg.seed)

    root = get_project_root()

    if player_boxes is None:
        player_boxes = load_eoin_player_boxes(root)
    if team_boxes is None:
        try:
            team_boxes = load_eoin_team_boxes(root)
        except Exception:
            team_boxes = None

    df = _ensure_datetime(player_boxes)
    df = _attach_team_totals(df)

    minutes_col = _detect_minutes_column(df)
    if minutes_col is not None:
        df["minutes_actual"] = pd.to_numeric(df[minutes_col], errors="coerce")
        df = df[df["minutes_actual"] >= float(min_minutes)].copy()
    else:
        df["minutes_actual"] = np.nan

    if start_date is not None:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    g = df.groupby("player_id", sort=False)

    df["cum_games"] = g.cumcount()
    df["cum_points_sum"] = g["points"].cumsum() - pd.to_numeric(df["points"], errors="coerce")
    df["season_mean_pts"] = df["cum_points_sum"] / df["cum_games"].replace(0, np.nan)

    df["recency_mean_pts"] = _prior_rolling_mean(g["points"], window=int(cfg.recent_window))

    df["points_per_min"] = _safe_div(df["points"], df["minutes_actual"])

    df["minutes_expected"] = _prior_rolling_mean(g["minutes_actual"], window=10)

    global_min_med = float(pd.to_numeric(df["minutes_actual"], errors="coerce").median()) if df["minutes_actual"].notna().any() else 24.0
    df["minutes_coherent"] = df["minutes_expected"].fillna(global_min_med).clip(lower=0, upper=48)

    df["usage_coherent"] = _as_1d_series(
        g.apply(_decoherence_prior_mean, value_col="points_share", tau_days=cfg.decoherence.usage_tau),
        df.index,
        name="usage_coherent",
    )
    df["efficiency_coherent"] = _as_1d_series(
        g.apply(_decoherence_prior_mean, value_col="points_per_min", tau_days=cfg.decoherence.efficiency_tau),
        df.index,
        name="efficiency_coherent",
    )
    df["variance_prior"] = _as_1d_series(
        g.apply(_historical_variance, value_col="points", tau_days=cfg.decoherence.variance_tau),
        df.index,
        name="variance_prior",
    )

    weights = cfg.weights.normalized()

    comp_from_parts = df["minutes_coherent"] * df["efficiency_coherent"]
    df["predicted_points"] = (
        weights.season * df["season_mean_pts"]
        + weights.recency * df["recency_mean_pts"]
        + weights.decoherence * comp_from_parts
    )
    df["predicted_points"] = df["predicted_points"].fillna(df["recency_mean_pts"]).fillna(df["season_mean_pts"])

    if "team_id" not in df.columns:
        if "teamid" in df.columns:
            df["team_id"] = df["teamid"]
        else:
            df["team_id"] = pd.NA

    df["team_id"] = pd.to_numeric(df["team_id"], errors="coerce").astype("Int64")
    df = _attach_team_context(df, team_boxes)

    if "team_points_game" in df.columns:
        team_var_raw = df.groupby("team_id").apply(
            _historical_variance,
            value_col="team_points_game",
            tau_days=cfg.decoherence.variance_tau,
        )
        df["team_points_prior_var"] = _as_1d_series(team_var_raw, df.index, name="team_points_prior_var")
    else:
        df["team_points_prior_var"] = np.nan

    def _compute_corr(sub: pd.DataFrame) -> pd.Series:
        corr_vals: list[float] = []
        pts_hist: list[float] = []
        team_hist: list[float] = []
        for _, row in sub.iterrows():
            if len(pts_hist) >= int(cfg.entanglement.min_games):
                pts = np.array(pts_hist, dtype="float64")
                tm = np.array(team_hist, dtype="float64")
                mask = np.isfinite(pts) & np.isfinite(tm)
                if int(mask.sum()) >= int(cfg.entanglement.min_games):
                    corr = float(np.corrcoef(pts[mask], tm[mask])[0, 1])
                    corr_vals.append(_shrink_correlation(corr, int(mask.sum()), float(cfg.entanglement.shrinkage)))
                else:
                    corr_vals.append(0.0)
            else:
                corr_vals.append(0.0)
            pts_hist.append(pd.to_numeric(row.get("points"), errors="coerce"))
            team_hist.append(pd.to_numeric(row.get("team_points_game"), errors="coerce"))
        return pd.Series(corr_vals, index=sub.index)

    if cfg.entanglement.enabled and team_boxes is not None and "team_points_game" in df.columns:
        corr_raw = df.groupby("player_id").apply(_compute_corr)
        df["entanglement_corr"] = _as_1d_series(corr_raw, df.index, name="entanglement_corr")
    else:
        df["entanglement_corr"] = 0.0

    global_var = float(pd.to_numeric(df["points"], errors="coerce").var()) if df["points"].notna().any() else 25.0
    base_var = pd.to_numeric(df["variance_prior"], errors="coerce").fillna(global_var).clip(lower=1e-6)
    df["predicted_variance"] = base_var

    if cfg.entanglement.enabled:
        df["predicted_variance"] = df["predicted_variance"] + (
            np.square(pd.to_numeric(df["entanglement_corr"], errors="coerce").fillna(0.0))
            * float(cfg.entanglement.variance_boost)
            * pd.to_numeric(df["team_points_prior_var"], errors="coerce").fillna(0.0)
        )

    out_cols = [
        "game_id",
        "player_id",
        "team_id",
        "team_name",
        "game_date",
        "points",
        "predicted_points",
        "predicted_variance",
        "cum_games",
        "season_mean_pts",
        "recency_mean_pts",
        "minutes_actual",
        "minutes_expected",
        "minutes_coherent",
        "usage_coherent",
        "efficiency_coherent",
        "entanglement_corr",
    ]
    out = df[out_cols].rename(columns={"points": "actual_points"}).copy()

    if cfg.min_history_games and int(cfg.min_history_games) > 0:
        out = out[out["cum_games"] >= int(cfg.min_history_games)].reset_index(drop=True)

    return out


def backtest_player_points(
    *,
    start_date: str,
    end_date: str,
    min_minutes: int = 0,
    config: Optional[Union[PlayerPointsConfig, Dict]] = None,
    buckets: Optional[Iterable[int]] = None,
) -> Dict:
    """Run a backtest over the specified date range and return metrics."""
    preds = build_player_points_expectations(
        start_date=start_date,
        end_date=end_date,
        min_minutes=min_minutes,
        config=config,
    ).copy()

    preds = preds.dropna(subset=["predicted_points", "actual_points"])
    preds["error"] = preds["predicted_points"] - preds["actual_points"]

    mae = float(np.abs(preds["error"]).mean()) if not preds.empty else np.nan
    bias = float(preds["error"].mean()) if not preds.empty else np.nan

    mins = pd.to_numeric(preds.get("minutes_actual", preds.get("minutes_coherent")), errors="coerce").fillna(0.0)

    if buckets is None:
        buckets = [0, 20, 28, 34]

    bin_edges = list(buckets) + [np.inf]
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]} mins" for i in range(len(bin_edges) - 1)]

    preds["minutes_bucket"] = pd.cut(
        mins,
        bins=bin_edges,
        labels=labels,
        include_lowest=True,
        right=False,
    )

    calib_rows = []
    for bucket_label, group in preds.groupby("minutes_bucket", observed=False):
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
