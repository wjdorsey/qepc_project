"""Player points projection utilities with quantum-inspired twists.

This module projects NBA player **points** from Eoin boxscores in a way that is:

- **Leakage-safe**: all rolling/recency features use only prior games (shift(1)).
- **Portable**: all data paths resolve via QEPC PROJECT_ROOT auto-detect.
- **Robust**: dtype-hardened joins + NaN-safe weighted statistics.

Key idea (minutes coherence)
----------------------------
We treat minutes as a *latent/expected* quantity ("coherent minutes") derived
from **prior** games, not the current game's boxscore. This prevents leakage.

Improvements in this version
----------------------------
- Minutes artifact clipping (Eoin has min=-5 and max=96 in `numminutes`)
- Nonlinear minutes scaling so bench players aren't over-credited and high-
  minutes players aren't artificially capped.

Returned DataFrame contains at least:
- game_id, player_id, team_id, team_name, game_date
- actual_points, predicted_points, predicted_variance
- cum_games, minutes_actual, minutes_expected, minutes_coherent
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

from qepc.utils.paths import get_project_root
from .eoin_data_source import load_eoin_player_boxes, load_eoin_team_boxes


# ---------------------------------------------------------------------------
# Dataclasses for configuration
# ---------------------------------------------------------------------------


@dataclass
class DecoherenceConfig:
    """Time constants (days) for exponential recency decay of priors."""
    minutes_tau: float = 18.0
    usage_tau: float = 16.0
    efficiency_tau: float = 20.0
    variance_tau: float = 30.0


@dataclass
class BlendWeights:
    """Weights for blending season/recency/decoherence components."""
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
    shrinkage: float = 0.5  # 0=no shrink, 1=shrink strongly to 0
    variance_boost: float = 0.25
    min_games: int = 6


@dataclass
class PlayerPointsConfig:
    """Tunable knobs for the player points model."""
    recent_window: int = 5
    min_history_games: int = 4

    # Minutes handling / stability
    minutes_clip_min: float = 0.0
    minutes_clip_max: float = 55.0   # OT/outlier bugs exist in Eoin; cap them
    minutes_alpha: float = 1.18      # nonlinear scaling (>1 boosts high-minute roles)

    decoherence: DecoherenceConfig = field(default_factory=DecoherenceConfig)
    weights: BlendWeights = field(default_factory=BlendWeights)
    entanglement: EntanglementConfig = field(default_factory=EntanglementConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict) -> "PlayerPointsConfig":
        base = cls()
        deco = data.get("decoherence", {})
        weights = data.get("weights", {})
        ent = data.get("entanglement", {})
        return cls(
            recent_window=int(data.get("recent_window", base.recent_window)),
            min_history_games=int(data.get("min_history_games", base.min_history_games)),
            minutes_clip_min=float(data.get("minutes_clip_min", base.minutes_clip_min)),
            minutes_clip_max=float(data.get("minutes_clip_max", base.minutes_clip_max)),
            minutes_alpha=float(data.get("minutes_alpha", base.minutes_alpha)),
            decoherence=DecoherenceConfig(**deco) if not isinstance(deco, DecoherenceConfig) else deco,
            weights=(BlendWeights(**weights) if not isinstance(weights, BlendWeights) else weights).normalized(),
            entanglement=EntanglementConfig(**ent) if not isinstance(ent, EntanglementConfig) else ent,
            seed=int(data.get("seed", base.seed)),
        )

    def to_dict(self) -> Dict:
        return {
            "recent_window": self.recent_window,
            "min_history_games": self.min_history_games,
            "minutes_clip_min": self.minutes_clip_min,
            "minutes_clip_max": self.minutes_clip_max,
            "minutes_alpha": self.minutes_alpha,
            "decoherence": asdict(self.decoherence),
            "weights": asdict(self.weights.normalized()),
            "entanglement": asdict(self.entanglement),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
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


def _as_1d_series(obj, index: pd.Index, name: str) -> pd.Series:
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
    cols = {c.lower(): c for c in df.columns}
    preferred = [
        "minutes", "min", "mp", "mins", "minutes_played", "minutesplayed",
        "minsplayed", "numminutes",
    ]
    for key in preferred:
        if key in cols:
            return cols[key]
    for lc, orig in cols.items():
        if "minute" in lc:
            return orig
        if lc.endswith("min") and len(lc) <= 8:
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
    merged["team_points"] = merged["team_points"].replace(0, np.nan)
    merged["points_share"] = _safe_div(merged["points"], merged["team_points"])
    return merged


def _decoherence_prior_mean(group: pd.DataFrame, value_col: str, tau_days: float) -> pd.Series:
    """Exponential-decay mean using only previous games."""
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
    """Exponential-decay variance using only previous games."""
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
    return s.shift(1).rolling(window=window, min_periods=1).mean()


def _minutes_scaled(minutes: pd.Series, alpha: float) -> pd.Series:
    """Scale minutes nonlinearly while preserving scale at 24 minutes."""
    m = pd.to_numeric(minutes, errors="coerce").fillna(0.0)
    alpha = float(alpha)
    if not np.isfinite(alpha) or alpha <= 0:
        alpha = 1.0
    base = 24.0
    ratio = (m / base).clip(lower=1e-6)
    return m * (ratio ** (alpha - 1.0))


# ---------------------------------------------------------------------------
# Public API
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
    else:
        df["minutes_actual"] = np.nan

    df["minutes_actual"] = df["minutes_actual"].clip(
        lower=float(cfg.minutes_clip_min),
        upper=float(cfg.minutes_clip_max),
    )

    if min_minutes and min_minutes > 0:
        df = df[df["minutes_actual"] >= float(min_minutes)].copy()

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

    global_min_med = float(df["minutes_actual"].median()) if df["minutes_actual"].notna().any() else 24.0
    df["minutes_coherent"] = (
        df["minutes_expected"]
        .fillna(global_min_med)
        .clip(lower=float(cfg.minutes_clip_min), upper=float(cfg.minutes_clip_max))
    )

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
    minutes_component = _minutes_scaled(df["minutes_coherent"], alpha=float(cfg.minutes_alpha)) * df["efficiency_coherent"]

    df["predicted_points"] = (
        weights.season * df["season_mean_pts"]
        + weights.recency * df["recency_mean_pts"]
        + weights.decoherence * minutes_component
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
        min_n = int(cfg.entanglement.min_games)
        for _, row in sub.iterrows():
            if len(pts_hist) >= min_n:
                pts = np.array(pts_hist, dtype="float64")
                tm = np.array(team_hist, dtype="float64")
                mask = np.isfinite(pts) & np.isfinite(tm)
                if int(mask.sum()) >= min_n:
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
        "game_id", "player_id", "team_id", "team_name", "game_date",
        "points", "predicted_points", "predicted_variance", "cum_games",
        "season_mean_pts", "recency_mean_pts",
        "minutes_actual", "minutes_expected", "minutes_coherent",
        "usage_coherent", "efficiency_coherent", "entanglement_corr",
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

    return {"mae": mae, "bias": bias, "calibration": calib_rows, "predictions": preds}
