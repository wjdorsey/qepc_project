"""Player points projection utilities with quantum-inspired twists.

This module projects NBA player **points** from Eoin boxscores in a way that is:

- **Leakage-safe**: all rolling/recency features use only prior games (shift(1)).
- **Portable**: all data paths resolve via QEPC PROJECT_ROOT auto-detect.
- **Robust**: dtype-hardened joins + NaN-safe weighted statistics.

Key idea (minutes coherence)
----------------------------
We treat minutes as a *latent/expected* quantity ("coherent minutes") derived
from **prior** games, not the current game's boxscore. This prevents leakage.

Important bug fix (rolling leakage across players)
--------------------------------------------------
Older drafts accidentally applied `.rolling(...)` to a globally-shifted Series,
which mixes players together. This version uses **groupby-rolling** with a
within-player shift to ensure recency features are player-local and leakage-safe.

Returned DataFrame contains at least:
- game_id, player_id, team_id, team_name, game_date
- actual_points, predicted_points, predicted_variance
- cum_games, minutes_actual, minutes_expected, minutes_coherent
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd

import sys
import time

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

    # Minutes coherence blend (recency vs decoherence prior)
    minutes_decoherence_weight: float = 0.15  # small smoothing; keep it reactive

    decoherence: DecoherenceConfig = field(default_factory=DecoherenceConfig)
    weights: BlendWeights = field(default_factory=BlendWeights)
    entanglement: EntanglementConfig = field(default_factory=EntanglementConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlayerPointsConfig":
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
            minutes_decoherence_weight=float(
                data.get("minutes_decoherence_weight", base.minutes_decoherence_weight)
            ),
            decoherence=DecoherenceConfig(**deco) if not isinstance(deco, DecoherenceConfig) else deco,
            weights=(BlendWeights(**weights) if not isinstance(weights, BlendWeights) else weights).normalized(),
            entanglement=EntanglementConfig(**ent) if not isinstance(ent, EntanglementConfig) else ent,
            seed=int(data.get("seed", base.seed)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recent_window": self.recent_window,
            "min_history_games": self.min_history_games,
            "minutes_clip_min": self.minutes_clip_min,
            "minutes_clip_max": self.minutes_clip_max,
            "minutes_alpha": self.minutes_alpha,
            "minutes_decoherence_weight": self.minutes_decoherence_weight,
            "decoherence": asdict(self.decoherence),
            "weights": asdict(self.weights.normalized()),
            "entanglement": asdict(self.entanglement),
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log(msg: str, enabled: bool) -> None:
    """Print a stage message when progress is enabled."""
    if enabled:
        print(msg, flush=True)


def _progress_iter(iterable, *, total: Optional[int] = None, desc: str = "", enabled: bool = False):
    """Iterate with a progress bar (tqdm if available, else a simple text bar)."""
    if not enabled:
        for item in iterable:
            yield item
        return

    try:
        from tqdm import tqdm  # type: ignore
        for item in tqdm(iterable, total=total, desc=desc):
            yield item
        return
    except Exception:
        pass

    n = 0
    t0 = time.time()
    for item in iterable:
        n += 1
        if total:
            pct = 100.0 * n / float(total)
            elapsed = time.time() - t0
            sys.stdout.write(f"\r{desc}: {n}/{total} ({pct:5.1f}%) elapsed={elapsed:6.1f}s")
        else:
            sys.stdout.write(f"\r{desc}: {n}")
        sys.stdout.flush()
        yield item
    sys.stdout.write("\n")
    sys.stdout.flush()


def _apply_by_group(
    df: pd.DataFrame,
    group_col: str,
    fn,
    *,
    desc: str,
    enabled: bool,
) -> pd.Series:
    """Apply `fn(group_df)->Series` per group and stitch results back to original index.

    Used only to show progress for expensive per-player loops. When progress is
    disabled, we fall back to pandas groupby.apply for speed.
    """
    if df.empty:
        return pd.Series(index=df.index, dtype="float64")

    out = np.full(len(df), np.nan, dtype="float64")
    gb = df.groupby(group_col, sort=False)
    total = int(getattr(gb, "ngroups", 0) or 0) if enabled else None

    for _, grp in _progress_iter(gb, total=total, desc=desc, enabled=enabled):
        res = fn(grp)
        if not isinstance(res, pd.Series):
            res = pd.Series(res, index=grp.index)
        idx = grp.index.to_numpy(dtype=int, copy=False)
        out[idx] = pd.to_numeric(res, errors="coerce").to_numpy(dtype="float64", copy=False)

    return pd.Series(out, index=df.index)

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
    """Convert a groupby-apply result to a flat Series aligned to `index`."""
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


def _prior_group_rolling_mean(df: pd.DataFrame, group_col: str, value_col: str, window: int) -> pd.Series:
    """Within-group rolling mean using only previous rows (shift(1))."""
    s = pd.to_numeric(df[value_col], errors="coerce")
    rolled = s.groupby(df[group_col], sort=False).rolling(window=window, min_periods=1).mean()
    rolled = rolled.groupby(level=0).shift(1)
    return rolled.reset_index(level=0, drop=True)


def _prior_group_rolling_sum(df: pd.DataFrame, group_col: str, value_col: str, window: int) -> pd.Series:
    """Within-group rolling sum using only previous rows (shift(1))."""
    s = pd.to_numeric(df[value_col], errors="coerce")
    rolled = s.groupby(df[group_col], sort=False).rolling(window=window, min_periods=1).sum()
    rolled = rolled.groupby(level=0).shift(1)
    return rolled.reset_index(level=0, drop=True)


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
    config: Optional[Union[PlayerPointsConfig, Dict[str, Any]]] = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Compute leakage-free per-game player points expectations."""
    if config is None:
        cfg = PlayerPointsConfig()
    elif isinstance(config, PlayerPointsConfig):
        cfg = config
    else:
        cfg = PlayerPointsConfig.from_dict(config)
    cfg = PlayerPointsConfig.from_dict(cfg.to_dict())  # normalize weights & types
    np.random.seed(cfg.seed)

    root = get_project_root()

    _log("Stage 1/7: Loading Eoin boxscores…", progress)

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
        # This is an evaluation filter; it's okay that it uses current game's minutes.
        df = df[df["minutes_actual"] >= float(min_minutes)].copy()

    if start_date is not None:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    df["game_id"] = pd.to_numeric(df["game_id"], errors="coerce").astype("Int64")
    df["player_id"] = pd.to_numeric(df.get("player_id"), errors="coerce").astype("Int64")
    df = df.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)

    # Harden points to numeric
    df["points_num"] = pd.to_numeric(df["points"], errors="coerce")

    _log("Stage 2/7: Building leakage-safe rolling features…", progress)

    # ---- Season / recency features (leakage-safe) ----
    g = df.groupby("player_id", sort=False)

    df["cum_games"] = g.cumcount()

    df["cum_points_sum"] = g["points_num"].cumsum() - df["points_num"]
    df["season_mean_pts"] = df["cum_points_sum"] / df["cum_games"].replace(0, np.nan)

    df["recency_mean_pts"] = _prior_group_rolling_mean(df, "player_id", "points_num", window=int(cfg.recent_window))

    # minutes expected (prior)
    df["minutes_expected"] = _prior_group_rolling_mean(df, "player_id", "minutes_actual", window=10)

    # optional smoother minutes prior (decoherence)
    _log("Stage 3/7: Computing minutes coherence priors…", progress)

    if progress:
        df["minutes_decoherent"] = _apply_by_group(
            df,
            "player_id",
            lambda sub: _decoherence_prior_mean(
                sub, value_col="minutes_actual", tau_days=cfg.decoherence.minutes_tau
            ),
            desc="Minutes decoherence",
            enabled=True,
        )
    else:
        minutes_deco_raw = df.groupby("player_id", sort=False).apply(
            _decoherence_prior_mean, value_col="minutes_actual", tau_days=cfg.decoherence.minutes_tau
        )
        df["minutes_decoherent"] = _as_1d_series(minutes_deco_raw, df.index, name="minutes_decoherent")

    global_min_med = float(df["minutes_actual"].median()) if df["minutes_actual"].notna().any() else 24.0
    w_m = float(np.clip(cfg.minutes_decoherence_weight, 0.0, 1.0))
    df["minutes_coherent"] = (
        (1.0 - w_m) * df["minutes_expected"] + w_m * df["minutes_decoherent"]
    )
    df["minutes_coherent"] = (
        df["minutes_coherent"]
        .fillna(df["minutes_expected"])
        .fillna(df["minutes_decoherent"])
        .fillna(global_min_med)
        .clip(lower=float(cfg.minutes_clip_min), upper=float(cfg.minutes_clip_max))
    )

    df["minutes_scaled"] = _minutes_scaled(df["minutes_coherent"], alpha=float(cfg.minutes_alpha))

    # ---- PPM features (for role-aware calibration) ----
    df["points_per_min"] = _safe_div(df["points_num"], df["minutes_actual"])

    df["cum_minutes_sum"] = g["minutes_actual"].cumsum() - df["minutes_actual"]
    df["season_ppm"] = _safe_div(df["cum_points_sum"], df["cum_minutes_sum"])

    df["recency_points_sum"] = _prior_group_rolling_sum(df, "player_id", "points_num", window=int(cfg.recent_window))
    df["recency_minutes_sum"] = _prior_group_rolling_sum(df, "player_id", "minutes_actual", window=int(cfg.recent_window))
    df["recency_ppm"] = _safe_div(df["recency_points_sum"], df["recency_minutes_sum"])

    # ---- Decoherence priors (usage/efficiency/variance) ----
    _log("Stage 4/7: Computing decoherence priors (usage/efficiency/variance)…", progress)

    if progress:
        df["usage_coherent"] = _apply_by_group(
            df,
            "player_id",
            lambda sub: _decoherence_prior_mean(
                sub, value_col="points_share", tau_days=cfg.decoherence.usage_tau
            ),
            desc="Usage decoherence",
            enabled=True,
        )
    else:
        df["usage_coherent"] = _as_1d_series(
            df.groupby("player_id", sort=False).apply(
                _decoherence_prior_mean, value_col="points_share", tau_days=cfg.decoherence.usage_tau
            ),
            df.index,
            name="usage_coherent",
        )
    if progress:
        df["efficiency_coherent"] = _apply_by_group(
            df,
            "player_id",
            lambda sub: _decoherence_prior_mean(
                sub, value_col="points_per_min", tau_days=cfg.decoherence.efficiency_tau
            ),
            desc="Efficiency decoherence",
            enabled=True,
        )
    else:
        df["efficiency_coherent"] = _as_1d_series(
            df.groupby("player_id", sort=False).apply(
                _decoherence_prior_mean, value_col="points_per_min", tau_days=cfg.decoherence.efficiency_tau
            ),
            df.index,
            name="efficiency_coherent",
        )
    if progress:
        df["variance_prior"] = _apply_by_group(
            df,
            "player_id",
            lambda sub: _historical_variance(
                sub, value_col="points_num", tau_days=cfg.decoherence.variance_tau
            ),
            desc="Points variance prior",
            enabled=True,
        )
    else:
        df["variance_prior"] = _as_1d_series(
            df.groupby("player_id", sort=False).apply(
                _historical_variance, value_col="points_num", tau_days=cfg.decoherence.variance_tau
            ),
            df.index,
            name="variance_prior",
        )

    _log("Stage 5/7: Blending priors into PPM + recomputing points…", progress)

    # ---- Prediction: blend in PPM-space, then scale by coherent minutes ----
    weights = cfg.weights.normalized()
    df["ppm_pred"] = (
        weights.season * df["season_ppm"]
        + weights.recency * df["recency_ppm"]
        + weights.decoherence * df["efficiency_coherent"]
    )
    df["ppm_pred"] = (
        df["ppm_pred"]
        .fillna(df["recency_ppm"])
        .fillna(df["season_ppm"])
        .fillna(df["efficiency_coherent"])
    )

    df["predicted_points"] = df["minutes_scaled"] * df["ppm_pred"]
    # Final fallback: if minutes_scaled is 0/NaN, fall back to point-space priors
    df["predicted_points"] = df["predicted_points"].fillna(df["recency_mean_pts"]).fillna(df["season_mean_pts"])

    # ---- Team context + entanglement variance adjustment ----
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
            pts_hist.append(pd.to_numeric(row.get("points_num"), errors="coerce"))
            team_hist.append(pd.to_numeric(row.get("team_points_game"), errors="coerce"))
        return pd.Series(corr_vals, index=sub.index)

    _log("Stage 6/7: Computing entanglement correlations…", progress)

    if cfg.entanglement.enabled and team_boxes is not None and "team_points_game" in df.columns:
        if progress:
            df["entanglement_corr"] = _apply_by_group(
                df,
                "player_id",
                _compute_corr,
                desc="Entanglement corr",
                enabled=True,
            )
        else:
            corr_raw = df.groupby("player_id").apply(_compute_corr)
            df["entanglement_corr"] = _as_1d_series(corr_raw, df.index, name="entanglement_corr")
    else:
        df["entanglement_corr"] = 0.0

    global_var = float(pd.to_numeric(df["points_num"], errors="coerce").var()) if df["points_num"].notna().any() else 25.0
    base_var = pd.to_numeric(df["variance_prior"], errors="coerce").fillna(global_var).clip(lower=1e-6)
    df["predicted_variance"] = base_var

    if cfg.entanglement.enabled:
        df["predicted_variance"] = df["predicted_variance"] + (
            np.square(pd.to_numeric(df["entanglement_corr"], errors="coerce").fillna(0.0))
            * float(cfg.entanglement.variance_boost)
            * pd.to_numeric(df["team_points_prior_var"], errors="coerce").fillna(0.0)
        )

    _log("Stage 7/7: Finalizing output…", progress)

    out_cols = [
        "game_id", "player_id", "team_id", "team_name", "game_date",
        "points_num", "predicted_points", "predicted_variance", "cum_games",
        "season_mean_pts", "recency_mean_pts",
        "minutes_actual", "minutes_expected", "minutes_decoherent", "minutes_coherent", "minutes_scaled",
        "season_ppm", "recency_ppm", "ppm_pred",
        "usage_coherent", "efficiency_coherent", "entanglement_corr",
    ]
    out = df[out_cols].rename(columns={"points_num": "actual_points"}).copy()

    if cfg.min_history_games and int(cfg.min_history_games) > 0:
        out = out[out["cum_games"] >= int(cfg.min_history_games)].reset_index(drop=True)

    return out



def backtest_player_points(
    *,
    start_date: str,
    end_date: str,
    min_minutes: int = 0,
    config: Optional[Union[PlayerPointsConfig, Dict[str, Any]]] = None,
    buckets: Optional[Iterable[int]] = None,
    # Points affine calibration
    affine_slope: Optional[float] = None,
    affine_intercept: Optional[float] = None,
    apply_insample_affine: bool = False,
    # Minutes affine calibration (applied BEFORE points prediction)
    minutes_affine_slope: Optional[float] = None,
    minutes_affine_intercept: Optional[float] = None,
    apply_insample_minutes_affine: bool = False,
    progress: bool = False,
) -> Dict[str, Any]:
    """Run a backtest over the specified date range and return metrics + diagnostics.

    Notes
    -----
    - Raw predictions are produced by build_player_points_expectations (leakage-safe).
    - Optional **minutes affine** calibration adjusts minutes_coherent before recomputing points.
    - Optional **points affine** calibration adjusts the chosen points prediction (raw or minutes-cal).
    """
    # Normalize config (so we can reuse minutes_alpha / clip bounds consistently)
    if config is None:
        cfg = PlayerPointsConfig()
    elif isinstance(config, PlayerPointsConfig):
        cfg = config
    else:
        cfg = PlayerPointsConfig.from_dict(config)
    cfg = PlayerPointsConfig.from_dict(cfg.to_dict())  # normalize weights & types

    # Build predictions using *all* historical games up to end_date.
    # Important: do NOT apply evaluation filters (start_date/min_minutes) before feature building,
    # otherwise we distort each player's priors (e.g., dropping low-minute games) and inflate leakage-like bias.
    preds_all = build_player_points_expectations(
        start_date=None,
        end_date=end_date,
        min_minutes=0,
        config=cfg,
        progress=bool(progress),
    ).copy()

    # Apply evaluation filters AFTER predictions are computed.
    preds = preds_all.copy()
    preds = preds[preds["game_date"] >= pd.to_datetime(start_date)]
    preds = preds[preds["game_date"] <= pd.to_datetime(end_date)]
    if min_minutes and min_minutes > 0:
        mins_eval = pd.to_numeric(preds.get("minutes_actual"), errors="coerce")
        preds = preds[mins_eval >= float(min_minutes)]
    preds = preds.copy()

    preds = preds.dropna(subset=["predicted_points", "actual_points"])
    preds["predicted_points_raw"] = pd.to_numeric(preds["predicted_points"], errors="coerce")
    preds["actual_points"] = pd.to_numeric(preds["actual_points"], errors="coerce")

    if buckets is None:
        buckets = [0, 20, 28, 34]
    bin_edges = list(buckets) + [np.inf]
    labels = [f"{bin_edges[i]}-{bin_edges[i+1]} mins" for i in range(len(bin_edges) - 1)]

    mins_actual = pd.to_numeric(preds.get("minutes_actual"), errors="coerce")
    mins_pred = pd.to_numeric(preds.get("minutes_coherent"), errors="coerce")

    def _calib(mins: pd.Series, pred_col: str) -> list[Dict[str, Any]]:
        mins2 = mins.fillna(0.0)
        tmp = preds.copy()
        tmp["_mins_bucket"] = pd.cut(
            mins2,
            bins=bin_edges,
            labels=labels,
            include_lowest=True,
            right=False,
        )
        rows: list[Dict[str, Any]] = []
        for bucket_label, group in tmp.groupby("_mins_bucket", observed=False):
            if group.empty:
                continue
            err = pd.to_numeric(group[pred_col], errors="coerce") - pd.to_numeric(group["actual_points"], errors="coerce")
            rows.append(
                {
                    "bucket": str(bucket_label),
                    "n": int(group.shape[0]),
                    "mae": float(np.abs(err).mean()),
                    "bias": float(err.mean()),
                    "pred_mean": float(pd.to_numeric(group[pred_col], errors="coerce").mean()),
                    "actual_mean": float(pd.to_numeric(group["actual_points"], errors="coerce").mean()),
                }
            )
        return rows

    # -----------------------
    # RAW points metrics
    # -----------------------
    raw_err = preds["predicted_points_raw"] - preds["actual_points"]
    raw_mae = float(np.abs(raw_err).mean()) if not preds.empty else np.nan
    raw_bias = float(raw_err.mean()) if not preds.empty else np.nan

    calib_actual_raw = _calib(mins_actual, pred_col="predicted_points_raw")
    calib_pred_raw = _calib(mins_pred, pred_col="predicted_points_raw")

    # -----------------------
    # Minutes diagnostics + optional minutes affine calibration
    # -----------------------
    mins_mask = mins_actual.notna() & mins_pred.notna()
    mins_err = (mins_pred - mins_actual).where(mins_mask)
    minutes_mae = float(np.abs(mins_err).mean()) if mins_mask.any() else np.nan
    minutes_bias = float(mins_err.mean()) if mins_mask.any() else np.nan

    # Fit minutes scale diagnostic: actual_minutes ≈ intercept + slope * pred_minutes
    if mins_mask.any():
        x = mins_pred[mins_mask].to_numpy(dtype="float64")
        y = mins_actual[mins_mask].to_numpy(dtype="float64")
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        x_var = float(np.var(x))
        cov = float(np.mean((x - x_mean) * (y - y_mean)))
        minutes_slope = cov / x_var if x_var > 1e-12 else np.nan
        minutes_intercept = y_mean - minutes_slope * x_mean if np.isfinite(minutes_slope) else np.nan
        minutes_corr = float(np.corrcoef(x, y)[0, 1]) if x.size > 2 else np.nan
        minutes_std_ratio = float(np.std(x) / np.std(y)) if np.std(y) > 1e-12 else np.nan
    else:
        minutes_slope = minutes_intercept = minutes_corr = minutes_std_ratio = np.nan

    # Optionally "fit on this range" (diagnostic only)
    if apply_insample_minutes_affine and np.isfinite(minutes_slope) and np.isfinite(minutes_intercept):
        minutes_affine_slope = float(minutes_slope)
        minutes_affine_intercept = float(minutes_intercept)

    applied_minutes_affine: Optional[Dict[str, float]] = None
    mins_pred_cal = None
    mins_scaled_cal = None

    if minutes_affine_slope is not None and minutes_affine_intercept is not None:
        a_m = float(minutes_affine_intercept)
        b_m = float(minutes_affine_slope)
        mins_pred_cal = (a_m + b_m * mins_pred).clip(
            lower=float(cfg.minutes_clip_min),
            upper=float(cfg.minutes_clip_max),
        )
        mins_scaled_cal = _minutes_scaled(mins_pred_cal, alpha=float(cfg.minutes_alpha))
        preds["minutes_coherent_calibrated"] = mins_pred_cal
        preds["minutes_scaled_calibrated"] = mins_scaled_cal
        applied_minutes_affine = {"slope": b_m, "intercept": a_m}

        # Minutes diagnostics after calibration
        mins_mask2 = mins_actual.notna() & mins_pred_cal.notna()
        mins_err2 = (mins_pred_cal - mins_actual).where(mins_mask2)
        minutes_mae_cal = float(np.abs(mins_err2).mean()) if mins_mask2.any() else np.nan
        minutes_bias_cal = float(mins_err2.mean()) if mins_mask2.any() else np.nan
    else:
        minutes_mae_cal = minutes_bias_cal = np.nan

    # -----------------------
    # PPM diagnostics (implied pred PPM vs actual PPM)
    # -----------------------
    ppm_actual = _safe_div(preds["actual_points"], mins_actual)
    ppm_pred_implied = _safe_div(preds["predicted_points_raw"], mins_pred)
    ppm_mask = ppm_actual.notna() & ppm_pred_implied.notna()
    ppm_err = (ppm_pred_implied - ppm_actual).where(ppm_mask)
    ppm_mae = float(np.abs(ppm_err).mean()) if ppm_mask.any() else np.nan
    ppm_bias = float(ppm_err.mean()) if ppm_mask.any() else np.nan

    # -----------------------
    # Optional minutes-calibrated POINTS (recompute points using calibrated minutes)
    # -----------------------
    points_base_col = "predicted_points_raw"
    points_mincal_mae = points_mincal_bias = np.nan
    calib_actual_mincal: list[Dict[str, Any]] = []
    calib_pred_mincal: list[Dict[str, Any]] = []

    if mins_scaled_cal is not None:
        ppm_model = pd.to_numeric(preds.get("ppm_pred"), errors="coerce")
        pts_mincal = (mins_scaled_cal * ppm_model).clip(lower=0.0)
        # fallback if ppm missing
        pts_mincal = pts_mincal.fillna(preds["predicted_points_raw"])
        preds["predicted_points_mincal"] = pts_mincal
        points_base_col = "predicted_points_mincal"

        err_mincal = preds["predicted_points_mincal"] - preds["actual_points"]
        points_mincal_mae = float(np.abs(err_mincal).mean()) if not preds.empty else np.nan
        points_mincal_bias = float(err_mincal.mean()) if not preds.empty else np.nan

        calib_actual_mincal = _calib(mins_actual, pred_col="predicted_points_mincal")
        # bucket by *calibrated* predicted minutes if available
        calib_pred_mincal = _calib(
            pd.to_numeric(preds.get("minutes_coherent_calibrated"), errors="coerce"),
            pred_col="predicted_points_mincal",
        )

    # -----------------------
    # Scale diagnostic for the current base points prediction
    # -----------------------
    p = pd.to_numeric(preds[points_base_col], errors="coerce")
    a = pd.to_numeric(preds["actual_points"], errors="coerce")
    mask = p.notna() & a.notna()
    if mask.any():
        p0 = p[mask].to_numpy(dtype="float64")
        a0 = a[mask].to_numpy(dtype="float64")
        p_mean = float(np.mean(p0))
        a_mean = float(np.mean(a0))
        p_var = float(np.var(p0))
        cov = float(np.mean((p0 - p_mean) * (a0 - a_mean)))
        slope = cov / p_var if p_var > 1e-12 else np.nan
        intercept = a_mean - slope * p_mean if np.isfinite(slope) else np.nan
        corr = float(np.corrcoef(p0, a0)[0, 1]) if p0.size > 2 else np.nan
        std_pred = float(np.std(p0))
        std_actual = float(np.std(a0))
        std_ratio = std_pred / std_actual if std_actual > 1e-12 else np.nan
    else:
        slope = intercept = corr = std_pred = std_actual = std_ratio = np.nan

    diagnostics = {
        "minutes": {"mae": minutes_mae, "bias": minutes_bias},
        "minutes_scale": {"slope": minutes_slope, "intercept": minutes_intercept, "corr": minutes_corr, "std_ratio": minutes_std_ratio},
        "minutes_calibrated": {"mae": minutes_mae_cal, "bias": minutes_bias_cal} if applied_minutes_affine else {},
        "ppm": {"mae": ppm_mae, "bias": ppm_bias},
        "scale": {"slope": slope, "intercept": intercept, "corr": corr, "std_pred": std_pred, "std_actual": std_actual, "std_ratio": std_ratio},
    }

    # -----------------------
    # Optional affine calibration of POINTS (applied to base prediction)
    # -----------------------
    mae_cal = bias_cal = np.nan
    calib_actual_cal: list[Dict[str, Any]] = []
    calib_pred_cal: list[Dict[str, Any]] = []
    applied_affine: Optional[Dict[str, float]] = None

    if apply_insample_affine and np.isfinite(slope) and np.isfinite(intercept):
        affine_slope = float(slope)
        affine_intercept = float(intercept)

    if affine_slope is not None and affine_intercept is not None:
        a_p = float(affine_intercept)
        b_p = float(affine_slope)
        preds["predicted_points_calibrated"] = (a_p + b_p * preds[points_base_col]).clip(lower=0.0)
        err_cal = preds["predicted_points_calibrated"] - preds["actual_points"]
        mae_cal = float(np.abs(err_cal).mean()) if not preds.empty else np.nan
        bias_cal = float(err_cal.mean()) if not preds.empty else np.nan

        calib_actual_cal = _calib(mins_actual, pred_col="predicted_points_calibrated")
        # bucket by predicted minutes (calibrated if present)
        pred_minutes_for_bucket = pd.to_numeric(
            preds.get("minutes_coherent_calibrated", preds.get("minutes_coherent")),
            errors="coerce",
        )
        calib_pred_cal = _calib(pred_minutes_for_bucket, pred_col="predicted_points_calibrated")
        applied_affine = {"slope": b_p, "intercept": a_p}

    return {
        # Raw points metrics (always the un-calibrated model output)
        "mae": raw_mae,
        "bias": raw_bias,
        "calibration_actual_minutes": calib_actual_raw,
        "calibration_pred_minutes": calib_pred_raw,

        # Optional minutes-affine results
        "applied_minutes_affine": applied_minutes_affine,
        "mae_minutes_calibrated": points_mincal_mae,
        "bias_minutes_calibrated": points_mincal_bias,
        "calibration_actual_minutes_mincal": calib_actual_mincal,
        "calibration_pred_minutes_mincal": calib_pred_mincal,

        # Optional points-affine results (applied to the base prediction: raw or minutes-cal)
        "mae_calibrated": mae_cal,
        "bias_calibrated": bias_cal,
        "applied_affine": applied_affine,
        "calibration_actual_minutes_calibrated": calib_actual_cal,
        "calibration_pred_minutes_calibrated": calib_pred_cal,

        "diagnostics": diagnostics,
        "predictions": preds,
        # indicate which column was used as the base for points affine
        "points_base_col": points_base_col,
    }
