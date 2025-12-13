"""Quantum-inspired totals predictor for QEPC NBA workflows.

This module provides a **lightweight**, **leakage-aware** totals model intended
for rapid iteration and reproducible backtests.

Key ideas used here (quantum-themed but classical compute):
- **Decoherence / recency weighting**: exponential decay on team offense/defense/pace.
- **Entanglement (correlation)**: correlated sampling of home/away points.
- **Entropy**: distribution spread used as a confidence signal for shrinkage/blending.

Design goals:
- Keep dependencies minimal (numpy/pandas).
- Avoid accidental leakage by using shifted/chronological features.
- Be robust to mixed datetime types (tz-aware vs naive) in the Eoin dataset.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from qepc.nba.eoin_data_source import load_eoin_games, load_eoin_team_boxes
from qepc.nba.odds_long_loader import attach_odds_to_games, load_long_odds
from qepc.utils.paths import get_project_root


# -----------------------
# Config
# -----------------------

@dataclass
class TotalsConfig:
    """Configuration for the QPA totals model.

    Note: defaults are intentionally conservative so the model doesn't
    explode before tuning.
    """

    # decoherence half-life-like constants (days)
    tau_offense: float = 35.0
    tau_defense: float = 35.0
    tau_pace: float = 25.0

    # blend weights for expected points (offense vs opponent defense)
    offense_weight: float = 0.70
    defense_weight: float = 0.30

    # how strongly pace scales the *total* (not added as raw points)
    pace_weight: float = 0.05

    # max strength of vegas blending; actual applied weight depends on uncertainty
    vegas_weight: float = 0.25

    # confidence shrinkage strength for league prior (larger => more shrink when uncertain)
    entropy_shrink: float = 0.30

    # correlation shrinkage toward 0
    corr_shrink: float = 0.25

    sample_size: int = 256

    @classmethod
    def from_dict(cls, data: Dict) -> "TotalsConfig":
        return cls(**data)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


# -----------------------
# Datetime helpers
# -----------------------

def _to_game_date_series(df: pd.DataFrame) -> pd.Series:
    """Best-effort game date series (tz-safe), normalized to midnight (naive)."""
    s = df.get("game_date")
    if s is None:
        s = df.get("game_datetime")
    if s is None:
        return pd.Series(pd.NaT, index=df.index)

    dt_series = pd.to_datetime(s, errors="coerce", utc=True)
    # Convert to naive local-ish date (we only care about date ordering / matching)
    dt_series = dt_series.dt.tz_convert(None)
    return dt_series.dt.normalize()


# -----------------------
# Decoherence features
# -----------------------

def _exp_decay_shifted(values: pd.Series, dates: pd.Series, tau: float) -> pd.Series:
    """Leakage-free exponential decay using exp(-Δt/τ) with shift(1).

    The returned series represents the decayed average *before* the
    current observation. If insufficient history exists, NaN is returned.
    """
    if tau <= 0:
        return pd.Series(np.nan, index=values.index)

    out = []
    weighted_sum = 0.0
    weight_total = 0.0
    prev_date: Optional[pd.Timestamp] = None

    for value, dt_ in zip(values, pd.to_datetime(dates, errors="coerce")):
        if pd.isna(dt_):
            out.append(np.nan)
            continue
        if prev_date is None:
            out.append(np.nan)
            weighted_sum = float(value)
            weight_total = 1.0
            prev_date = dt_
            continue

        delta_days = max((dt_ - prev_date).days, 0)
        decay = math.exp(-delta_days / tau)
        weighted_sum *= decay
        weight_total *= decay

        prior_mean = weighted_sum / weight_total if weight_total > 0 else np.nan
        out.append(prior_mean)

        weighted_sum += float(value)
        weight_total += 1.0
        prev_date = dt_

    return pd.Series(out, index=values.index)


def build_team_decoherence_features(
    team_boxes: pd.DataFrame,
    tau_offense: float,
    tau_defense: float,
    tau_pace: float,
) -> pd.DataFrame:
    """Compute leakage-free decayed offense/defense/pace features.

    Expects team_boxes to contain:
    - team_id
    - teamscore (points scored)
    - opponentscore (points allowed)
    - game_date or game_datetime
    """
    df = team_boxes.copy()
    if "team_id" not in df.columns:
        raise KeyError("team_boxes missing required column: team_id")

    df["game_date_join"] = _to_game_date_series(df)

    # be explicit about chronological order inside each team
    df = df.sort_values(["team_id", "game_date_join", "game_id"], kind="mergesort")

    offense_src = df.get("teamscore")
    defense_src = df.get("opponentscore")
    if offense_src is None or defense_src is None:
        raise KeyError("team_boxes must include teamscore and opponentscore columns")

    # Pace proxy: if possessions exists use it; otherwise use total/2 as a crude proxy.
    if "possessions" in df.columns:
        pace_src = df["possessions"]
    else:
        pace_src = (offense_src + defense_src) * 0.5

    df["offense_qpa"] = df.groupby("team_id", sort=False).apply(
        lambda g: _exp_decay_shifted(g["teamscore"], g["game_date_join"], tau_offense)
    ).reset_index(level=0, drop=True)

    df["defense_qpa"] = df.groupby("team_id", sort=False).apply(
        lambda g: _exp_decay_shifted(g["opponentscore"], g["game_date_join"], tau_defense)
    ).reset_index(level=0, drop=True)

    df["pace_qpa"] = df.groupby("team_id", sort=False).apply(
        lambda g: _exp_decay_shifted(pace_src.loc[g.index], g["game_date_join"], tau_pace)
    ).reset_index(level=0, drop=True)

    keep = ["game_id", "team_id", "game_date_join", "offense_qpa", "defense_qpa", "pace_qpa"]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()


def _merge_team_features(games: pd.DataFrame, team_state: pd.DataFrame) -> pd.DataFrame:
    """Attach (shifted) team state to games as home_* and away_* columns."""
    g = games.copy()
    g["game_date_join"] = _to_game_date_series(g)

    # ensure types are stable for joins
    for col in ("home_team_id", "away_team_id"):
        if col in g.columns:
            g[col] = pd.to_numeric(g[col], errors="coerce").astype("Int64")

    ts = team_state.copy()
    ts["team_id"] = pd.to_numeric(ts["team_id"], errors="coerce").astype("Int64")

    home = ts.rename(
        columns={
            "team_id": "home_team_id",
            "offense_qpa": "home_offense_qpa",
            "defense_qpa": "home_defense_qpa",
            "pace_qpa": "home_pace_qpa",
        }
    )
    away = ts.rename(
        columns={
            "team_id": "away_team_id",
            "offense_qpa": "away_offense_qpa",
            "defense_qpa": "away_defense_qpa",
            "pace_qpa": "away_pace_qpa",
        }
    )

    # join on (game_id, team_id). game_date_join is kept for diagnostics, not keying.
    out = g.merge(home.drop(columns=["game_date_join"], errors="ignore"), on=["game_id", "home_team_id"], how="left")
    out = out.merge(away.drop(columns=["game_date_join"], errors="ignore"), on=["game_id", "away_team_id"], how="left")
    return out


def _compute_league_prior_totals(df: pd.DataFrame) -> pd.Series:
    """Compute expanding mean of totals, shifted by 1 game (order dependent)."""
    totals = df.get("total_score")
    if totals is None:
        return pd.Series(np.nan, index=df.index)
    return totals.shift(1).expanding(min_periods=5).mean()


def _compute_league_prior_pace(df: pd.DataFrame) -> pd.Series:
    """Compute expanding mean of game pace proxy, shifted by 1 game (order dependent)."""
    if "home_pace_qpa" not in df.columns or "away_pace_qpa" not in df.columns:
        return pd.Series(np.nan, index=df.index)
    pace_game = (df["home_pace_qpa"] + df["away_pace_qpa"]) * 0.5
    return pace_game.shift(1).expanding(min_periods=5).mean()


def enrich_games_with_config(
    games: pd.DataFrame, team_boxes: pd.DataFrame, config: TotalsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (team_state, games_enriched)."""
    team_state = build_team_decoherence_features(
        team_boxes, config.tau_offense, config.tau_defense, config.tau_pace
    )
    games_enriched = _merge_team_features(games, team_state)

    # ensure total_score exists (used for priors + evaluation)
    if "total_score" not in games_enriched.columns:
        if {"home_score", "away_score"}.issubset(games_enriched.columns):
            games_enriched["total_score"] = games_enriched["home_score"] + games_enriched["away_score"]
        elif {"home_pts", "away_pts"}.issubset(games_enriched.columns):
            games_enriched["total_score"] = games_enriched["home_pts"] + games_enriched["away_pts"]

    # IMPORTANT: priors should be computed in chronological order
    tmp = games_enriched.copy()
    tmp["_sort_date"] = _to_game_date_series(tmp)
    tmp = tmp.sort_values(["_sort_date", "game_id"], kind="mergesort").reset_index(drop=True)

    tmp["league_prior_total"] = _compute_league_prior_totals(tmp)
    tmp["league_prior_pace"] = _compute_league_prior_pace(tmp)

    tmp = tmp.drop(columns=["_sort_date"], errors="ignore")
    return team_state, tmp


# -----------------------
# Correlation + sampling
# -----------------------

def estimate_score_correlation(training_games: pd.DataFrame, shrink: float) -> Tuple[float, float, float]:
    """Estimate std_home, std_away, corr(home, away) with shrinkage."""
    scores = training_games.dropna(subset=["home_score", "away_score"])
    if scores.empty:
        return 12.0, 12.0, 0.15

    std_home = float(scores["home_score"].std(ddof=0) or 12.0)
    std_away = float(scores["away_score"].std(ddof=0) or 12.0)
    corr = float(scores[["home_score", "away_score"]].corr().iloc[0, 1])
    if not np.isfinite(corr):
        corr = 0.0
    corr = (1.0 - float(np.clip(shrink, 0.0, 1.0))) * corr
    corr = float(np.clip(corr, -0.95, 0.95))
    return std_home, std_away, corr


def sample_correlated_totals(
    mean_home: float,
    mean_away: float,
    std_home: float,
    std_away: float,
    rho: float,
    sample_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    cov = np.array(
        [[std_home**2, rho * std_home * std_away], [rho * std_home * std_away, std_away**2]],
        dtype=float,
    )
    samples = rng.multivariate_normal(
        mean=[mean_home, mean_away],
        cov=cov,
        size=int(sample_size),
        check_valid="ignore",
    )
    totals = samples.sum(axis=1)
    return totals


# -----------------------
# Prediction
# -----------------------

def _blend_points(offense: float, opp_defense: float, w_off: float, w_def: float) -> float:
    if pd.isna(offense) and pd.isna(opp_defense):
        return float("nan")
    if pd.isna(offense):
        return float(opp_defense)
    if pd.isna(opp_defense):
        return float(offense)
    return float(w_off * float(offense) + w_def * float(opp_defense))


def predict_totals(
    games_with_features: pd.DataFrame,
    config: TotalsConfig,
    corr_stats: Tuple[float, float, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Add prediction columns to a copy of games_with_features."""
    df = games_with_features.copy()
    std_home, std_away, rho = corr_stats

    # base expected points from offense/defense blend
    df["pred_home_points_base"] = [
        _blend_points(h_off, a_def, config.offense_weight, config.defense_weight)
        for h_off, a_def in zip(df.get("home_offense_qpa"), df.get("away_defense_qpa"))
    ]
    df["pred_away_points_base"] = [
        _blend_points(a_off, h_def, config.offense_weight, config.defense_weight)
        for a_off, h_def in zip(df.get("away_offense_qpa"), df.get("home_defense_qpa"))
    ]

    base_total = df["pred_home_points_base"] + df["pred_away_points_base"]

    # Pace should scale totals, not be added as raw points to both teams.
    pace_factor = pd.Series(1.0, index=df.index, dtype=float)
    if "home_pace_qpa" in df.columns and "away_pace_qpa" in df.columns:
        pace_game = (df["home_pace_qpa"] + df["away_pace_qpa"]) * 0.5
        pace_prior = df.get("league_prior_pace")
        if pace_prior is None:
            # compute on-the-fly (order dependent; caller should already be sorted)
            pace_prior = pace_game.shift(1).expanding(min_periods=5).mean()

        rel = pace_game / pace_prior
        # If missing, default to neutral
        rel = rel.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        pace_factor = 1.0 + config.pace_weight * (rel - 1.0)
        pace_factor = pace_factor.clip(0.85, 1.15)

    pred_total_mean = base_total * pace_factor

    # split scaled total back into home/away in proportion to base components
    share_home = (df["pred_home_points_base"] / base_total).replace([np.inf, -np.inf], np.nan).fillna(0.5)
    share_home = share_home.clip(0.1, 0.9)
    df["pred_home_points"] = pred_total_mean * share_home
    df["pred_away_points"] = pred_total_mean * (1.0 - share_home)

    df["pred_total_mean"] = df["pred_home_points"] + df["pred_away_points"]

    totals = []
    entropies = []
    medians = []
    p10s = []
    p90s = []

    prior_series = df.get("league_prior_total", pd.Series(np.nan, index=df.index))
    vegas_series = df.get("total_points", pd.Series(np.nan, index=df.index))

    for mean_home, mean_away, prior, vegas in zip(
        df["pred_home_points"],
        df["pred_away_points"],
        prior_series,
        vegas_series,
    ):
        if pd.isna(mean_home) or pd.isna(mean_away):
            totals.append(np.nan)
            entropies.append(np.nan)
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)
            continue

        samples = sample_correlated_totals(
            float(mean_home),
            float(mean_away),
            float(std_home),
            float(std_away),
            float(rho),
            int(config.sample_size),
            rng,
        )
        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples))

        # differential entropy proxy for a normal (monotone in std)
        entropy = 0.5 * math.log(2 * math.pi * math.e * (sample_std**2 + 1e-6))

        # Convert entropy to a [0,1] confidence: higher entropy -> lower confidence
        entropy_pos = max(float(entropy), 0.0)
        confidence = 1.0 / (1.0 + config.entropy_shrink * entropy_pos)

        # shrink toward league prior when uncertain
        if not pd.isna(prior):
            prior_f = float(prior)
            sample_mean = confidence * sample_mean + (1.0 - confidence) * prior_f

        # blend in vegas when uncertain (but cap at config.vegas_weight)
        if not pd.isna(vegas):
            vegas_f = float(vegas)
            wv = float(np.clip(config.vegas_weight * (1.0 - confidence), 0.0, config.vegas_weight))
            sample_mean = wv * vegas_f + (1.0 - wv) * sample_mean

        totals.append(sample_mean)
        entropies.append(entropy)
        medians.append(float(np.median(samples)))
        p10s.append(float(np.percentile(samples, 10)))
        p90s.append(float(np.percentile(samples, 90)))

    df["pred_total_qpa"] = totals
    df["pred_entropy"] = entropies
    df["pred_total_median"] = medians
    df["pred_total_p10"] = p10s
    df["pred_total_p90"] = p90s
    return df


# -----------------------
# IO / pipeline helpers
# -----------------------

def load_games_with_features(
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_odds: bool = False,
    config: Optional[TotalsConfig] = None,
    project_root: Optional[Path] = None,
    odds_csv_path: Optional[Union[str, Path]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load games + team boxes and return (games, team_state, games_with_features)."""
    if project_root is None:
        project_root = get_project_root(Path(__file__).resolve())

    games = load_eoin_games(project_root)
    team_boxes = load_eoin_team_boxes(project_root)

    # filter by date (tz-safe)
    if start is not None:
        start_dt = pd.to_datetime(start, errors="coerce", utc=True).tz_convert(None).normalize()
        games = games.loc[_to_game_date_series(games) >= start_dt]
        team_boxes = team_boxes.loc[_to_game_date_series(team_boxes) >= start_dt]
    if end is not None:
        end_dt = pd.to_datetime(end, errors="coerce", utc=True).tz_convert(None).normalize()
        games = games.loc[_to_game_date_series(games) <= end_dt]
        team_boxes = team_boxes.loc[_to_game_date_series(team_boxes) <= end_dt]

    if with_odds:
        if odds_csv_path is None:
            odds_csv_path = project_root / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
        odds_df = load_long_odds(odds_csv_path)
        games, _ = attach_odds_to_games(games, odds_df)

    if config is None:
        config = TotalsConfig()

    team_state, games_enriched = enrich_games_with_config(games, team_boxes, config)
    return games, team_state, games_enriched


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    """Return MAE/bias/n for pred_total_qpa vs total_score."""
    truth = df.get("total_score")
    pred = df.get("pred_total_qpa")
    if truth is None or pred is None:
        return {"mae": float("nan"), "bias": float("nan"), "n": 0}

    mask = truth.notna() & pred.notna()
    if mask.sum() == 0:
        return {"mae": float("nan"), "bias": float("nan"), "n": 0}

    mae = float((truth[mask] - pred[mask]).abs().mean())
    bias = float((pred[mask] - truth[mask]).mean())
    return {"mae": mae, "bias": bias, "n": int(mask.sum())}


def save_config(config: TotalsConfig, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config.to_dict(), indent=2))


def load_config(path: Union[str, Path]) -> TotalsConfig:
    data = json.loads(Path(path).read_text())
    return TotalsConfig.from_dict(data)


def smoketest_small_sample(project_root: Optional[Path] = None) -> None:
    """Quick sanity-check: run a tiny prediction pass."""
    if project_root is None:
        project_root = get_project_root(Path(__file__).resolve())
    games, team_state, enriched = load_games_with_features(
        start="2023-10-01",
        end="2024-01-15",
        with_odds=True,
        project_root=project_root,
    )
    corr_stats = estimate_score_correlation(enriched, TotalsConfig().corr_shrink)
    rng = np.random.default_rng(7)
    preds = predict_totals(enriched, TotalsConfig(), corr_stats, rng)
    metrics = evaluate_predictions(preds)
    print("[qpa_totals smoketest]", metrics)
