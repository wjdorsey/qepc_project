"""Quantum-inspired totals predictor for QEPC NBA workflows.

This module keeps leakage-free computations while remaining lightweight
and reproducible.  It intentionally avoids heavyweight model
dependencies so it can run in constrained environments and during rapid
iteration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from qepc.nba.eoin_data_source import load_eoin_games, load_eoin_team_boxes
from qepc.nba.odds_long_loader import attach_odds_to_games, load_long_odds
from qepc.utils.paths import get_project_root


@dataclass
class TotalsConfig:
    """Configuration for the QPA totals model."""

    tau_offense: float = 18.0
    tau_defense: float = 22.0
    tau_pace: float = 20.0
    offense_weight: float = 0.55
    defense_weight: float = 0.45
    pace_weight: float = 0.15
    vegas_weight: float = 0.35
    entropy_shrink: float = 0.2
    corr_shrink: float = 0.2
    sample_size: int = 256

    @classmethod
    def from_dict(cls, data: Dict) -> "TotalsConfig":
        return cls(**data)

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


def _exp_decay_shifted(values: pd.Series, dates: pd.Series, tau: float) -> pd.Series:
    """Leakage-free exponential decay using exp(-Δt/τ) with shift(1).

    The returned series represents the decayed average *before* the
    current observation.  If insufficient history exists, NaN is
    returned.
    """

    if tau <= 0:
        return pd.Series(np.nan, index=values.index)

    out = []
    weighted_sum = 0.0
    weight_total = 0.0
    prev_date: Optional[pd.Timestamp] = None

    for value, dt in zip(values, pd.to_datetime(dates, errors="coerce")):
        if pd.isna(dt):
            out.append(np.nan)
            continue
        if prev_date is None:
            out.append(np.nan)
            weighted_sum = float(value)
            weight_total = 1.0
            prev_date = dt
            continue

        delta_days = max((dt - prev_date).days, 0)
        decay = math.exp(-delta_days / tau)
        weighted_sum *= decay
        weight_total *= decay

        prior_mean = weighted_sum / weight_total if weight_total > 0 else np.nan
        out.append(prior_mean)

        weighted_sum += float(value)
        weight_total += 1.0
        prev_date = dt

    return pd.Series(out, index=values.index)


def build_team_decoherence_features(
    team_boxes: pd.DataFrame,
    tau_offense: float,
    tau_defense: float,
    tau_pace: float,
) -> pd.DataFrame:
    """Compute leakage-free decayed offense/defense/pace features.

    Parameters
    ----------
    team_boxes : pd.DataFrame
        Team-level box scores containing ``team_id``, ``teamscore``,
        ``opponentscore``, and date columns.
    tau_offense, tau_defense, tau_pace : float
        Half-life like decay constants (in days) for each feature family.
    """

    df = team_boxes.copy()
    if "game_date" not in df.columns:
        df["game_date"] = pd.to_datetime(df["game_datetime"], errors="coerce").dt.date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    df = df.sort_values(["team_id", "game_date", "game_id"])

    pace_source = None
    if "possessions" in df.columns:
        pace_source = df["possessions"]
    elif {"teamscore", "opponentscore"}.issubset(df.columns):
        pace_source = (df["teamscore"] + df["opponentscore"]) / 2.0

    features = []
    df["offense_qpa"] = _exp_decay_shifted(df["teamscore"], df["game_date"], tau_offense)
    df["defense_qpa"] = _exp_decay_shifted(df["opponentscore"], df["game_date"], tau_defense)
    features.extend(["offense_qpa", "defense_qpa"])

    if pace_source is not None:
        df["pace_qpa"] = _exp_decay_shifted(pace_source, df["game_date"], tau_pace)
        features.append("pace_qpa")

    keep_cols = ["game_id", "team_id", "game_date", *features]
    return df[keep_cols]


def _merge_team_features(games: pd.DataFrame, team_state: pd.DataFrame) -> pd.DataFrame:
    df = games.copy()
    if "game_date" not in df.columns:
        df["game_date"] = pd.to_datetime(df["game_datetime"], errors="coerce").dt.date
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")

    merged = df.merge(
        team_state.add_prefix("home_"),
        left_on=["game_id", "home_team_id"],
        right_on=["home_game_id", "home_team_id"],
        how="left",
    )
    merged = merged.merge(
        team_state.add_prefix("away_"),
        left_on=["game_id", "away_team_id"],
        right_on=["away_game_id", "away_team_id"],
        how="left",
    )

    for col in ["home_game_id", "away_game_id"]:
        if col in merged.columns:
            merged = merged.drop(columns=[col])
    return merged


def _blend_points(offense: float, defense: float, offense_weight: float, defense_weight: float) -> float:
    num = 0.0
    denom = 0.0
    if not pd.isna(offense):
        num += offense_weight * offense
        denom += offense_weight
    if not pd.isna(defense):
        num += defense_weight * defense
        denom += defense_weight
    return num / denom if denom > 0 else np.nan


def _compute_league_prior_totals(games: pd.DataFrame) -> pd.Series:
    totals = games.get("total_score")
    if totals is None:
        totals = games.get("home_score", pd.Series(dtype=float)) + games.get(
            "away_score", pd.Series(dtype=float)
        )
    prior = totals.shift(1).expanding().mean()
    return prior


def estimate_score_correlation(training_games: pd.DataFrame, shrink: float = 0.2) -> Tuple[float, float, float]:
    """Estimate home/away score std devs and correlation with shrinkage."""

    scores = training_games.dropna(subset=["home_score", "away_score"])
    if scores.empty:
        return 12.0, 12.0, 0.15
    std_home = float(scores["home_score"].std(ddof=0) or 12.0)
    std_away = float(scores["away_score"].std(ddof=0) or 12.0)
    corr = float(scores[["home_score", "away_score"]].corr().iloc[0, 1])
    corr = (1 - shrink) * corr
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
        [[std_home**2, rho * std_home * std_away], [rho * std_home * std_away, std_away**2]]
    )
    samples = rng.multivariate_normal(
        mean=[mean_home, mean_away], cov=cov, size=sample_size, check_valid="ignore"
    )
    totals = samples.sum(axis=1)
    return totals


def predict_totals(
    games_with_features: pd.DataFrame,
    config: TotalsConfig,
    corr_stats: Tuple[float, float, float],
    rng: np.random.Generator,
) -> pd.DataFrame:
    df = games_with_features.copy()
    std_home, std_away, rho = corr_stats

    # Base point predictions
    df["pred_home_points"] = [
        _blend_points(h_off, a_def, config.offense_weight, config.defense_weight)
        for h_off, a_def in zip(df.get("home_offense_qpa"), df.get("away_defense_qpa"))
    ]
    df["pred_away_points"] = [
        _blend_points(a_off, h_def, config.offense_weight, config.defense_weight)
        for a_off, h_def in zip(df.get("away_offense_qpa"), df.get("home_defense_qpa"))
    ]

    if "home_pace_qpa" in df.columns and "away_pace_qpa" in df.columns:
        pace_component = (df["home_pace_qpa"] + df["away_pace_qpa"]) * 0.5
        df["pred_home_points"] = df["pred_home_points"] + config.pace_weight * pace_component
        df["pred_away_points"] = df["pred_away_points"] + config.pace_weight * pace_component

    df["pred_total_mean"] = df["pred_home_points"] + df["pred_away_points"]

    totals = []
    entropies = []
    medians = []
    p10s = []
    p90s = []
    for mean_home, mean_away, prior, vegas in zip(
        df["pred_home_points"],
        df["pred_away_points"],
        df.get("league_prior_total", pd.Series(np.nan, index=df.index)),
        df.get("total_points", pd.Series(np.nan, index=df.index)),
    ):
        if pd.isna(mean_home) or pd.isna(mean_away):
            totals.append(np.nan)
            entropies.append(np.nan)
            medians.append(np.nan)
            p10s.append(np.nan)
            p90s.append(np.nan)
            continue
        samples = sample_correlated_totals(
            mean_home, mean_away, std_home, std_away, rho, config.sample_size, rng
        )
        sample_mean = float(np.mean(samples))
        sample_std = float(np.std(samples))
        entropy = 0.5 * math.log(2 * math.pi * math.e * (sample_std**2 + 1e-6))

        if not pd.isna(prior):
            confidence = 1.0 / (1.0 + config.entropy_shrink * max(entropy, 0.0))
            sample_mean = confidence * sample_mean + (1 - confidence) * float(prior)
        if not pd.isna(vegas):
            vegas_weight = config.vegas_weight * (1.0 - 1.0 / (1.0 + math.exp(-entropy)))
            sample_mean = vegas_weight * float(vegas) + (1 - vegas_weight) * sample_mean

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


def load_games_with_features(
    start: Optional[str] = None,
    end: Optional[str] = None,
    with_odds: bool = False,
    config: Optional[TotalsConfig] = None,
    project_root: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load games, team boxes, and attach decoherence features.

    Returns (games, team_features, games_with_features).
    """

    if project_root is None:
        project_root = get_project_root(Path(__file__).resolve())

    games = load_eoin_games(project_root)
    team_boxes = load_eoin_team_boxes(project_root)

    if start is not None:
        games = games.loc[pd.to_datetime(games.get("game_date", games.get("game_datetime"))) >= pd.to_datetime(start)]
        team_boxes = team_boxes.loc[
            pd.to_datetime(team_boxes.get("game_date", team_boxes.get("game_datetime")))
            >= pd.to_datetime(start)
        ]
    if end is not None:
        games = games.loc[pd.to_datetime(games.get("game_date", games.get("game_datetime"))) <= pd.to_datetime(end)]
        team_boxes = team_boxes.loc[
            pd.to_datetime(team_boxes.get("game_date", team_boxes.get("game_datetime")))
            <= pd.to_datetime(end)
        ]

    if with_odds:
        odds_path = project_root / "data" / "raw" / "nba" / "odds_long" / "nba_2008-2025.csv"
        odds_df = load_long_odds(odds_path)
        games, _ = attach_odds_to_games(games, odds_df)

    if config is None:
        config = TotalsConfig()

    team_state, games_enriched = enrich_games_with_config(games, team_boxes, config)
    return games, team_state, games_enriched


def enrich_games_with_config(
    games: pd.DataFrame, team_boxes: pd.DataFrame, config: TotalsConfig
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    team_state = build_team_decoherence_features(
        team_boxes, config.tau_offense, config.tau_defense, config.tau_pace
    )
    games_enriched = _merge_team_features(games, team_state)

    if "total_score" not in games_enriched.columns:
        if {"home_score", "away_score"}.issubset(games_enriched.columns):
            games_enriched["total_score"] = (
                games_enriched["home_score"] + games_enriched["away_score"]
            )

    games_enriched["league_prior_total"] = _compute_league_prior_totals(games_enriched)
    return team_state, games_enriched


def evaluate_predictions(df: pd.DataFrame) -> Dict[str, float]:
    truth = df.get("total_score")
    pred = df.get("pred_total_qpa")
    if truth is None or pred is None:
        return {"mae": float("nan"), "bias": float("nan")}
    mae = float((truth - pred).abs().mean())
    bias = float((pred - truth).mean())
    return {"mae": mae, "bias": bias}


def save_config(config: TotalsConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)


def load_config(path: Path) -> TotalsConfig:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return TotalsConfig.from_dict(data)


def smoketest_small_sample(project_root: Optional[Path] = None) -> Dict[str, float]:
    """Tiny end-to-end run to validate schema wiring without heavy data."""

    if project_root is None:
        project_root = get_project_root(Path(__file__).resolve())

    games = load_eoin_games(project_root).head(50)
    team_boxes = load_eoin_team_boxes(project_root).head(200)
    config = TotalsConfig(sample_size=64)
    team_state = build_team_decoherence_features(
        team_boxes, config.tau_offense, config.tau_defense, config.tau_pace
    )
    merged = _merge_team_features(games, team_state)
    merged["total_score"] = merged.get("home_score", 0) + merged.get("away_score", 0)
    merged["league_prior_total"] = _compute_league_prior_totals(merged)
    corr_stats = estimate_score_correlation(merged, config.corr_shrink)
    rng = np.random.default_rng(123)
    pred_df = predict_totals(merged, config, corr_stats, rng)
    return evaluate_predictions(pred_df)
