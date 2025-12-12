"""
QEPC NBA: Team scoring & stat expectations from Eoin data.

Takes:
  - team_stats (off_ppg, def_ppg, reb_pg, ast_pg per team)
  - matchup rows (home/away team_id + strengths)

Outputs:
  - per-game expected points (exp_home_pts, exp_away_pts)
  - per-game expected rebounds (exp_home_reb, exp_away_reb) if available
  - per-game expected assists (exp_home_ast, exp_away_ast) if available
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from qepc.utils.paths import get_project_root

from .eoin_team_stats import build_team_stats_from_eoin


@dataclass
class ScoringConfig:
    """
    Config for team stat expectations.

    The first three fields control points (kept compatible with earlier use).
    The new fields control how much to adjust rebounds/assists.
    """
    home_court_advantage: float = 1.5
    weight_off_edge: float = 0.7
    weight_def_edge: float = 0.7

    # Rebounds/assists tuning (smaller because they tend to be less swingy)
    weight_reb_edge: float = 0.5
    weight_ast_edge: float = 0.5
    home_reb_advantage: float = 0.3
    home_ast_advantage: float = 0.3


def _compute_league_baselines(team_stats: pd.DataFrame) -> dict:
    """
    Compute league-wide baselines from team_stats.
    """
    total_pts = team_stats["pts_for"].sum()
    total_games = team_stats["games_played"].sum()
    league_ppg = total_pts / total_games

    league_off_ppg = team_stats["off_ppg"].mean()
    league_def_ppg = team_stats["def_ppg"].mean()

    baselines = {
        "league_ppg": league_ppg,
        "league_off_ppg": league_off_ppg,
        "league_def_ppg": league_def_ppg,
    }

    if "reb_pg" in team_stats.columns:
        baselines["league_reb_pg"] = team_stats["reb_pg"].mean()
    else:
        baselines["league_reb_pg"] = None

    if "ast_pg" in team_stats.columns:
        baselines["league_ast_pg"] = team_stats["ast_pg"].mean()
    else:
        baselines["league_ast_pg"] = None

    return baselines


def attach_scoring_predictions(
    matchups: pd.DataFrame,
    team_stats: Optional[pd.DataFrame] = None,
    config: Optional[ScoringConfig] = None,
    project_root: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Attach expected points, rebounds, and assists for home and away teams
    to a matchup DataFrame.

    Expects matchups to have:
        - home_team_id
        - away_team_id

    Expects team_stats to have:
        - team_id
        - off_ppg
        - def_ppg
        - reb_pg (optional)
        - ast_pg (optional)

    Adds columns:
        - home_off_ppg, home_def_ppg, away_off_ppg, away_def_ppg
        - exp_home_pts, exp_away_pts
        - home_reb_pg, away_reb_pg, exp_home_reb, exp_away_reb (if reb_pg present)
        - home_ast_pg, away_ast_pg, exp_home_ast, exp_away_ast (if ast_pg present)
    """
    if config is None:
        config = ScoringConfig()

    if team_stats is None:
        team_stats = build_team_stats_from_eoin(project_root=project_root)

    df = matchups.copy()

    # Prepare a slim team_stats frame for joining
    ts_cols = ["team_id", "off_ppg", "def_ppg"]
    if "reb_pg" in team_stats.columns:
        ts_cols.append("reb_pg")
    if "ast_pg" in team_stats.columns:
        ts_cols.append("ast_pg")

    ts_core = team_stats[ts_cols].copy()

    # Join home stats
    df = df.merge(
        ts_core.add_prefix("home_"),
        left_on="home_team_id",
        right_on="home_team_id",
        how="left",
    )

    # Join away stats
    ts_core_away = ts_core.add_prefix("away_")
    df = df.merge(
        ts_core_away,
        left_on="away_team_id",
        right_on="away_team_id",
        how="left",
    )

    # League baselines
    baselines = _compute_league_baselines(team_stats)
    league_ppg = baselines["league_ppg"]
    league_off_ppg = baselines["league_off_ppg"]
    league_def_ppg = baselines["league_def_ppg"]
    league_reb_pg = baselines.get("league_reb_pg")
    league_ast_pg = baselines.get("league_ast_pg")

    # Offensive edges (team vs league average)
    df["home_off_edge"] = df["home_off_ppg"] - league_off_ppg
    df["away_off_edge"] = df["away_off_ppg"] - league_off_ppg

    # Defensive edges: lower def_ppg is better, so invert relative to league
    df["home_def_edge"] = league_def_ppg - df["home_def_ppg"]
    df["away_def_edge"] = league_def_ppg - df["away_def_ppg"]

    # Expected points for home/away based on edges
    w_off = config.weight_off_edge
    w_def = config.weight_def_edge
    hca = config.home_court_advantage

    df["exp_home_pts"] = (
        league_ppg
        + w_off * df["home_off_edge"]
        + w_def * df["away_def_edge"]
        + hca
    )

    df["exp_away_pts"] = (
        league_ppg
        + w_off * df["away_off_edge"]
        + w_def * df["home_def_edge"]
    )

    # Expected rebounds, if we have reb_pg
    if league_reb_pg is not None and "home_reb_pg" in df.columns:
        w_reb = config.weight_reb_edge
        h_reb = config.home_reb_advantage

        df["home_reb_edge"] = df["home_reb_pg"] - league_reb_pg
        df["away_reb_edge"] = df["away_reb_pg"] - league_reb_pg

        df["exp_home_reb"] = league_reb_pg + w_reb * df["home_reb_edge"] + h_reb
        df["exp_away_reb"] = league_reb_pg + w_reb * df["away_reb_edge"]

    # Expected assists, if we have ast_pg
    if league_ast_pg is not None and "home_ast_pg" in df.columns:
        w_ast = config.weight_ast_edge
        h_ast = config.home_ast_advantage

        df["home_ast_edge"] = df["home_ast_pg"] - league_ast_pg
        df["away_ast_edge"] = df["away_ast_pg"] - league_ast_pg

        df["exp_home_ast"] = league_ast_pg + w_ast * df["home_ast_edge"] + h_ast
        df["exp_away_ast"] = league_ast_pg + w_ast * df["away_ast_edge"]

    return df
