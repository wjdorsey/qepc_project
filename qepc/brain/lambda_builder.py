# qepc/brain/lambda_builder.py

from __future__ import annotations

import pandas as pd


def build_script_level_lambdas(
    games_df: pd.DataFrame,
    scripts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute script-level scoring 'lambdas' (means) for each script type.

    We join games_df with scripts_df on GAME_ID and then, for each SCRIPT_LABEL,
    compute:

        - mean_total_pts (always)
        - std_total_pts
        - count_games

    If HOME_TEAM_SCORE / AWAY_TEAM_SCORE exist in games_df, we also compute:
        - mean_home_pts
        - mean_away_pts

    This is QEPC λ-builder v0: one global λ per script, based on TOTAL_POINTS.
    """
    # We only *require* GAME_ID and TOTAL_POINTS
    needed_cols_games = ["GAME_ID", "TOTAL_POINTS"]
    missing_games = [c for c in needed_cols_games if c not in games_df.columns]
    if missing_games:
        raise ValueError(f"games_df missing required columns: {missing_games}")

    # Optional columns
    has_home = "HOME_TEAM_SCORE" in games_df.columns
    has_away = "AWAY_TEAM_SCORE" in games_df.columns

    # Expect labels in scripts_df
    needed_cols_scripts = ["GAME_ID", "SCRIPT_LABEL"]
    missing_scripts = [c for c in needed_cols_scripts if c not in scripts_df.columns]
    if missing_scripts:
        raise ValueError(f"scripts_df missing required columns: {missing_scripts}")

    # Join
    merged = games_df.merge(
        scripts_df[["GAME_ID", "SCRIPT_LABEL"]],
        on="GAME_ID",
        how="inner",
    )

    # Build aggregation dict dynamically
    agg_dict: dict[str, tuple[str, str]] = {
        "mean_total_pts": ("TOTAL_POINTS", "mean"),
        "std_total_pts": ("TOTAL_POINTS", "std"),
        "count_games": ("GAME_ID", "nunique"),
    }

    if has_home:
        agg_dict["mean_home_pts"] = ("HOME_TEAM_SCORE", "mean")
    if has_away:
        agg_dict["mean_away_pts"] = ("AWAY_TEAM_SCORE", "mean")

    agg = (
        merged.groupby("SCRIPT_LABEL")
        .agg(**agg_dict)
        .reset_index()
    )

    return agg


def expected_total_from_script_mix(
    script_lambdas: pd.DataFrame,
    p_grind: float,
    p_balanced: float,
    p_chaos: float,
) -> float:
    """
    Given script-level lambdas and script probabilities for a single game,
    compute the script-mixture expected total:

        E[Total] = sum_s P(s) * lambda_total(s)

    script_lambdas: DataFrame from build_script_level_lambdas, with:
        - SCRIPT_LABEL
        - mean_total_pts
    """
    # Map labels to mean totals
    label_to_mean = (
        script_lambdas.set_index("SCRIPT_LABEL")["mean_total_pts"].to_dict()
    )

    # Defensive: default to 0 if any labels missing
    lambda_grind = label_to_mean.get("GRIND", 0.0)
    lambda_bal   = label_to_mean.get("BALANCED", 0.0)
    lambda_chaos = label_to_mean.get("CHAOS", 0.0)

    expected = (
        p_grind   * lambda_grind
        + p_balanced * lambda_bal
        + p_chaos  * lambda_chaos
    )

    return float(expected)
