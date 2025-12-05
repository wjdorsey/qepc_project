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


def build_team_script_lambdas(
    team_games_df: pd.DataFrame,
    games_df: pd.DataFrame,
    scripts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build team+script-level scoring lambdas.

    Inputs:
        team_games_df: team-game logs from fetch_league_games(season)
            Must contain:
                - 'GAME_ID'
                - 'TEAM_ID'
                - a points column, usually 'PTS'

        games_df: game-level table from build_games_table(...)
            Must contain:
                - 'GAME_ID'
                - 'HOME_TEAM_ID'
                - 'AWAY_TEAM_ID'

        scripts_df: script labels table from label_game_scripts_by_total_points(...)
            Must contain:
                - 'GAME_ID'
                - 'SCRIPT_LABEL'

    Output:
        DataFrame with columns:
            - TEAM_ID
            - TEAM_ROLE  ('HOME' or 'AWAY')
            - SCRIPT_LABEL ('GRIND', 'BALANCED', 'CHAOS')
            - mean_team_pts
            - std_team_pts
            - count_games
    """
    # --- 1) Check columns in team_games_df ---

    needed_team_cols = ["GAME_ID", "TEAM_ID"]
    missing_team = [c for c in needed_team_cols if c not in team_games_df.columns]
    if missing_team:
        raise ValueError(f"team_games_df missing required columns: {missing_team}")

    # Figure out which column is points
    if "PTS" in team_games_df.columns:
        pts_col = "PTS"
    elif "TEAM_POINTS" in team_games_df.columns:
        pts_col = "TEAM_POINTS"
    else:
        raise ValueError(
            "team_games_df must contain a points column ('PTS' or 'TEAM_POINTS')."
        )

    df = team_games_df[["GAME_ID", "TEAM_ID", pts_col]].copy()
    df = df.rename(columns={pts_col: "TEAM_POINTS"})

    # --- 2) Attach home/away role from games_df ---

    needed_games_cols = ["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]
    missing_games = [c for c in needed_games_cols if c not in games_df.columns]
    if missing_games:
        raise ValueError(f"games_df missing required columns: {missing_games}")

    games_small = games_df[["GAME_ID", "HOME_TEAM_ID", "AWAY_TEAM_ID"]].copy()

    df = df.merge(games_small, on="GAME_ID", how="left")

    # Determine if the team is home or away
    df["TEAM_ROLE"] = "UNKNOWN"
    df.loc[df["TEAM_ID"] == df["HOME_TEAM_ID"], "TEAM_ROLE"] = "HOME"
    df.loc[df["TEAM_ID"] == df["AWAY_TEAM_ID"], "TEAM_ROLE"] = "AWAY"

    # Keep only rows where we could assign a role
    df = df[df["TEAM_ROLE"].isin(["HOME", "AWAY"])].copy()

    # --- 3) Attach script labels ---

    needed_script_cols = ["GAME_ID", "SCRIPT_LABEL"]
    missing_scripts = [c for c in needed_script_cols if c not in scripts_df.columns]
    if missing_scripts:
        raise ValueError(f"scripts_df missing required columns: {missing_scripts}")

    scripts_small = scripts_df[["GAME_ID", "SCRIPT_LABEL"]].copy()

    df = df.merge(scripts_small, on="GAME_ID", how="left")

    # --- 4) Group by TEAM_ID, TEAM_ROLE, SCRIPT_LABEL and aggregate ---

    agg = (
        df.groupby(["TEAM_ID", "TEAM_ROLE", "SCRIPT_LABEL"])
        .agg(
            mean_team_pts=("TEAM_POINTS", "mean"),
            std_team_pts=("TEAM_POINTS", "std"),
            count_games=("GAME_ID", "nunique"),
        )
        .reset_index()
    )

    return agg
    
