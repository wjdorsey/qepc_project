# qepc/nba/backtest_pipeline.py

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from qepc.config import QEPCConfig, detect_project_root
from qepc.logging_utils import qstep
from qepc.nba.data_loaders import load_nba_team_logs
from qepc.core.strengths import compute_team_strengths, TeamStrengthsConfig
from qepc.core.lambda_engine import build_lambda_table
from qepc.core.simulator import simulate_games_multiverse
from qepc.core.calibration import fit_linear_calibration


def build_backtest_games(game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team logs (2 rows per game) into 1-row-per-game backtest set.
    Requires columns: gameId, gameDate, teamName, teamScore, home
    """
    games_rows = []
    for gid, group in game_logs.groupby("gameId"):
        home_rows = group[group["home"] == 1]
        away_rows = group[group["home"] == 0]
        if len(home_rows) == 0 or len(away_rows) == 0:
            continue
        home_row = home_rows.iloc[0]
        away_row = away_rows.iloc[0]

        home_name = str(home_row["teamName"])
        away_name = str(away_row["teamName"])

        games_rows.append(
            {
                "gameId": gid,
                "gameDate": home_row["gameDate"],
                "Home_Team": home_name,
                "Away_Team": away_name,
                "Home_Score": home_row["teamScore"],
                "Away_Score": away_row["teamScore"],
            }
        )
    return pd.DataFrame(games_rows)


def run_nba_backtest(
    lookback_mode: str = "all",  # "all", "years", or "days"
    lookback_years: int = 3,
    lookback_days: int = 60,
    strengths_cfg: Optional[TeamStrengthsConfig] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Full pipeline:
      - Load logs
      - Filter window
      - Build strengths
      - Build λ
      - Sim multiverse
      - Compute metrics & calibration (train/test split)
    Returns:
      results_df, metrics dict
    """
    root = detect_project_root()
    config = QEPCConfig.from_project_root(root)
    logs = load_nba_team_logs(config)

    # filter window
    latest = logs["gameDate"].max()
    earliest = logs["gameDate"].min()

    if lookback_mode == "all":
        start = earliest
    elif lookback_mode == "years":
        start = latest - pd.Timedelta(days=365 * lookback_years)
    elif lookback_mode == "days":
        start = latest - pd.Timedelta(days=lookback_days)
    else:
        raise ValueError(f"Unknown lookback_mode: {lookback_mode}")

    mask = (logs["gameDate"] >= start) & (logs["gameDate"] <= latest)
    window_logs = logs[mask].copy()

    qstep(
        f"Backtest window: {start.date()} to {latest.date()} "
        f"({len(window_logs)} team-rows)"
    )

    # build backtest games (one row per game)
    games_df = build_backtest_games(window_logs)
    games_df = games_df.sort_values("gameDate").reset_index(drop=True)
    qstep(f"Constructed {len(games_df)} games for backtest")

    # strengths from logs up to each game date (for now: global, cut at latest)
    strengths = compute_team_strengths(
        game_logs=logs,  # all logs
        config=strengths_cfg or TeamStrengthsConfig(),
        cutoff_date=latest,
    )

    # schedule for lambda
    schedule = games_df[["Home_Team", "Away_Team"]].rename(
        columns={"Home_Team": "Home_Team", "Away_Team": "Away_Team"}
    )
    lam_df = build_lambda_table(schedule, strengths)
    lam_df = lam_df.reset_index(drop=True)

    # align with games (may drop some if strengths missing)
    merged = pd.merge(
        games_df,
        lam_df,
        on=["Home_Team", "Away_Team"],
        how="inner",
    )
    qstep(f"Joined games with λ for {len(merged)} games")

    sim_df = simulate_games_multiverse(
        merged[["Home_Team", "Away_Team", "lambda_home", "lambda_away", "vol_home", "vol_away"]],
        num_universes=5000,
        seed=config.seed,
    )

    full = pd.merge(
        merged,
        sim_df[["Home_Team", "Away_Team", "Sim_Home_Score", "Sim_Away_Score", "Home_Win_Prob", "Total_Mean", "Spread_Mean"]],
        on=["Home_Team", "Away_Team"],
        how="inner",
    )

    full["Pred_Total"] = full["Total_Mean"]
    full["Pred_Spread"] = full["Spread_Mean"]
    full["Actual_Total"] = full["Home_Score"] + full["Away_Score"]
    full["Actual_Spread"] = full["Home_Score"] - full["Away_Score"]

    full["Winner_Correct"] = (
        (full["Home_Score"] > full["Away_Score"]) ==
        (full["Home_Win_Prob"] > 0.5)
    )

    full["Error_Total"] = (full["Pred_Total"] - full["Actual_Total"]).abs()
    full["Error_Spread"] = (full["Pred_Spread"] - full["Actual_Spread"]).abs()

    # Train/test split (60/40 by date)
    full_sorted = full.sort_values("gameDate").reset_index(drop=True)
    n = len(full_sorted)
    split = int(n * 0.6)
    train = full_sorted.iloc[:split].copy()
    test = full_sorted.iloc[split:].copy()

    # calibration on totals
    calib = fit_linear_calibration(train["Pred_Total"], train["Actual_Total"])
    full_sorted["Pred_Total_cal"] = calib.apply(full_sorted["Pred_Total"].values)
    full_sorted["Error_Total_cal"] = (full_sorted["Pred_Total_cal"] - full_sorted["Actual_Total"]).abs()

    test = full_sorted.iloc[split:].copy()

    metrics = {
        "n_games": len(full_sorted),
        "win_acc_total": float(full_sorted["Winner_Correct"].mean()),
        "mae_total_raw": float(full_sorted["Error_Total"].mean()),
        "mae_total_cal": float(full_sorted["Error_Total_cal"].mean()),
        "test_win_acc": float(test["Winner_Correct"].mean()),
        "test_mae_total_raw": float(test["Error_Total"].mean()),
        "test_mae_total_cal": float(test["Error_Total_cal"].mean()),
        "calibration_intercept": calib.intercept,
        "calibration_slope": calib.slope,
    }

    return full_sorted, metrics
