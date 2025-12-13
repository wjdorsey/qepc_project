import pandas as pd

from qepc.nba.league_field import compute_env_drift
from qepc.nba.qpa_totals import collapse_total_with_odds


def test_env_drift_shift():
    df = pd.DataFrame({
        "game_date": pd.date_range("2023-01-01", periods=5, freq="D"),
        "total_actual": [210, 215, 220, 205, 200],
        "total_pred": [200, 205, 210, 215, 220],
    })
    drift = compute_env_drift(df, W_fast=2, W_slow=3, weights=(0.5, 0.5))
    assert pd.isna(drift.iloc[0]) or drift.iloc[0] == 0
    assert drift.iloc[1] != (df.loc[1, "total_actual"] - df.loc[1, "total_pred"])


def test_collapse_total_with_odds_shift_safe():
    df = pd.DataFrame({
        "game_date": pd.date_range("2023-01-01", periods=4, freq="D"),
        "total_actual": [200, 210, 205, 215],
        "total_pred_env": [198, 208, 207, 210],
        "total_points": [202, 212, 203, 220],
    })
    posterior = collapse_total_with_odds(df)
    assert len(posterior) == len(df)
    # first weight falls back to neutral 0.5 because of shift/NaN ema
    assert abs(posterior.iloc[0] - ((df.loc[0, "total_pred_env"] + df.loc[0, "total_points"]) / 2)) < 1e-6
