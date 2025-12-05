# qepc/brain/scripts.py

from __future__ import annotations

import pandas as pd


def label_game_scripts_by_total_points(
    games_df: pd.DataFrame,
    low_quantile: float = 0.25,
    high_quantile: float = 0.75,
) -> pd.DataFrame:
    """
    Label games as GRIND / BALANCED / CHAOS based on TOTAL_POINTS.

    - GRIND    = total points <= low_quantile (e.g. bottom 25%)
    - CHAOS    = total points >= high_quantile (e.g. top 25%)
    - BALANCED = everything in between

    Input:
        games_df: must contain at least:
            ['GAME_ID', 'GAME_DATE', 'TOTAL_POINTS']
        low_quantile: float between 0 and 1 (e.g. 0.25)
        high_quantile: float between 0 and 1 (e.g. 0.75)

    Output:
        DataFrame with:
            ['GAME_ID', 'GAME_DATE', 'TOTAL_POINTS',
             'SCRIPT_LABEL', 'SCRIPT_INDEX',
             'TOTAL_Q', 'TOTAL_Q_LOW', 'TOTAL_Q_HIGH']
    """
    if "TOTAL_POINTS" not in games_df.columns:
        raise ValueError("games_df must contain 'TOTAL_POINTS'")

    df = games_df.copy()

    # Compute quantiles on total points
    q_low = df["TOTAL_POINTS"].quantile(low_quantile)
    q_high = df["TOTAL_POINTS"].quantile(high_quantile)

    # For reference / debugging, keep game-wise percentile too
    df["TOTAL_Q"] = df["TOTAL_POINTS"].rank(pct=True)

    def _label(total: float) -> str:
        if total <= q_low:
            return "GRIND"
        elif total >= q_high:
            return "CHAOS"
        else:
            return "BALANCED"

    df["SCRIPT_LABEL"] = df["TOTAL_POINTS"].apply(_label)

    # Numeric index if we want to use it in models:
    # GRIND=0, BALANCED=1, CHAOS=2
    label_to_idx = {"GRIND": 0, "BALANCED": 1, "CHAOS": 2}
    df["SCRIPT_INDEX"] = df["SCRIPT_LABEL"].map(label_to_idx)

    # Attach thresholds for debugging
    df["TOTAL_Q_LOW"] = q_low
    df["TOTAL_Q_HIGH"] = q_high

    keep_cols = [
        "GAME_ID",
        "GAME_DATE",
        "TOTAL_POINTS",
        "TOTAL_Q",
        "SCRIPT_LABEL",
        "SCRIPT_INDEX",
        "TOTAL_Q_LOW",
        "TOTAL_Q_HIGH",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]
