import pandas as pd


def build_rolling_player_state_vectors(
    boxscores_df: pd.DataFrame,
    games_df: pd.DataFrame,
    window: int = 10,
) -> pd.DataFrame:
    """
    Build game-by-game rolling state vectors for each player.

    Inputs:
      - boxscores_df: traditional boxscores with at least some kind of
        player/team/game id columns. We try to normalize:
           gameId  -> GAME_ID  (if GAME_ID not already present)
           teamId  -> TEAM_ID
           personId/person_id/PERSON_ID -> PLAYER_ID

        And we expect per-game stats like:
           minutes (as 'MM:SS' strings), points, fieldGoalsMade, etc.
      - games_df: table with ['GAME_ID', 'GAME_DATE'] (datetime)
      - window: rolling window size in games

    Output:
      - DataFrame with one row per (PLAYER_ID, GAME_ID)
      - Includes rolling means / stds for key stats.
    """
    # --- 0) Normalize ID columns defensively ---
    bs = boxscores_df.copy()

    # Normalize GAME_ID
    if "GAME_ID" in bs.columns and "gameId" in bs.columns:
        bs = bs.drop(columns=["gameId"])
    elif "GAME_ID" not in bs.columns and "gameId" in bs.columns:
        bs = bs.rename(columns={"gameId": "GAME_ID"})

    # Normalize TEAM_ID
    if "TEAM_ID" in bs.columns and "teamId" in bs.columns:
        bs = bs.drop(columns=["teamId"])
    elif "TEAM_ID" not in bs.columns and "teamId" in bs.columns:
        bs = bs.rename(columns={"teamId": "TEAM_ID"})

    # Normalize PLAYER_ID from various possible names
    player_id_candidates = ["PLAYER_ID", "personId", "PERSON_ID", "person_id"]
    if "PLAYER_ID" not in bs.columns:
        for cand in player_id_candidates:
            if cand in bs.columns:
                bs = bs.rename(columns={cand: "PLAYER_ID"})
                break

    # Final sanity check
    missing_ids = [name for name in ["GAME_ID", "PLAYER_ID"] if name not in bs.columns]
    if missing_ids:
        raise ValueError(
            f"boxscores_df is missing required id columns after normalization: {missing_ids}"
        )

    # --- 1) Merge boxscores with game dates ---
    g = games_df[["GAME_ID", "GAME_DATE"]].copy()
    g["GAME_DATE"] = pd.to_datetime(g["GAME_DATE"])

    merged = bs.merge(g, on="GAME_ID", how="left")

    # --- 2) Sort by player, then game date ---
    merged = merged.sort_values(["PLAYER_ID", "GAME_DATE"])

    # --- 3) Define the numeric stat columns for rolling features ---
    stat_cols = [
        "minutes",
        "points",
        "fieldGoalsMade",
        "fieldGoalsAttempted",
        "threePointersMade",
        "threePointersAttempted",
        "freeThrowsMade",
        "freeThrowsAttempted",
        "reboundsOffensive",
        "reboundsDefensive",
        "reboundsTotal",
        "assists",
        "steals",
        "blocks",
        "turnovers",
        "foulsPersonal",
    ]
    # Only keep columns that actually exist in this DataFrame
    stat_cols = [c for c in stat_cols if c in merged.columns]

    # --- 3a) Convert 'minutes' from 'MM:SS' strings to float minutes ---
    def _minutes_to_float(x):
        if x is None or pd.isna(x):
            return 0.0
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x)
        parts = s.split(":")
        try:
            if len(parts) == 2:
                m, sec = parts
                return int(m) + int(sec) / 60.0
            elif len(parts) == 3:
                h, m, sec = parts
                return 60 * int(h) + int(m) + int(sec) / 60.0
        except Exception:
            return 0.0
        return 0.0

    if "minutes" in stat_cols:
        merged["minutes"] = merged["minutes"].apply(_minutes_to_float)

    # --- 3b) Ensure all stat columns are numeric ---
    for col in stat_cols:
        if col == "minutes":
            # already coerced above
            continue
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    # --- 4) Group by player and compute rolling stats ---
    group = merged.groupby("PLAYER_ID", group_keys=False)

    def _add_rolling(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in stat_cols:
            roll = df[col].rolling(window=window, min_periods=1)
            df[f"roll_mean_{col}_{window}"] = roll.mean()
            df[f"roll_std_{col}_{window}"] = roll.std().fillna(0.0)
        return df

    rolled = group.apply(_add_rolling)

    # --- 5) Select the ID columns + rolling features ---
    id_cols = ["PLAYER_ID", "TEAM_ID", "GAME_ID", "GAME_DATE"]
    id_cols = [c for c in id_cols if c in rolled.columns]

    roll_cols = [c for c in rolled.columns if c.startswith("roll_")]

    out = rolled[id_cols + roll_cols].reset_index(drop=True)
    return out
