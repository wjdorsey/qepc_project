"""
QEPC NBA – Live Injury Impact Analysis

Beginner-friendly module that:
- Builds a per-player "impact" rating from historical player stats.
- Fetches live injury reports from qepc.sports.nba.data_source.load_live_injuries().
- Combines the two into an "ExpectedImpactLoss" for each injured player.

You can:
- Import this module from notebooks and call the functions, or
- Run it directly with:  python -m qepc.sports.nba.injury_impact_analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np

from qepc.autoload import paths
from qepc.sports.nba.data_source import load_live_injuries
from qepc.sports.nba.player_data import load_raw_player_data


# ---------------------------------------------------------------------------
# Helper: probability a listed player actually misses the game
# ---------------------------------------------------------------------------

# Priors for how often a status translates to "does not play".
_STATUS_PRIORS: Dict[str, float] = {
    "out": 0.98,
    "doubtful": 0.85,
    "questionable": 0.65,
    "gt d": 0.55,           # "GTD" / "Game-time decision"
    "game-time": 0.55,
    "probable": 0.35,
    "active": 0.05,
    "available": 0.05,
}


def estimate_prob_out(status: str) -> float:
    """
    Turn a free-text injury status into a probability the player will NOT play.
    """
    if not isinstance(status, str):
        return 0.5

    s = status.strip().lower()
    for key, prob in _STATUS_PRIORS.items():
        if key in s:
            return float(prob)

    # Completely unknown status → coin flip
    return 0.5


# ---------------------------------------------------------------------------
# Step 1: Build a per-player impact reference from historical data
# ---------------------------------------------------------------------------

def build_player_impact_table(
    player_df: Optional[pd.DataFrame] = None,
    min_games: int = 10,
) -> pd.DataFrame:
    """
    Build a reference table with one row per (PlayerName, Team) and a simple
    "ImpactRating" that approximates on-court value.

    ImpactRating is *not* a true RAPM/BPM model – it's a quick, interpretable
    score built from:
        - Per-36 PTS, REB, AST
        - Average plus-minus
        - Games played filter (to avoid tiny samples)
    """
    if player_df is None:
        player_df = load_raw_player_data()

    if player_df is None or player_df.empty:
        print("[InjuryImpact] No player data available – returning empty table.")
        return pd.DataFrame(
            columns=[
                "PlayerName",
                "Team",
                "GamesPlayed",
                "AvgMinutes",
                "AvgPTS",
                "AvgREB",
                "AvgAST",
                "AvgPlusMinus",
                "ImpactRating",
            ]
        )

    df = player_df.copy()

    # Basic cleaning – drop rows without a player name
    df = df.dropna(subset=["PlayerName"])

    # Ensure core numeric columns are numeric
    for col in ["PTS", "REB", "AST", "MIN", "PM"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Group by player & team
    grouped = df.groupby(["PlayerName", "Team"], dropna=False).agg(
        GamesPlayed=("gameId", "nunique") if "gameId" in df.columns else ("PlayerName", "size"),
        AvgMinutes=("MIN", "mean"),
        AvgPTS=("PTS", "mean"),
        AvgREB=("REB", "mean"),
        AvgAST=("AST", "mean"),
        AvgPlusMinus=("PM", "mean"),
    ).reset_index()

    # Filter out tiny samples
    grouped = grouped[grouped["GamesPlayed"] >= min_games].copy()

    # Per-36 stats (guard against divide-by-zero)
    eps = 1e-6
    grouped["Per36_PTS"] = grouped["AvgPTS"] * (36.0 / (grouped["AvgMinutes"] + eps))
    grouped["Per36_REB"] = grouped["AvgREB"] * (36.0 / (grouped["AvgMinutes"] + eps))
    grouped["Per36_AST"] = grouped["AvgAST"] * (36.0 / (grouped["AvgMinutes"] + eps))

    # Simple, interpretable impact metric
    grouped["ImpactRating"] = (
        grouped["Per36_PTS"] * 1.0
        + grouped["Per36_REB"] * 0.7
        + grouped["Per36_AST"] * 0.7
        + grouped["AvgPlusMinus"].fillna(0.0)
    )

    grouped = grouped.sort_values("ImpactRating", ascending=False).reset_index(drop=True)

    print(f"[InjuryImpact] Built impact table for {len(grouped)} players.")
    return grouped


# ---------------------------------------------------------------------------
# Step 2: Merge live injuries with impact reference
# ---------------------------------------------------------------------------

def _ensure_injury_columns(inj_df: pd.DataFrame) -> pd.DataFrame:
    """
    Make sure the live injury DataFrame has:
        - PlayerName
        - Team
        - Status
    """
    df = inj_df.copy()

    # PlayerName
    if "PlayerName" not in df.columns:
        for cand in ["player_name", "player", "name"]:
            if cand in df.columns:
                df["PlayerName"] = df[cand]
                break

    if "PlayerName" not in df.columns:
        raise ValueError(
            "[InjuryImpact] Injury data is missing a PlayerName column and no obvious "
            "fallback (player_name / player / name) was found."
        )

    # Team
    if "Team" not in df.columns:
        for cand in ["team", "team_name", "teamTricode", "team_code"]:
            if cand in df.columns:
                df["Team"] = df[cand]
                break
        if "Team" not in df.columns:
            df["Team"] = ""

    # Status
    if "Status" not in df.columns:
        for cand in ["status", "injury_status", "note"]:
            if cand in df.columns:
                df["Status"] = df[cand]
                break
        if "Status" not in df.columns:
            df["Status"] = "Unknown"

    return df


def _merge_injuries_with_impact(
    injuries_df: pd.DataFrame,
    impact_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Inner helper: merge a clean injuries table with the player impact table.
    """
    if impact_table is None or impact_table.empty:
        print("[InjuryImpact] Impact table is empty – returning injuries as-is.")
        return injuries_df.copy()

    inj = _ensure_injury_columns(injuries_df)

    merged = inj.merge(
        impact_table,
        on=["PlayerName", "Team"],
        how="left",
        suffixes=("", "_impact"),
    )

    # Fallback second pass merge on PlayerName only
    missing_mask = merged["ImpactRating"].isna()
    if missing_mask.any():
        fallback = inj.merge(
            impact_table[["PlayerName", "ImpactRating"]],
            on="PlayerName",
            how="left",
            suffixes=("", "_fallback"),
        )
        merged.loc[missing_mask, "ImpactRating"] = fallback.loc[missing_mask, "ImpactRating"]

    merged["ProbOut"] = merged["Status"].astype(str).apply(estimate_prob_out)
    merged["ExpectedImpactLoss"] = merged["ProbOut"] * merged["ImpactRating"].fillna(0.0)

    merged = merged.sort_values("ExpectedImpactLoss", ascending=False).reset_index(drop=True)
    return merged


def generate_injury_overrides(
    impact_reference: Optional[pd.DataFrame] = None,
    injuries_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    High-level helper: build a full "injury overrides" table.
    """
    if impact_reference is None:
        impact_reference = build_player_impact_table()

    if injuries_df is None:
        injuries_df = load_live_injuries()

    if injuries_df is None or injuries_df.empty:
        print("[InjuryImpact] No live injuries found – returning empty table.")
        return pd.DataFrame()

    merged = _merge_injuries_with_impact(injuries_df, impact_reference)
    print(f"[InjuryImpact] Generated injury overrides for {len(merged)} players.")
    return merged


def merge_with_live_injuries(
    impact_reference: pd.DataFrame,
) -> pd.DataFrame:
    """
    Backwards-compatible wrapper name.
    """
    return generate_injury_overrides(impact_reference=impact_reference, injuries_df=None)


# ---------------------------------------------------------------------------
# Step 3: Single-player helper for notebooks
# ---------------------------------------------------------------------------

def analyze_player_impact(
    player_name: str,
    team: Optional[str] = None,
    impact_reference: Optional[pd.DataFrame] = None,
    injury_overrides: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Return a small dictionary describing baseline impact + live injury overlay.
    """
    if impact_reference is None:
        impact_reference = build_player_impact_table()

    if injury_overrides is None:
        injury_overrides = generate_injury_overrides(impact_reference=impact_reference)

    name_key = player_name.strip().lower()

    def _norm_series(s: pd.Series) -> pd.Series:
        return s.astype(str).str.strip().str.lower()

    # Baseline impact
    base = impact_reference[_norm_series(impact_reference["PlayerName"]) == name_key]
    if team:
        base = base[_norm_series(base["Team"]).str.contains(team.strip().lower())]

    if base.empty:
        return {
            "PlayerName": player_name,
            "Team": team,
            "message": "Player not found in impact reference. Check spelling or data recency.",
        }

    base_row = base.iloc[0]

    # Injury overlay
    inj = injury_overrides[_norm_series(injury_overrides["PlayerName"]) == name_key]
    if team:
        inj = inj[_norm_series(inj["Team"]).str.contains(team.strip().lower())]

    if inj.empty:
        injury_status = "Not listed as injured"
        prob_out = 0.0
        expected_loss = 0.0
    else:
        inj_row = inj.iloc[0]
        injury_status = str(inj_row.get("Status", "Unknown"))
        prob_out = float(inj_row.get("ProbOut", 0.0))
        expected_loss = float(inj_row.get("ExpectedImpactLoss", 0.0))

    return {
        "PlayerName": str(base_row["PlayerName"]),
        "Team": str(base_row["Team"]),
        "GamesPlayed": int(base_row.get("GamesPlayed", 0)),
        "AvgMinutes": float(base_row.get("AvgMinutes", 0.0)),
        "AvgPTS": float(base_row.get("AvgPTS", 0.0)),
        "AvgREB": float(base_row.get("AvgREB", 0.0)),
        "AvgAST": float(base_row.get("AvgAST", 0.0)),
        "ImpactRating": float(base_row.get("ImpactRating", 0.0)),
        "InjuryStatus": injury_status,
        "ProbOut": prob_out,
        "ExpectedImpactLoss": expected_loss,
    }


# ---------------------------------------------------------------------------
# Script entry-point (safe to import)
# ---------------------------------------------------------------------------

def _demo() -> None:
    """
    Small demo so you can run this file directly while you're developing.
    """
    project_root: Path = paths.get_project_root()
    print(f"[InjuryImpact] Project root: {project_root}")

    impact_ref = build_player_impact_table()
    overrides = generate_injury_overrides(impact_reference=impact_ref)
    if not overrides.empty:
        print("\n[InjuryImpact] Top 10 injury overrides by ExpectedImpactLoss:")
        print(overrides.head(10))
    else:
        print("\n[InjuryImpact] No live injuries found.")

    try:
        example_player = "LeBron James"
        print(f"\n[InjuryImpact] Example lookup for {example_player}:")
        info = analyze_player_impact(
            player_name=example_player,
            team=None,
            impact_reference=impact_ref,
            injury_overrides=overrides,
        )
        print(info)
    except Exception as exc:
        print(f"[InjuryImpact] Could not compute example player impact: {exc}")

    manual_path = project_root / "data" / "Injury_Overrides.csv"
    if manual_path.exists() and not overrides.empty:
        try:
            manual_df = pd.read_csv(manual_path)
            if {"PlayerName", "Team"}.issubset(manual_df.columns):
                comp = manual_df.merge(
                    overrides[["PlayerName", "Team", "ExpectedImpactLoss"]],
                    on=["PlayerName", "Team"],
                    how="inner",
                    suffixes=("_manual", "_model"),
                )
                print("\n[InjuryImpact] Manual vs model (first 10 rows):")
                print(comp.head(10))
            else:
                print(
                    "[InjuryImpact] Manual Injury_Overrides.csv is missing "
                    "'PlayerName' and/or 'Team' columns – skipping comparison."
                )
        except Exception as exc:
            print(f"[InjuryImpact] Error while reading Injury_Overrides.csv: {exc}")
    else:
        print("[InjuryImpact] No Injury_Overrides.csv found – skipping manual comparison.")


if __name__ == "__main__":
    _demo()
