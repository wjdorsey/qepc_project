# qepc/brain/games_loader.py

from typing import Literal
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

SeasonString = str  # e.g. "2023-24"

def fetch_league_games(season: SeasonString,
                       season_type: Literal["Regular Season", "Playoffs"] = "Regular Season",
                       verbose: bool = True) -> pd.DataFrame:
    """
    Fetch all NBA games for a given season using LeagueGameLog.

    Returns one row per *team-game* (so each game appears twice: once per team).
    We'll later pivot this into a single row per game.
    """
    if verbose:
        print(f"[games_loader] Fetching LeagueGameLog for season={season}, season_type={season_type}...")

    lg = leaguegamelog.LeagueGameLog(
        season=season,
        season_type_all_star=season_type,
    )
    df = lg.get_data_frames()[0]

    if verbose:
        print(f"[games_loader] Retrieved {len(df)} team-games.")

    return df


def build_games_table(team_games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level game logs to a single row per game with
    home/away teams and final score.
    """
    # Basic assumption: MATCHUP contains '@' for away, 'vs.' for home.
    away = team_games_df[team_games_df["MATCHUP"].str.contains("@")].copy()
    home = team_games_df[team_games_df["MATCHUP"].str.contains("vs.")].copy()

    # Rename columns to indicate home/away
    suffix_map = {
        "TEAM_ID": "TEAM_ID",
        "TEAM_NAME": "TEAM_NAME",
        "TEAM_ABBREVIATION": "TEAM_ABBREVIATION",
        "PTS": "PTS",
        "PLUS_MINUS": "PLUS_MINUS",
    }

    away = away.rename(columns={k: f"AWAY_{v}" for k, v in suffix_map.items()})
    home = home.rename(columns={k: f"HOME_{v}" for k, v in suffix_map.items()})

    merged = pd.merge(
        home,
        away,
        on="GAME_ID",
        how="inner",
        suffixes=("", "_AWAY"),
    )

    # Simplify columns: keep date, season, matchup, etc. from the home row
    keep_cols = [
        "GAME_ID", "GAME_DATE", "SEASON_ID",
        "HOME_TEAM_ID", "HOME_TEAM_NAME", "HOME_TEAM_ABBREVIATION", "HOME_PTS",
        "AWAY_TEAM_ID", "AWAY_TEAM_NAME", "AWAY_TEAM_ABBREVIATION", "AWAY_PTS",
        "WL",  # result from home perspective
    ]
    games = merged[keep_cols].copy()
    games.rename(columns={"WL": "HOME_RESULT"}, inplace=True)

    # Cast date to datetime
    games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"])

    # You can add margin, total, etc.
    games["MARGIN"] = games["HOME_PTS"] - games["AWAY_PTS"]
    games["TOTAL_POINTS"] = games["HOME_PTS"] + games["AWAY_PTS"]

    return games
