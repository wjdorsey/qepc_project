# qepc/brain/teams_loader.py

from typing import Literal
import pandas as pd

from nba_api.stats.static import teams as static_teams
from nba_api.stats.endpoints import teamgamelog, leaguedashteamstats

SeasonString = str  # e.g. "2023-24"


def fetch_teams_static() -> pd.DataFrame:
    """
    Get a static list of NBA teams (current + some historical).

    Columns typically include:
        id, full_name, abbreviation, nickname, city, state, year_founded
    """
    teams_list = static_teams.get_teams()
    df = pd.DataFrame(teams_list)
    return df


def fetch_team_game_logs(
    team_id: int,
    season: SeasonString,
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season",
) -> pd.DataFrame:
    """
    Game logs for a single team in a given season.

    Each row = one game played by that team.
    """
    gl = teamgamelog.TeamGameLog(
        team_id=team_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = gl.get_data_frames()[0]
    df["TEAM_ID"] = team_id
    return df


def fetch_league_team_season_stats(
    season: SeasonString,
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season",
    measure_type: str = "Base",
) -> pd.DataFrame:
    """
    League-wide team season stats via LeagueDashTeamStats.

    measure_type can be 'Base', 'Advanced', 'Four Factors', 'Scoring', etc.
    (These are the same options you see on NBA.com/stats team tables.)
    """
    lds = leaguedashteamstats.LeagueDashTeamStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense=measure_type,
    )
    df = lds.get_data_frames()[0]
    return df
