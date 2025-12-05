# qepc/brain/players_loader.py

from typing import Literal, Iterable
import pandas as pd

from nba_api.stats.static import players as static_players
from nba_api.stats.endpoints import playergamelog, leaguedashplayerstats

SeasonString = str  # e.g. "2023-24"


def fetch_players_static() -> pd.DataFrame:
    """
    Static table of NBA players from nba_api.stats.static.players.

    Columns typically include:
        id, full_name, first_name, last_name, is_active, etc.
    """
    players_list = static_players.get_players()
    df = pd.DataFrame(players_list)
    return df


def fetch_league_player_season_stats(
    season: SeasonString,
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season",
    measure_type: str = "Base",
) -> pd.DataFrame:
    """
    League-wide player season stats via LeagueDashPlayerStats.

    measure_type options on NBA.com/stats include:
        'Base', 'Advanced', 'Usage', 'Scoring', 'Misc', etc.

    We'll start with 'Base'; later you can experiment with others.
    """
    lds = leaguedashplayerstats.LeagueDashPlayerStats(
        season=season,
        season_type_all_star=season_type,
        measure_type_detailed_defense=measure_type,
    )
    df = lds.get_data_frames()[0]
    return df


def fetch_player_game_log(
    player_id: int,
    season: SeasonString,
    season_type: Literal["Regular Season", "Playoffs"] = "Regular Season",
) -> pd.DataFrame:
    """
    Game log for a single player in a given season.
    (We might build a bulk version later; for now this is a simple helper.)
    """
    gl = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=season,
        season_type_all_star=season_type,
    )
    df = gl.get_data_frames()[0]
    df["PLAYER_ID"] = player_id
    return df
