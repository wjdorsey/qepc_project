# qepc/brain/boxscores_loader.py

from typing import Iterable, List
import time
import pandas as pd
from nba_api.stats.endpoints import boxscoretraditionalv3, boxscoreadvancedv3


def _concat_player_like_frames(dfs: List[pd.DataFrame], game_id: str, kind: str) -> pd.DataFrame:
    """
    Try to pick out the player-level tables from a list of dataframes.
    If we can't confidently find player tables, fall back to concatenating everything.

    This is defensive because the V3 endpoints don't always use 'PLAYER_ID';
    they often use 'person_id' or similar.
    """
    if not dfs:
        # Nothing came back at all
        return pd.DataFrame({"GAME_ID": [game_id], "WARNING": [f"no_{kind}_datasets"]})

    candidate_id_cols = ["PLAYER_ID", "PERSON_ID", "personId", "person_id", "PERSONID"]

    player_dfs = []
    for df in dfs:
        cols = set(map(str, df.columns))
        if any(c in cols for c in candidate_id_cols):
            player_dfs.append(df)

    if not player_dfs:
        # Fallback: just concat everything so we at least see the structure
        combined = pd.concat(dfs, ignore_index=True)
        combined["GAME_ID"] = game_id
        combined["NOTE"] = f"no_explicit_player_id_column_in_{kind}"
        return combined

    combined = pd.concat(player_dfs, ignore_index=True)
    combined["GAME_ID"] = game_id
    return combined


def fetch_boxscore_traditional(game_id: str, verbose: bool = False) -> pd.DataFrame:
    """
    Fetch the traditional box score (player-level) for a game using BoxScoreTraditionalV3.
    """
    if verbose:
        print(f"[boxscores_loader] Fetching TRADITIONAL boxscore for GAME_ID={game_id}...")

    bs = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    dfs = bs.get_data_frames()
    trad_players_df = _concat_player_like_frames(dfs, game_id, kind="traditional")
    return trad_players_df


def fetch_boxscore_advanced(game_id: str, verbose: bool = False) -> pd.DataFrame:
    """
    Fetch advanced box score metrics (ORtg, DRtg, etc.) for a game using BoxScoreAdvancedV3.
    """
    if verbose:
        print(f"[boxscores_loader] Fetching ADVANCED boxscore for GAME_ID={game_id}...")

    bs_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
    dfs = bs_adv.get_data_frames()
    adv_players_df = _concat_player_like_frames(dfs, game_id, kind="advanced")
    return adv_players_df


def fetch_boxscores_for_games(
    game_ids: Iterable[str],
    sleep_seconds: float = 0.6,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loop over many GAME_IDs and build:
      - traditional player boxscores
      - advanced player boxscores

    Returns (trad_df, adv_df).
    """
    trad_rows = []
    adv_rows = []

    for i, gid in enumerate(game_ids, start=1):
        if verbose:
            print(f"[boxscores_loader] ({i}) Fetching boxscores for GAME_ID={gid}...")

        try:
            trad = fetch_boxscore_traditional(gid, verbose=False)
            adv = fetch_boxscore_advanced(gid, verbose=False)
        except Exception as e:
            print(f"[boxscores_loader] ERROR for GAME_ID={gid}: {e}")
            continue

        trad_rows.append(trad)
        adv_rows.append(adv)

        time.sleep(sleep_seconds)  # respect rate limits

    trad_df = pd.concat(trad_rows, ignore_index=True) if trad_rows else pd.DataFrame()
    adv_df = pd.concat(adv_rows, ignore_index=True) if adv_rows else pd.DataFrame()
    return trad_df, adv_df
