"""
QEPC Autoload Helper
====================

Convenience entry point for notebooks and scripts.

Responsibilities
---------------
- Ensure the QEPC project root is on sys.path.
- Provide a qepc_step() visual helper for notebooks.
- Re-export commonly-used QEPC functions under short, friendly names:

    load_games          -> qepc.sports.nba.sim.load_nba_schedule
    get_today_games     -> qepc.sports.nba.sim.get_today_games
    get_tomorrow_games  -> qepc.sports.nba.sim.get_tomorrow_games
    get_upcoming_games  -> qepc.sports.nba.sim.get_upcoming_games
    load_team_stats     -> qepc.sports.nba.strengths_v2.calculate_advanced_strengths
    load_raw_player_data
    compute_lambda
    run_qepc_simulation
    run_diagnostics
    run_daily_backtest
    run_season_backtest
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# -----------------------------------------------------------------------------
# 1. Path Setup
# -----------------------------------------------------------------------------

try:
    # Preferred: ask the internal paths helper for the project root
    from qepc.autoload.paths import get_project_root

    PROJECT_ROOT = get_project_root()
    if not isinstance(PROJECT_ROOT, Path):
        PROJECT_ROOT = Path(PROJECT_ROOT)
except Exception:
    # Fallback: walk up from this file
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Make sure project root is first on sys.path so local code wins
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# -----------------------------------------------------------------------------
# 2. Visual Step Helper (for notebooks)
# -----------------------------------------------------------------------------

try:
    from IPython.display import display, HTML
    _HAS_IPYTHON = True
except ImportError:  # non-notebook environments
    _HAS_IPYTHON = False
    display = None
    HTML = None


def qepc_step(msg: str) -> None:
    """
    Nice visual step marker for Jupyter notebooks.

    In a notebook:
        qepc_step("Loading games for today")

    In non-notebook environments this just prints a plain text line.
    """
    if _HAS_IPYTHON and display is not None and HTML is not None:
        display(HTML(
            f"<div style='font-family:monospace; color:#4ea3ff; "
            f"font-weight:bold; margin-top:10px;'>"
            f"⧉ QEPC: {msg}</div>"
        ))
    else:
        print(f"[QEPC] {msg}")


# -----------------------------------------------------------------------------
# 3. Core Function Imports & Proxies
# -----------------------------------------------------------------------------

# We try to import all core helpers. If anything fails, we fall back to
# no-op versions that print a helpful message instead of crashing.

try:
    # Schedule / games
    from qepc.sports.nba.sim import (
        load_nba_schedule,
        get_today_games,
        get_tomorrow_games,
        get_upcoming_games,
    )

    # Data & strengths
    from qepc.utils.data_cleaning import (
        load_team_stats as load_dummy_team_stats,  # kept for compatibility
    )
    from qepc.sports.nba.player_data import load_raw_player_data
    from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths

    # Core engine
    from qepc.core.lambda_engine import compute_lambda
    from qepc.core.simulator import run_qepc_simulation

    # Diagnostics
    from qepc.utils.diagnostics import run_system_check

    # Backtesting
    from qepc.backtest.backtest_engine import (
        run_daily_backtest,
        run_season_backtest,
    )

    # Friendly proxy names used in notebooks
    load_games = load_nba_schedule
    load_team_stats = calculate_advanced_strengths
    run_diagnostics = run_system_check

except ImportError as e:
    # If anything fails, expose dummy functions so notebooks don't crash
    print(f"[QEPC Autoload] ERROR: Failed to import core functions: {e}")

    def _missing(*args: Any, **kwargs: Any) -> None:
        print("[QEPC Autoload] Core function not available. "
              "Check that your project is installed and paths are correct.")

    # Schedule / games
    def get_today_games(show: str = "clean") -> None:  # type: ignore[no-redef]
        _missing()

    def get_tomorrow_games(show: str = "clean") -> None:  # type: ignore[no-redef]
        _missing()

    def get_upcoming_games(days: int = 7, show: str = "clean") -> None:  # type: ignore[no-redef]
        _missing()

    def load_nba_schedule(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    load_games = load_nba_schedule  # type: ignore[assignment]

    # Data & strengths
    def load_team_stats(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    def load_raw_player_data(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    calculate_advanced_strengths = load_team_stats  # type: ignore[assignment]

    # Core engine
    def compute_lambda(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    def run_qepc_simulation(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    # Diagnostics
    def run_diagnostics(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    # Backtesting
    def run_daily_backtest(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()

    def run_season_backtest(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        _missing()


# -----------------------------------------------------------------------------
# 4. Public API
# -----------------------------------------------------------------------------

__all__ = [
    "PROJECT_ROOT",
    "qepc_step",
    "load_nba_schedule",
    "get_today_games",
    "get_tomorrow_games",
    "get_upcoming_games",
    "load_games",
    "load_team_stats",
    "load_raw_player_data",
    "compute_lambda",
    "run_qepc_simulation",
    "run_diagnostics",
    "run_daily_backtest",
    "run_season_backtest",
]

print("[QEPC] Autoload complete.")
