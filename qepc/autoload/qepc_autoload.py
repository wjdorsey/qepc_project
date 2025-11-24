import sys
from pathlib import Path
from IPython.display import display, HTML
from typing import Optional
import pandas as pd

# 1. Path Setup
try:
    from qepc.autoload.paths import get_project_root
    ROOT_DIR = get_project_root()
except ImportError:
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.append(str(ROOT_DIR))

# 2. Visual Step Helper
def qepc_step(msg: str):
    display(HTML(
        f"<div style='font-family:monospace; color:#4ea3ff; font-weight:bold; margin-top:10px;'>"
        f"⧉ QEPC: {msg}</div>"
    ))

# 3. Core Function Proxies
try:
    from qepc.sports.nba.sim import load_nba_schedule, get_today_games, get_tomorrow_games, get_upcoming_games
    from qepc.utils.data_cleaning import load_team_stats as load_dummy_team_stats 
    from qepc.sports.nba.player_data import load_raw_player_data
    from qepc.core.lambda_engine import compute_lambda 
    from qepc.core.simulator import run_qepc_simulation 
    from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths 
    from qepc.utils.diagnostics import run_system_check 
    # Backtesting Engines
    from qepc.backtest.backtest_engine import run_daily_backtest, run_season_backtest # <--- NEW IMPORT
    
    # Proxies
    load_games = load_nba_schedule 
    load_team_stats = calculate_advanced_strengths 
    run_diagnostics = run_system_check 
    
except ImportError as e:
    print(f"[QEPC Autoload] ERROR: Failed to import core functions: {e}")
    
    def get_today_games(show='clean'): print("Missing.")
    def get_tomorrow_games(show='clean'): print("Missing.")
    def get_upcoming_games(days=7, show='clean'): print("Missing.")
    def load_games(): print("Missing.")
    def load_team_stats(): print("Missing.")
    def compute_lambda(schedule, strengths): print("Missing.")
    def run_qepc_simulation(df, num_trials=20000): print("Missing.")
    def load_raw_player_data(file_name="PlayerStatistics.csv", usecols=None): print("Missing.")
    def run_diagnostics(): print("Missing.")
    def run_daily_backtest(target_date, num_trials=5000): print("Missing.")
    def run_season_backtest(start_date, end_date): print("Missing.") # Dummy


# 4. Final Confirmation
print("[QEPC] Autoload complete.")