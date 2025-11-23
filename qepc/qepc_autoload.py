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
    """Displays a styled step message in Jupyter."""
    display(HTML(
        f"<div style='font-family:monospace; color:#4ea3ff; font-weight:bold; margin-top:10px;'>"
        f"⧉ QEPC: {msg}</div>"
    ))

# 3. Core Function Proxies (Links all project modules)
try:
    # Schedule and Game Helpers
    from qepc.sports.nba.sim import load_nba_schedule, get_today_games, get_tomorrow_games, get_upcoming_games
    # Data Cleaning / Stats Calculators
    from qepc.utils.data_cleaning import load_team_stats as load_dummy_team_stats 
    from qepc.sports.nba.player_data import load_raw_player_data
    # Core Modeling Engines
    from qepc.core.lambda_engine import compute_lambda 
    from qepc.core.simulator import run_qepc_simulation 
    # Advanced Strengths (V2)
    from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths 
    # System Diagnostics
    from qepc.utils.diagnostics import run_system_check # <--- NEW IMPORT
    
    # Define primary user proxies 
    load_games = load_nba_schedule 
    load_team_stats = calculate_advanced_strengths 
    run_diagnostics = run_system_check # <--- NEW PROXY
    
except ImportError as e:
    # CRITICAL: This line and all lines below MUST be indented correctly.
    print(f"[QEPC Autoload] ERROR: Failed to import core functions: {e}")
    
    # Define safe dummies to prevent crashes if imports fail
    def get_today_games(show='clean'): print("Schedule functions missing.")
    def get_tomorrow_games(show='clean'): print("Schedule functions missing.")
    def get_upcoming_games(days=7, show='clean'): print("Schedule functions missing.")
    def load_games(): print("Loader function missing.")
    def load_team_stats(): print("Team Stats Loader missing.")
    def compute_lambda(schedule, strengths): print("Lambda function missing.")
    def run_qepc_simulation(df, num_trials=20000): print("Simulator function missing.")
    def load_raw_player_data(file_name="PlayerStatistics.csv", usecols=None): print("Raw player data loader missing.")
    def run_diagnostics(): print("Diagnostics module missing.")


# 4. Final Confirmation
print("[QEPC] Autoload complete.")
