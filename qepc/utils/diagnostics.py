"""
QEPC Module: diagnostics.py
System health check and pre-flight validation tool.
"""
import os
import sys
import pandas as pd
from pathlib import Path
from IPython.display import display, HTML

# Define expected locations relative to project root
REQUIRED_FILES = {
    "Canonical Schedule": "data/Games.csv",
    "Raw Player Stats": "data/raw/PlayerStatistics.csv",
    "Raw Team Stats": "data/raw/TeamStatistics.csv",
    "Context Bootloader": "notebook_context.py",
    "Autoload Shim": "qepc_autoload.py",
    "Restore Guide": "RESTORE_GUIDE.md"
}

# Define critical columns that MUST exist for the pipeline to run
CRITICAL_SCHEMA = {
    "data/Games.csv": ["Date", "Time", "Away Team", "Home Team"],
    "data/raw/PlayerStatistics.csv": ["playerteamName", "gameDate", "gameId", "points"],
    "data/raw/TeamStatistics.csv": ["teamName", "opponentTeamName", "teamScore"]
}

def print_status(name, status, message=""):
    icon = "✅" if status else "❌"
    color = "green" if status else "red"
    display(HTML(f"<div style='color:{color}; font-family:monospace;'>{icon} <b>{name}</b>: {message}</div>"))

def run_system_check():
    """
    Executes a full pre-flight check of the QEPC environment.
    """
    print("🚀 QEPC SYSTEM DIAGNOSTICS INITIALIZED...\n")
    
    # 1. PATH CHECK
    try:
        from qepc.autoload import paths
        root = paths.get_project_root()
        print_status("Project Root", True, f"Found at {root}")
    except ImportError:
        print_status("Project Root", False, "Could not load path helper.")
        return

    # 2. FILE EXISTENCE CHECK
    all_files_ok = True
    for label, rel_path in REQUIRED_FILES.items():
        full_path = root / rel_path
        if full_path.exists():
            print_status(f"File: {label}", True, "OK")
        else:
            print_status(f"File: {label}", False, f"MISSING at {rel_path}")
            all_files_ok = False
    
    if not all_files_ok:
        print("\n⚠️ CRITICAL: Missing files detected. Restore from backup immediately.")
        return

    # 3. DATA SCHEMA CHECK (Lightweight)
    print("\n--- Checking Data Integrity (Headers Only) ---")
    for rel_path, required_cols in CRITICAL_SCHEMA.items():
        full_path = root / rel_path
        try:
            # Read only the header
            df = pd.read_csv(full_path, nrows=0)
            missing = [c for c in required_cols if c not in df.columns]
            
            if not missing:
                print_status(f"Schema: {Path(rel_path).name}", True, "Columns Valid")
            else:
                print_status(f"Schema: {Path(rel_path).name}", False, f"Missing Columns: {missing}")
        except Exception as e:
            print_status(f"Schema: {Path(rel_path).name}", False, f"Read Error: {e}")

    # 4. MODULE IMPORT CHECK
    print("\n--- Checking Core Modules ---")
    modules_to_test = [
        "qepc.core.lambda_engine",
        "qepc.core.simulator",
        "qepc.sports.nba.sim",
        "qepc.sports.nba.strengths_v2",
        "qepc.sports.nba.player_data",
        "qepc.sports.nba.opponent_data",
        "qepc.utils.backup"
    ]
    
    for mod in modules_to_test:
        try:
            __import__(mod)
            print_status(f"Module: {mod}", True, "Loaded")
        except ImportError as e:
            print_status(f"Module: {mod}", False, f"Import Failed: {e}")

    print("\n✨ DIAGNOSTICS COMPLETE.")
