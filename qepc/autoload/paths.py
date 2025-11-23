"""
QEPC Module: paths.py
Project path resolution and file location helpers.
"""
from pathlib import Path

# This file (paths.py) is located at:
# /project_root/qepc/autoload/paths.py

# Resolve the project root by going up two levels from this file's directory.
def get_project_root() -> Path:
    """Returns the absolute path to the project root directory."""
    # .resolve() handles symlinks
    # .parent gets the directory
    # .parent.parent is two steps up: autoload/ -> qepc/ -> project_root/
    return Path(__file__).resolve().parent.parent.parent

def get_data_dir() -> Path:
    """Returns the absolute path to the data directory (project_root/data)."""
    return get_project_root() / "data"

def get_games_path() -> Path:
    """Returns the absolute path to the Games.csv file."""
    return get_data_dir() / "Games.csv"

def get_backup_dir() -> Path:
    """Returns the absolute path to the data backups directory."""
    return get_data_dir() / "backups"

def get_notebooks_dir() -> Path:
    """Returns the absolute path to the notebooks directory."""
    return get_project_root() / "notebooks"

# Quick check on import
PROJECT_ROOT = get_project_root()
print(f"[QEPC Paths] Project Root set: {PROJECT_ROOT}")