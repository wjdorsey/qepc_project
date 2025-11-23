"""
QEPC Module: backup.py
Provides a function to create a timestamped ZIP archive of the core project files.
"""
import zipfile
from datetime import datetime
from pathlib import Path
import os

from qepc.autoload import paths

# --- Configuration ---
# Folders/files to INCLUDE in the backup (relative to project root)
# This list is exhaustive for all functional parts of the project.
FILES_TO_INCLUDE = [
    Path("qepc"),        # Includes ALL code modules
    Path("notebooks"),    # Includes ALL notebooks
    Path("data"),         # Includes data/metadata, data/cache, and canonical CSVs (data/raw is filtered out below)
    Path("main.py"),      # CLI entry point
    Path("qepc_autoload.py"), # Project shim
    Path("RESTORE_GUIDE.md"), # Restoration guide
    Path("RESTORE_GUIDE.ipynb"),
    Path("requirements.txt"), # Dependencies list
    Path("README.md"),      # Project documentation
    Path(".gitignore"),     # Version control config
]

# Paths to EXCLUDE inside the included folders (using starts-with logic on path strings)
# This prevents the backup from becoming huge due to raw files.
PATHS_TO_EXCLUDE_STR = [
    "data/raw/",
    "data/cache/",
    "__pycache__",
    ".ipynb_checkpoints"
]


def create_project_backup(verbose: bool = True) -> Path:
    """
    Creates a timestamped ZIP archive of the core project structure, excluding raw data.
    """
    ROOT_DIR = paths.get_project_root()
    BACKUP_DIR = paths.get_backup_dir()
    
    # 1. Ensure the backup directory exists
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Define the output file name
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_filename = f"qepc_backup_{timestamp}.zip"
    zip_path = BACKUP_DIR / zip_filename

    if verbose:
        print(f"📦 Starting QEPC project backup to: {zip_path.name}")
        
    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for target in FILES_TO_INCLUDE:
                target_path = ROOT_DIR / target
                
                if not target_path.exists():
                    if verbose:
                        print(f"   ⚠️ Skipping missing path: {target}")
                    continue

                if target_path.is_file():
                    # If it's a file, just write it
                    zf.write(target_path, target)
                    continue

                if target_path.is_dir():
                    # If it's a directory, walk through its contents
                    for dirpath, dirnames, filenames in os.walk(target_path):
                        dirpath = Path(dirpath)
                        
                        # Calculate the relative path from the project root
                        rel_dirpath = dirpath.relative_to(ROOT_DIR)
                        
                        # Filter out excluded paths
                        if any(rel_dirpath.as_posix().startswith(ex) for ex in PATHS_TO_EXCLUDE_STR):
                            dirnames[:] = [] # Stop os.walk from entering excluded dirs
                            continue
                        
                        # Add files in the current directory
                        for filename in filenames:
                            full_path = dirpath / filename
                            rel_path = full_path.relative_to(ROOT_DIR)
                            
                            # Final check against exclusions (e.g., specific files)
                            if any(rel_path.as_posix().startswith(ex) for ex in PATHS_TO_EXCLUDE_STR):
                                continue

                            zf.write(full_path, rel_path)

        if verbose:
            print("✅ Backup successful!")
            
        return zip_path

    except Exception as e:
        if verbose:
            print(f"❌ CRITICAL BACKUP FAILURE: {e}")
        # Clean up failed zip file
        if zip_path.exists():
            os.remove(zip_path)
        raise e

# --- Test function proxy for easy use in new main.py ---
def auto_backup():
    try:
        create_project_backup(verbose=True)
    except Exception as e:
        print("Backup failed. Check logs.")