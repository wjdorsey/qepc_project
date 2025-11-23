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
FILES_TO_INCLUDE = [
    Path("qepc"),        
    Path("notebooks"),    
    Path("data"),         
    Path("main.py"),      
    Path("qepc_autoload.py"), 
    Path("tools.ipymb"), 
    Path("RESTORE_GUIDE.ipynb"),
    Path("requirements.txt"), 
    Path("README.md"),      
    Path(".gitignore"),     
]

# CRITICAL FIX: Added 'data/backups' to prevent recursive zipping
PATHS_TO_EXCLUDE_STR = [
    "data/raw",      # Exclude massive raw data
    "data/backups",  # Exclude existing zip files
    "data/cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".git"
]

def create_project_backup(verbose: bool = True) -> Path:
    ROOT_DIR = paths.get_project_root()
    BACKUP_DIR = paths.get_backup_dir()
    
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
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
                    continue

                if target_path.is_file():
                    zf.write(target_path, target)
                    continue

                if target_path.is_dir():
                    for dirpath, dirnames, filenames in os.walk(target_path):
                        dirpath = Path(dirpath)
                        rel_dirpath = dirpath.relative_to(ROOT_DIR)
                        posix_path = rel_dirpath.as_posix()
                        
                        # CHECK: Is this directory in the exclude list?
                        if any(ex in posix_path for ex in PATHS_TO_EXCLUDE_STR):
                            if verbose:
                                # Only print top-level exclusions to avoid spam
                                if posix_path in PATHS_TO_EXCLUDE_STR:
                                    print(f"   ⛔ Skipping excluded directory: {posix_path}")
                            dirnames[:] = [] # Stop walking deeper
                            continue
                        
                        # Add files
                        for filename in filenames:
                            full_path = dirpath / filename
                            rel_path = full_path.relative_to(ROOT_DIR)
                            
                            # Final check for specific file exclusions
                            if any(ex in rel_path.as_posix() for ex in PATHS_TO_EXCLUDE_STR):
                                continue

                            zf.write(full_path, rel_path)

        if verbose:
            # Verify size
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"✅ Backup successful! Size: {size_mb:.2f} MB")
            
        return zip_path

    except Exception as e:
        if verbose:
            print(f"❌ CRITICAL BACKUP FAILURE: {e}")
        if zip_path.exists():
            os.remove(zip_path)
        raise e
