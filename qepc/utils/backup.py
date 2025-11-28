"""
QEPC Module: backup.py
======================

Create a timestamped ZIP archive of the core project files.

Usage (from main.py)
--------------------
from qepc.utils.backup import create_project_backup

create_project_backup(verbose=True)
"""

from __future__ import annotations

import os
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

from qepc.autoload import paths


# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

# Relative paths (from project root) to include in backups
FILES_TO_INCLUDE: List[Path] = [
    Path("qepc"),
    Path("notebooks"),
    Path("data"),
    Path("main.py"),
    Path("qepc_autoload.py"),
    Path("tools.ipynb"),          # <- fixed typo (was tools.ipymb)
    Path("RESTORE_GUIDE.ipynb"),
    Path("requirements.txt"),
    Path("README.md"),
    Path(".gitignore"),
]

# Directory / path substrings to exclude
# (substring match on POSIX-style paths)
PATHS_TO_EXCLUDE_STR: List[str] = [
    "data/raw",           # Exclude massive raw data
    "data/backups",       # Exclude existing backup zips (critical fix)
    "data/cache",
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
]


# -----------------------------------------------------------------------------
# INTERNAL HELPERS
# -----------------------------------------------------------------------------

def _should_exclude(rel_path: Path) -> bool:
    """
    Check whether a relative path should be excluded from the backup.
    """
    posix = rel_path.as_posix()
    return any(ex in posix for ex in PATHS_TO_EXCLUDE_STR)


def _iter_files(root: Path, targets: Iterable[Path]) -> Iterable[Path]:
    """
    Yield all files under the given targets (respecting exclusions).
    """
    for target in targets:
        target_path = root / target

        if not target_path.exists():
            continue

        if target_path.is_file():
            if not _should_exclude(target):
                yield target_path
            continue

        # target_path is a directory
        for dirpath, dirnames, filenames in os.walk(target_path):
            dirpath = Path(dirpath)
            rel_dir = dirpath.relative_to(root)

            # Skip whole directories if they match exclusions
            if _should_exclude(rel_dir):
                # Stop walking deeper into this directory
                dirnames[:] = []
                continue

            for filename in filenames:
                full_path = dirpath / filename
                rel_path = full_path.relative_to(root)

                if _should_exclude(rel_path):
                    continue

                yield full_path


# -----------------------------------------------------------------------------
# PUBLIC API
# -----------------------------------------------------------------------------

def create_project_backup(verbose: bool = True) -> Path:
    """
    Create a timestamped ZIP archive of the QEPC project.

    Parameters
    ----------
    verbose : bool
        If True, prints progress and result summary.

    Returns
    -------
    Path
        Path to the created ZIP file.

    Raises
    ------
    Exception
        Re-raises any failure after cleaning up the partial zip.
    """
    root_dir = paths.get_project_root()
    backup_dir = paths.get_backup_dir()

    # Normalize to Path objects
    if not isinstance(root_dir, Path):
        root_dir = Path(root_dir)
    if not isinstance(backup_dir, Path):
        backup_dir = Path(backup_dir)

    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    zip_filename = f"qepc_backup_{timestamp}.zip"
    zip_path = backup_dir / zip_filename

    if verbose:
        print(f"📦 Starting QEPC project backup to: {zip_path.name}")

    try:
        with zipfile.ZipFile(
            zip_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
        ) as zf:
            for full_path in _iter_files(root_dir, FILES_TO_INCLUDE):
                rel_path = full_path.relative_to(root_dir)
                zf.write(full_path, rel_path)

        if verbose:
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            print(f"✅ Backup successful! Size: {size_mb:.2f} MB")

        return zip_path

    except Exception as e:
        if verbose:
            print(f"❌ CRITICAL BACKUP FAILURE: {e}")
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)
        raise
