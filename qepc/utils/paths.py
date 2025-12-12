"""Utilities for locating the QEPC project root.

This avoids hardcoded, machine-specific paths.  The detection order is:

1) Environment variable ``QEPC_PROJECT_ROOT`` (if set and exists).
2) Walk upward from the caller's location (or cwd) looking for markers
   such as ``.git``, ``pyproject.toml``, or a ``qepc`` package folder.
3) Fall back to ``Path.cwd()`` with a friendly warning.

All NBA notebooks/modules should import :func:`get_project_root` instead
of duplicating their own logic.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Iterable, Optional


_ROOT_MARKERS: tuple[str, ...] = (".git", "pyproject.toml", "qepc")


def _has_marker(path: Path) -> bool:
    for marker in _ROOT_MARKERS:
        target = path / marker
        if marker == "qepc":
            if target.is_dir():
                return True
        elif target.exists():
            return True
    return False


def _walk_upwards(start: Path) -> Optional[Path]:
    current = start
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if _has_marker(candidate):
            return candidate
    return None


def get_project_root(start: Optional[Path | str] = None) -> Path:
    """Return the QEPC project root.

    Parameters
    ----------
    start : Path or str, optional
        Where to begin the upward search.  Defaults to the caller's
        directory (``__file__``) if available or ``Path.cwd()``.
    """

    env_root = os.environ.get("QEPC_PROJECT_ROOT")
    if env_root:
        env_path = Path(env_root).expanduser().resolve()
        if env_path.exists():
            return env_path
        warnings.warn(
            f"QEPC_PROJECT_ROOT is set to {env_root} but does not exist;"
            " ignoring."
        )

    if start is None:
        try:
            start = Path(__file__).resolve()
        except Exception:
            start = Path.cwd()
    else:
        start = Path(start).expanduser().resolve()

    found = _walk_upwards(start)
    if found is not None:
        return found

    warnings.warn(
        "Unable to locate QEPC project root via markers; defaulting to cwd."
    )
    return Path.cwd()


PROJECT_ROOT: Path = get_project_root()


def ensure_relative_to_root(*parts: Iterable[str | Path]) -> Path:
    """Helper to build a path under the detected project root."""

    return PROJECT_ROOT.joinpath(*parts)

