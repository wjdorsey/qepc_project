"""
QEPC Notebook Header
--------------------

Single source of truth for Jupyter notebook setup.

Usage in ANY notebook (first cell):

    from qepc.notebook_header import qepc_notebook_setup

    env = qepc_notebook_setup(run_diagnostics=False)
    data_dir = env.data_dir
    raw_dir = env.raw_dir

This will:
  - Resolve the QEPC project root.
  - Add it to sys.path (if needed).
  - Import qepc_autoload (banner + path wiring).
  - Expose data_dir and raw_dir.
  - Optionally run system diagnostics once.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class NotebookEnv:
    """Simple container for common notebook paths + info."""
    project_root: Path
    data_dir: Path
    raw_dir: Path
    diagnostics: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------

def _find_project_root(start: Optional[Path] = None) -> Path:
    """
    Try to locate the QEPC project root by walking up until we find
    qepc_autoload.py and a 'qepc' package folder.

    Fallbacks:
      - QEPC_PROJECT_ROOT env var
      - current working directory
    """
    # 1) Environment variable override
    env_root = os.environ.get("QEPC_PROJECT_ROOT")
    if env_root:
        root = Path(env_root).resolve()
        if root.exists():
            return root

    # 2) Walk up from the starting directory
    if start is None:
        start = Path.cwd()

    for candidate in [start] + list(start.parents):
        if (candidate / "qepc_autoload.py").exists() and (candidate / "qepc").exists():
            return candidate

    # 3) Last resort: assume current working dir *is* the project root
    return start.resolve()


def _ensure_on_sys_path(root: Path, quiet: bool = False) -> None:
    """
    Make sure project_root is on sys.path so 'import qepc' works from any notebook.
    """
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.append(root_str)
        if not quiet:
            print(f"[NotebookHeader] Added to sys.path: {root_str}")
    else:
        if not quiet:
            print(f"[NotebookHeader] Project root already on sys.path: {root_str}")


# ---------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------

def qepc_notebook_setup(
    run_diagnostics: bool = False,
    quiet: bool = False,
) -> NotebookEnv:
    """
    Initialize QEPC environment for a notebook.

    Parameters
    ----------
    run_diagnostics : bool
        If True, run qepc.utils.diagnostics.run_system_check() once and
        include the result in env.diagnostics.
    quiet : bool
        If True, suppress most printouts (except serious errors).

    Returns
    -------
    NotebookEnv
        Contains project_root, data_dir, raw_dir, and optional diagnostics.
    """
    # -----------------------------------------------------------------
    # 1) Resolve and register project_root
    # -----------------------------------------------------------------
    project_root = _find_project_root()
    if not quiet:
        print(f"[NotebookHeader] QEPC project root: {project_root}")

    _ensure_on_sys_path(project_root, quiet=quiet)

    # -----------------------------------------------------------------
    # 2) Import qepc_autoload (banner + autoload paths)
    # -----------------------------------------------------------------
    try:
        import qepc_autoload  # noqa: F401  (side-effects only)

        if not quiet:
            print("[NotebookHeader] qepc_autoload imported successfully.")
    except Exception as exc:
        print("⚠️ [NotebookHeader] Failed to import qepc_autoload:", exc)

    # -----------------------------------------------------------------
    # 3) Resolve data_dir and raw_dir via autoload.paths
    # -----------------------------------------------------------------
    data_dir: Path
    raw_dir: Path

    try:
        from qepc.autoload.paths import get_data_dir  # type: ignore

        data_dir = get_data_dir()
    except Exception as exc:
        print("⚠️ [NotebookHeader] get_data_dir failed, falling back to project_root/data:", exc)
        data_dir = project_root / "data"

    try:
        # get_raw_data_dir may or may not exist
        from qepc.autoload.paths import get_raw_data_dir  # type: ignore

        raw_dir = get_raw_data_dir()
    except Exception:
        raw_dir = data_dir / "raw"

    if not quiet:
        print(f"[NotebookHeader] data_dir: {data_dir}")
        print(f"[NotebookHeader] raw_dir:  {raw_dir}")

    # -----------------------------------------------------------------
    # 4) Optionally run diagnostics
    # -----------------------------------------------------------------
    diagnostics_result: Optional[Dict[str, Any]] = None
    if run_diagnostics:
        try:
            from qepc.utils.diagnostics import run_system_check  # type: ignore

            if not quiet:
                print("\n[NotebookHeader] Running QEPC system diagnostics...")
            diagnostics_result = run_system_check()
        except Exception as exc:
            print("⚠️ [NotebookHeader] Failed to run diagnostics:", exc)

    # -----------------------------------------------------------------
    # 5) Package everything into NotebookEnv and return
    # -----------------------------------------------------------------
    env = NotebookEnv(
        project_root=project_root,
        data_dir=data_dir,
        raw_dir=raw_dir,
        diagnostics=diagnostics_result,
    )

    if not quiet:
        print("[NotebookHeader] Notebook environment ready.")

    return env


__all__ = ["NotebookEnv", "qepc_notebook_setup"]
