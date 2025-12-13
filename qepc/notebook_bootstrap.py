from __future__ import annotations

from pathlib import Path
import sys

def ensure_project_root() -> Path:
    """
    Ensure the QEPC project root is on sys.path so `import qepc` works in notebooks.

    Finds the nearest parent directory containing:
      qepc/__init__.py
    """
    here = Path.cwd().resolve()
    project_root = None

    for p in [here] + list(here.parents):
        if (p / "qepc").is_dir() and (p / "qepc" / "__init__.py").exists():
            project_root = p
            break

    if project_root is None:
        raise RuntimeError(
            f"Could not find QEPC project root above {here}. "
            "Expected qepc/__init__.py to exist."
        )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root
