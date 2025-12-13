from pathlib import Path
import sys

# 1) Find repo root (the folder that contains qepc/__init__.py)
PROJECT_ROOT = next(
    p for p in [Path.cwd().resolve()] + list(Path.cwd().resolve().parents)
    if (p / "qepc" / "__init__.py").exists()
)

# 2) Make qepc importable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 3) Now this works
from qepc.notebook_bootstrap import ensure_project_root
PROJECT_ROOT = ensure_project_root()

from qepc.utils.paths import get_project_root
PROJECT_ROOT = get_project_root(PROJECT_ROOT)

print("PROJECT_ROOT:", PROJECT_ROOT)
