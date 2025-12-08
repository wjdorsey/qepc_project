# =========================
# CELL 1 – QEPC PATH SETUP
# =========================

import sys
from pathlib import Path
import pandas as pd  # still fine to keep this here

# Try to auto-detect the project root by walking up from the current directory
NOTEBOOK_DIR = Path.cwd()

PROJECT_ROOT = None
for parent in [NOTEBOOK_DIR] + list(NOTEBOOK_DIR.parents):
    # We treat any directory that contains a "qepc" folder as the project root
    if (parent / "qepc").is_dir():
        PROJECT_ROOT = parent
        break

# Fallback: if auto-detect fails, use the old hard-coded path
if PROJECT_ROOT is None:
    PROJECT_ROOT = Path(r"C:\Users\wdors\qepc_project").resolve()

# Make sure the project root is on sys.path so `import qepc...` works
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("NOTEBOOK_DIR:", NOTEBOOK_DIR)
print("PROJECT_ROOT:", PROJECT_ROOT)
print("qepc package exists here?:", (PROJECT_ROOT / "qepc").is_dir())

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

print("DATA_DIR:", DATA_DIR)
print("CACHE_DIR:", CACHE_DIR)
