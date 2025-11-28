# 0. QEPC Notebook Bootstrap – Setup & Helpers

from pathlib import Path
import sys
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime

def find_project_root() -> Path:
    """
    Walk upwards from the current working directory until we find
    a folder that looks like the QEPC project root.
    """
    here = Path.cwd()
    for parent in [here] + list(here.parents):
        if (parent / "main.py").exists() and (parent / "qepc").exists():
            return parent
    return here

project_root = find_project_root()

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("QEPC project root:", project_root)

# Try to load the shared notebook context (colors, helpers, etc.)
try:
    from notebook_context import *
    print("✅ notebook_context imported.")
except ModuleNotFoundError:
    print("⚠️ notebook_context not found; continuing with bare project_root only.")

# Where this notebook will read/write injury-related CSVs
data_dir = project_root / "data" / "injuries"
data_dir.mkdir(parents=True, exist_ok=True)
print("Injury data directory:", data_dir)
