# notebook_context.py
# Put this file in the ROOT of your qepc_project folder (same level as 'data' and 'notebooks')

import sys
from pathlib import Path

# Find the project root automatically
current = Path.cwd()
project_root = current
for p in [current] + list(current.parents):
    if (p / "data").exists() and (p / "qepc").exists():
        project_root = p
        break

# Add it to Python's path so imports work
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Handy shortcuts
project_root = project_root.resolve()
data_dir = project_root / "data"
raw_dir = data_dir / "raw"

print(f"QEPC project root set to: {project_root}")