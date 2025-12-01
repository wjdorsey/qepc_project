# --- Robust bootstrap to load notebook_header.py no matter where Jupyter started ---

import sys
import importlib.util
from pathlib import Path

# 1) Find the project root: the folder that contains notebook_header.py
cur = Path.cwd()
project_root = None

for _ in range(6):  # walk up a few levels just in case
    if (cur / "notebook_header.py").exists():
        project_root = cur
        break
    cur = cur.parent

if project_root is None:
    raise FileNotFoundError(
        "Could not find notebook_header.py in the current directory or its parents."
    )

# 2) Make sure project root is on sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 3) Load notebook_header.py as a proper module
header_path = project_root / "notebook_header.py"
spec = importlib.util.spec_from_file_location("notebook_header", header_path)
notebook_header = importlib.util.module_from_spec(spec)

# IMPORTANT: register it in sys.modules so @dataclass doesn't break
sys.modules[spec.name] = notebook_header

spec.loader.exec_module(notebook_header)

# 4) Now call qepc_notebook_setup from that module
env = notebook_header.qepc_notebook_setup(run_diagnostics=False)
data_dir = env.data_dir
raw_dir = env.raw_dir

print("✅ QEPC environment initialized")
print("project_root:", project_root)
print("data_dir:", data_dir)
print("raw_dir:", raw_dir)
