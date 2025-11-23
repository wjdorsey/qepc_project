import sys
from pathlib import Path

# This line adds the project root to Python's searchable paths.
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.append(str(root))

# This line then finds and runs the actual autoloader.
try:
    from qepc.autoload.qepc_autoload import *
    print("[QEPC] Root Shim Restored. Forwarding to qepc.autoload...")
except ImportError as e:
    print(f"[QEPC] Error loading internal autoloader: {e}")