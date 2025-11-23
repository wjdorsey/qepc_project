import sys
from pathlib import Path

# Step 1: Add the project root to the system path
# This allows the 'import qepc' package to be found from any notebook.
root = Path(__file__).parent.resolve()
if str(root) not in sys.path:
    sys.path.append(str(root))
    
# Step 2: Forward the import to the internal autoloader core
try:
    from qepc.autoload.qepc_autoload import *
    print("[QEPC] Root Shim Loaded. Forwarding to qepc.autoload...")
except ImportError as e:
    print(f"[QEPC] Error loading internal autoloader: {e}")