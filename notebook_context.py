"""
QEPC Notebook Context Bootloader
Centralizes all imports and setup logic for Jupyter Notebooks.
"""
import sys
import os
from pathlib import Path

# 1. Standard Library Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.express as px
except ImportError:
    pass # Handle cases where plotly isn't installed yet

# 2. Project Setup
# Ensure the project root is in the path so we can import qepc modules
# We assume this file is in the project root.
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 3. QEPC Imports
try:
    import qepc_autoload as qa
    from qepc.core.lambda_engine import compute_lambda
    from qepc.core.simulator import run_qepc_simulation
    from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths
    
    # Define a ready message function
    def print_ready_message():
        qa.qepc_step("🌌 QEPC Bootstrap Loaded. Environment Ready (via Context).")
        
except ImportError as e:
    # Define a fallback message so the notebook doesn't crash on import
    def print_ready_message():
        print(f"❌ CRITICAL ERROR: Could not load QEPC framework. {e}")

# 4. Export variables to the notebook
# This controls what becomes available when you type 'from notebook_context import *'
__all__ = [
    'os', 'sys', 'Path',
    'pd', 'np', 'plt', 'px',
    'qa', 
    'compute_lambda', 'calculate_advanced_strengths',
    'run_qepc_simulation',
    'print_ready_message'
]