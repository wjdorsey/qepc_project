"""
QEPC Notebook Context Bootloader
Centralizes all imports and setup logic for Jupyter Notebooks.
"""
import sys
import os
from pathlib import Path

# 1. Standard Library & Data Science Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
try:
    import plotly.express as px
except ImportError:
    pass 

# 2. Project Path Setup
# Ensure the project root is in the path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# 3. QEPC Imports
try:
    import qepc_autoload as qa
    from qepc.core.lambda_engine import compute_lambda
    from qepc.core.simulator import run_qepc_simulation
    from qepc.sports.nba.strengths_v2 import calculate_advanced_strengths
    
    def print_ready_message():
        qa.qepc_step("🌌 QEPC Bootstrap Loaded. Environment Ready (via Context).")
        
except ImportError as e:
    print(f"❌ CRITICAL ERROR: Could not load QEPC framework. {e}")
    
    # Define SAFE DUMMIES so 'from notebook_context import *' doesn't crash
    def print_ready_message(): print("QEPC Load Failed.")
    def compute_lambda(*args): print("Missing compute_lambda"); return None
    def run_qepc_simulation(*args, **kwargs): print("Missing run_qepc_simulation"); return None
    def calculate_advanced_strengths(): print("Missing strengths calculator"); return None
    qa = None

# 4. Export variables to the notebook
__all__ = [
    'os', 'sys', 'Path',
    'pd', 'np', 'plt', 'px',
    'qa', 
    'compute_lambda', 
    'calculate_advanced_strengths',
    'run_qepc_simulation',
    'print_ready_message'
]
