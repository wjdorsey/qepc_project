# 🚀 QEPC Project: Quick Restore Guide

**IF YOUR SERVER RESETS, FOLLOW THESE STEPS TO RESTORE FUNCTIONALITY:**

## 1. Install Dependencies
Open a Terminal (File -> New -> Terminal) and run this command line:
`pip install pandas numpy matplotlib plotly ipython`

## 2. Verify Data Integrity
Ensure these files exist in your file browser before running models:
* `data/Games.csv` (The canonical schedule file)
* `data/Team_Stats.csv` (The canonical team list)
* `data/raw/PlayerStatistics.csv` (The massive player history file)
* `data/raw/TeamStatistics.csv` (The massive team history file)

## 3. Run the Notebook Headers
In every notebook (`qepc_dashboard.ipynb`, `qepc_backtest.ipynb`), you **MUST** run the first "Universal Loader" cell before running any other code.

This cell fixes the Python path and imports the QEPC engine:
```python
# ====================================================================
# 🌌 QEPC UNIVERSAL LOADER
# ====================================================================
import sys, os
sys.path.append(os.path.abspath('..')) 
from notebook_context import * print_ready_message()
print("="*80)
