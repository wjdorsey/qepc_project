# 🚀 QEPC Project: Quick Restore Guide

**IF YOUR SERVER RESETS, FOLLOW THESE STEPS:**

## 1. Install Dependencies
Open a Terminal and run:
`pip install pandas numpy matplotlib plotly ipython`

## 2. Verify Data
Ensure these files exist in your file browser:
* `data/Games.csv` (The schedule)
* `data/Team_Stats.csv` (The canonical team list)
* `data/raw/PlayerStatistics.csv` (The massive player file)
* `data/raw/TeamStatistics.csv` (The massive team file)

## 3. Run the Notebook Headers
In every notebook (`qepc_dashboard.ipynb`, etc.), you MUST run the first two cells:
1. **Universal Loader** (Fixes the path)
2. **Autoload Framework** (Imports the QEPC engine)

## 4. Emergency Restore
If files are missing, upload your latest `qepc_backup_YYYY...zip` to `data/backups/`, unzip it, and overwrite the project folder.
