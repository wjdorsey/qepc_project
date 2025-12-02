# QEPC v2.1 Quick Start Guide

## 📥 Installation

1. **Extract** `QEPC_v2.1_Vegas.zip` to:
   ```
   C:\Users\wdors\qepc_project\experimental\CLAUDE_REWRITE\qepc_v2\
   ```

2. **Your folder structure should look like:**
   ```
   qepc_project/
   ├── data/                           # Your existing data
   ├── experimental/
   │   └── CLAUDE_REWRITE/
   │       └── qepc_v2/               # NEW - extract here
   │           ├── qepc/              # Python package
   │           ├── notebooks/         # Jupyter notebooks
   │           └── scripts/           # Utility scripts
   ```

---

## 🚀 Daily Workflow

### Step 1: Refresh Data (Local Only)
```
Open: notebooks/03_data_refresh.ipynb
Run all cells
```
This fetches:
- Today's games
- Vegas odds ← **NEW!**
- Team ratings

### Step 2: Make Predictions
```
Open: notebooks/01_daily_predictions.ipynb
Run all cells
```
This shows:
- Power rankings
- Today's predictions with Vegas comparison
- **Edge detection** ← Games where we disagree with Vegas!

### Step 3: (Optional) Backtest
```
Open: notebooks/02_backtest.ipynb
Run all cells
```
This validates accuracy over past games.

---

## ⭐ New Features in v2.1

### Vegas Odds Comparison
Every prediction now includes:
```
QEPC Spread: -3.5
Vegas Spread: -5.5
Difference: +2.0 pts
⭐ EDGE: Bet HOME (2.0 pt edge)
```

### Find Edges Function
```python
from qepc import find_edges

# Find all games where QEPC disagrees with Vegas by 2+ points
edges = find_edges()
```

### Quick Predict
```python
from qepc import quick_predict

pred = quick_predict("Boston Celtics", "Los Angeles Lakers")
# Shows prediction + Vegas comparison
```

---

## 📁 Clean Data Structure

After running `scripts/cleanup_data.py`, your data folder will be:

```
data/
├── live/                    # Refresh daily
│   ├── todays_games.csv     # Schedule from NBA API
│   ├── todays_odds.csv      # Vegas lines ← NEW!
│   └── team_ratings.csv     # ORtg, DRtg, Pace
│
├── raw/                     # Historical (rarely changes)
│   ├── TeamStatistics.csv   # Game-by-game stats
│   └── GameResults_2025.csv # For backtesting
│
├── injuries/
│   └── current_injuries.csv
│
└── results/
    ├── predictions/         # Your saved predictions
    └── backtests/           # Backtest results
```

---

## 🧹 Optional: Clean Up Old Files

Run the cleanup script to remove redundant files:

```bash
# Preview what would be deleted (dry run)
python scripts/cleanup_data.py

# Actually delete redundant files
python scripts/cleanup_data.py --execute
```

---

## 📊 Understanding Edge Detection

| Scenario | Meaning | Action |
|----------|---------|--------|
| `Spread_Diff > 2` | QEPC likes HOME more than Vegas | Consider betting HOME |
| `Spread_Diff < -2` | QEPC likes AWAY more than Vegas | Consider betting AWAY |
| `abs(Spread_Diff) < 2` | We agree with Vegas | No edge |

**Example:**
```
Game: Lakers @ Celtics
QEPC Spread: -8.5 (Celtics by 8.5)
Vegas Spread: -5.5 (Celtics by 5.5)
Difference: -3.0
→ EDGE: Bet CELTICS (we think they'll win by more)
```

---

## ⚠️ Disclaimer

This is for entertainment/research only. Sports betting involves risk. Always gamble responsibly.

---

## 🔧 Troubleshooting

**"Module not found"**
- Make sure the setup cell ran successfully
- Check that qepc_v2 folder is in the right place

**"No games found"**
- Run the data refresh notebook first
- Check if `data/live/todays_games.csv` exists

**"Vegas odds not available"**
- Run the data refresh notebook
- Odds only available for games that day

**Timezone errors**
- Fixed in v2.1! Let me know if they reappear.
