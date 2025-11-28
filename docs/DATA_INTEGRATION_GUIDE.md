# 🎯 QEPC Data Integration & Backtesting Guide

## 📊 What You Have (Amazing Dataset!)

| File | Rows | Contains | Use For |
|------|------|----------|---------|
| **TeamStatistics.csv** | 144,315 | 🏆 **GOLD!** Actual game results with stats | Real backtesting |
| LeagueSchedule24_25.csv | 1,409 | 2024-2025 season games | Schedule reference |
| LeagueSchedule25_26.csv | 1,279 | 2025-2026 season games | Future schedule |
| Games.csv | 785 | Current season schedule | Daily predictions |
| Players.csv | 6,679 | Player database | Player props, lineups |
| TeamHistories.csv | 141 | Team name changes | Data normalization |

---

## 🎯 Your Current Situation

### What's Happening:
1. ✅ You have **actual game results** through **Nov 17, 2025** (TeamStatistics.csv)
2. ✅ You have **future schedules** for 2025-2026 season
3. ❌ You're trying to backtest Oct-Nov 2025, but:
   - Games.csv has future dates (Oct 21, 2025+)
   - TeamStatistics.csv only goes to Nov 17, 2025
   - **Result: "No games found"**

### The Fix:
Use **TeamStatistics.csv** for backtesting - it has **278 actual games** from 2025!

---

## 🚀 Solution: Unified Data System

### Phase 1: Immediate Backtest Fix (5 minutes)

#### Step 1: Check Your Available Data
```python
import pandas as pd

# Load actual results
team_stats = pd.read_csv(project_root / "data" / "raw" / "TeamStatistics.csv")

# Parse dates
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])

# Check 2025 data
data_2025 = team_stats[team_stats['gameDate'].dt.year == 2025]

print(f"📊 2025 Season Data Available:")
print(f"   Date range: {data_2025['gameDate'].min()} to {data_2025['gameDate'].max()}")
print(f"   Total games: {len(data_2025) // 2}")  # Divide by 2 (each game = 2 rows)
print(f"   Rows: {len(data_2025)}")

# Sample games
print(f"\n🏀 Recent games with actual results:")
recent = data_2025.nlargest(10, 'gameDate')[['gameDate', 'teamCity', 'teamName', 
                                              'opponentTeamCity', 'opponentTeamName', 
                                              'teamScore', 'opponentScore', 'win']]
print(recent)
```

#### Step 2: Backtest Valid Date Range
```python
# In your backtest notebook:

# CORRECT: Backtest on dates you have actual results for
BACKTEST_START_DATE = pd.Timestamp("2025-10-22")  # Season start
BACKTEST_END_DATE = pd.Timestamp("2025-11-17")    # Latest data available

# This will find 26 days of games!
```

---

## 📈 Phase 2: Integrate TeamStatistics.csv (30 minutes)

### Create a Data Integration Notebook

```python
# notebooks/02_utilities/integrate_team_statistics.ipynb

import pandas as pd
from pathlib import Path

print("🔄 Integrating TeamStatistics.csv into QEPC")

# Load team statistics
team_stats_path = project_root / "data" / "raw" / "TeamStatistics.csv"
team_stats = pd.read_csv(team_stats_path)

print(f"✅ Loaded {len(team_stats)} rows")

# Convert dates
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])

# Filter to current season (2025)
season_2025 = team_stats[team_stats['gameDate'].dt.year == 2025].copy()

print(f"📊 2025 Season: {len(season_2025)} rows ({len(season_2025)//2} games)")

# Create QEPC-compatible Team_Stats.csv
# Group by team to get season averages
team_aggregates = season_2025.groupby(['teamCity', 'teamName']).agg({
    'fieldGoalsPercentage': 'mean',
    'threePointersPercentage': 'mean',
    'freeThrowsPercentage': 'mean',
    'reboundsTotal': 'mean',
    'assists': 'mean',
    'steals': 'mean',
    'blocks': 'mean',
    'turnovers': 'mean',
    'teamScore': 'mean',
    'opponentScore': 'mean',
    'seasonWins': 'last',
    'seasonLosses': 'last'
}).reset_index()

# Create full team name
team_aggregates['Team'] = team_aggregates['teamCity'] + ' ' + team_aggregates['teamName']

# Calculate offensive/defensive ratings
team_aggregates['ORtg'] = team_aggregates['teamScore']
team_aggregates['DRtg'] = team_aggregates['opponentScore']

print(f"\n✅ Created Team_Stats with {len(team_aggregates)} teams")

# Save to data folder
output_path = project_root / "data" / "raw" / "Team_Stats_2025.csv"
team_aggregates.to_csv(output_path, index=False)

print(f"💾 Saved to: {output_path}")

# Preview
print("\n📊 Top 5 Teams by ORtg:")
print(team_aggregates.nlargest(5, 'ORtg')[['Team', 'ORtg', 'DRtg', 'seasonWins', 'seasonLosses']])
```

---

## 🎯 Phase 3: Enhanced Backtesting (1 hour)

### Create Proper Backtest Pipeline

```python
# notebooks/01_core/qepc_backtest_enhanced.ipynb

import pandas as pd
from qepc.sports.nba.sim import simulate_game
from qepc.backtest.backtest_engine import run_season_backtest

print("🔬 Enhanced QEPC Backtest with Real Results")

# Load actual game results
team_stats = pd.read_csv(project_root / "data" / "raw" / "TeamStatistics.csv")
team_stats['gameDate'] = pd.to_datetime(team_stats['gameDate'])

# Get 2025 season games
games_2025 = team_stats[
    (team_stats['gameDate'].dt.year == 2025) &
    (team_stats['home'] == 1)  # Only home team rows (avoid duplicates)
].copy()

print(f"📊 Testing on {len(games_2025)} actual games")

# Backtest parameters
START_DATE = pd.Timestamp("2025-10-22")
END_DATE = pd.Timestamp("2025-11-17")

# Filter to date range
backtest_games = games_2025[
    (games_2025['gameDate'] >= START_DATE) &
    (games_2025['gameDate'] <= END_DATE)
]

print(f"🎯 Backtesting {len(backtest_games)} games from {START_DATE.date()} to {END_DATE.date()}")

# Run QEPC predictions
predictions = []

for idx, game in backtest_games.iterrows():
    home_team = f"{game['teamCity']} {game['teamName']}"
    away_team = f"{game['opponentTeamCity']} {game['opponentTeamName']}"
    
    # Get QEPC prediction
    try:
        pred = simulate_game(home_team, away_team)
        
        # Store prediction vs actual
        predictions.append({
            'date': game['gameDate'],
            'home_team': home_team,
            'away_team': away_team,
            'predicted_home_score': pred['home_score'],
            'predicted_away_score': pred['away_score'],
            'actual_home_score': game['teamScore'],
            'actual_away_score': game['opponentScore'],
            'predicted_winner': home_team if pred['home_score'] > pred['away_score'] else away_team,
            'actual_winner': home_team if game['win'] == 1 else away_team,
            'predicted_total': pred['home_score'] + pred['away_score'],
            'actual_total': game['teamScore'] + game['opponentScore']
        })
    except Exception as e:
        print(f"⚠️  Skipped {home_team} vs {away_team}: {e}")

# Convert to DataFrame
results = pd.DataFrame(predictions)

# Calculate accuracy
results['winner_correct'] = results['predicted_winner'] == results['actual_winner']
results['score_error_home'] = abs(results['predicted_home_score'] - results['actual_home_score'])
results['score_error_away'] = abs(results['predicted_away_score'] - results['actual_away_score'])
results['total_error'] = abs(results['predicted_total'] - results['actual_total'])

print(f"\n📊 Backtest Results:")
print(f"   Win Accuracy: {results['winner_correct'].mean():.1%}")
print(f"   Avg Score Error (Home): {results['score_error_home'].mean():.1f} points")
print(f"   Avg Score Error (Away): {results['score_error_away'].mean():.1f} points")
print(f"   Avg Total Error: {results['total_error'].mean():.1f} points")

# Save results
output_path = project_root / "data" / "results" / "backtests" / f"Enhanced_Backtest_{START_DATE.date()}_to_{END_DATE.date()}.csv"
results.to_csv(output_path, index=False)
print(f"\n💾 Saved detailed results to: {output_path}")

# Show best/worst predictions
print(f"\n🎯 Best Predictions (smallest total error):")
print(results.nsmallest(5, 'total_error')[['date', 'home_team', 'away_team', 'total_error']])

print(f"\n⚠️  Worst Predictions (largest total error):")
print(results.nlargest(5, 'total_error')[['date', 'home_team', 'away_team', 'total_error']])
```

---

## 🔥 Phase 4: Advanced Features (Future Enhancements)

### 1. Historical Trend Analysis
```python
# Use TeamStatistics.csv going back to 1946!
# Analyze team performance trends over decades
# Compare current season to historical averages
```

### 2. Player Props Integration
```python
# Use Players.csv for player prop predictions
# Link to TeamStatistics for historical player performance
# Build player-level models
```

### 3. Team Evolution Tracking
```python
# Use TeamHistories.csv to track team relocations
# Normalize team names across different eras
# Account for franchise moves in historical analysis
```

### 4. Schedule-Based Analysis
```python
# Use LeagueSchedule files for:
# - Rest days between games
# - Back-to-back analysis
# - Travel distance calculations
# - Arena advantage analysis
```

---

## 📁 Recommended File Organization

```
qepc_project/
└── data/
    ├── raw/                           # Raw source data
    │   ├── TeamStatistics.csv         # 🏆 Main source of truth
    │   ├── Players.csv                # Player database
    │   ├── TeamHistories.csv          # Team history
    │   ├── LeagueSchedule24_25.csv    # 2024-2025 schedule
    │   ├── LeagueSchedule25_26.csv    # 2025-2026 schedule
    │   └── Team_Stats_2025.csv        # Generated from TeamStatistics
    │
    ├── Games.csv                      # Current season schedule
    ├── Injury_Overrides.csv           # Injury data
    │
    └── results/
        └── backtests/                 # Backtest results
            └── Enhanced_Backtest_*.csv
```

---

## ⚡ Quick Start Checklist

### To Fix Your Backtest Issue:

- [ ] **Step 1:** Verify you have TeamStatistics.csv in `data/raw/`
- [ ] **Step 2:** Run diagnostic to check date range (code above)
- [ ] **Step 3:** Change backtest dates to Oct 22 - Nov 17, 2025
- [ ] **Step 4:** Re-run backtest
- [ ] **Step 5:** You'll get actual results!

### To Integrate Everything:

- [ ] **Step 1:** Create `integrate_team_statistics.ipynb`
- [ ] **Step 2:** Generate Team_Stats_2025.csv from TeamStatistics
- [ ] **Step 3:** Update QEPC to use new Team_Stats
- [ ] **Step 4:** Create enhanced backtest notebook
- [ ] **Step 5:** Run backtests on real data

---

## 🎯 Expected Results

### Current State:
```
❌ Backtest: Oct 21 - Nov 28, 2025
❌ Result: "No games found"
❌ Reason: Games.csv has future dates without results
```

### After Integration:
```
✅ Backtest: Oct 22 - Nov 17, 2025
✅ Result: 278 actual games with scores
✅ Accuracy: ~60-70% (typical for NBA predictions)
✅ Detailed metrics: Score accuracy, spread accuracy, total accuracy
```

---

## 💡 Pro Tips

### 1. Use Rolling Windows
```python
# Backtest in 7-day windows
# This shows how accuracy changes over time
```

### 2. Compare to Vegas Lines
```python
# If you have betting line data
# Compare QEPC accuracy to sportsbook accuracy
```

### 3. Track Calibration
```python
# Use backtest results to calibrate lambda values
# Improve future predictions
```

### 4. Analyze by Game Type
```python
# Split by:
# - Home/Away
# - Division games
# - Conference games
# - Back-to-backs
```

---

## 🚀 Next Steps

1. **Immediate (5 min):** Fix backtest dates to Oct 22 - Nov 17, 2025
2. **Short-term (30 min):** Create Team_Stats from TeamStatistics
3. **Medium-term (1 hour):** Build enhanced backtest pipeline
4. **Long-term (ongoing):** Add player props, historical trends, schedule analysis

---

## 📊 What Makes This Powerful

You now have:
- ✅ **278 actual games** to backtest against
- ✅ **Detailed statistics** for every game (48 metrics!)
- ✅ **Multiple seasons** of schedules
- ✅ **Player database** for props
- ✅ **Historical data** back to 1946

**This is a complete NBA analytics system!** 🏆

---

**Want me to create the specific notebooks for you?** 
Just let me know which phase you want to start with! 🚀
