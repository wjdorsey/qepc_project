# 🔧 02_utilities Notebooks - Review & Summary

## Overview

Your `02_utilities/` folder handles data management, fetching, and processing.

---

## ✅ Status Summary

| Notebook | Status | Action Needed |
|----------|--------|---------------|
| `qepc_project_backup.ipynb` | ✅ Perfect | None - keep as-is! |
| `injury_data_fetch.ipynb` | ✅ Excellent | None - already great! |
| `balldontlie_sync.ipynb` | ✅ Good | None - works well! |
| `Injury_impact_analysis.ipynb` | ❌ Broken | Replace with fixed version |
| `injuries_merge.ipynb` | ❓ Not uploaded | Can't review |

---

## 📄 Individual Notebook Details

### 1. qepc_project_backup.ipynb ✅
**Status:** Perfect!

**What it does:**
- Creates timestamped ZIP backups
- Excludes cache/raw/backups folders
- Saves to `data/backups/`
- Shows file size

**No changes needed** - this is clean and functional!

**Usage:**
```python
# Just run cell 2
# Creates: qepc_backup_YYYY-MM-DD_HHMMSS.zip
```

---

### 2. injury_data_fetch.ipynb ✅
**Status:** Excellent - already reviewed!

**What it does:**
- Fetches from 3 sources (ESPN, NBA official, balldontlie)
- Normalizes all to common schema
- Converts status to impact factors
- Creates master file with priority merging

**Features:**
- Multi-source fetching
- Intelligent priority system
- Impact calculations (OUT=0.70, Questionable=0.90, etc)
- Master file generation

**Keep this one as-is!**

---

### 3. balldontlie_sync.ipynb ✅
**Status:** Good!

**What it does:**
- Syncs NBA teams from balldontlie API
- Fetches recent games (configurable date range)
- Saves to CSVs for reference

**Key Features:**
- API authentication
- Error handling (401 unauthorized check)
- Configurable date ranges
- Team and games caching

**One note:** Remember to add your actual API key before running!

**Files created:**
- `data/balldontlie_teams.csv`
- `data/balldontlie_games_recent.csv`

---

### 4. Injury_impact_analysis.ipynb ❌ → ✅
**Original Status:** BROKEN - Missing module

**Problem:**
```python
from qepc.sports.nba.injury_impact_calculator import (...)
# ModuleNotFoundError: No module named 'qepc.sports.nba.injury_impact_calculator'
```

**Solution:** Rebuilt the notebook to work without that module!

**NEW Features:**
1. **Calculate Impact Factors** - Based on usage rate & minutes
2. **Generate Reference Data** - Creates `Injury_Overrides_data_driven.csv`
3. **Analyze Individual Players** - Lookup specific players
4. **Compare Manual vs Data** - See differences
5. **Impact Level Classification** - Critical/High/Moderate/Low

**Impact Formula:**
```python
Base Impact (by minutes):
- <15 min: 0.95 (bench)
- 15-25 min: 0.85 (role player)
- 25-32 min: 0.75 (starter)
- 32+ min: 0.65 (star)

Adjusted by usage rate
Clamped between 0.60-1.00
```

**Files:**
- Original (broken): `Injury_impact_analysis.ipynb`
- Fixed: `injury_impact_analysis_fixed.ipynb` ⭐

---

### 5. injuries_merge.ipynb ❓
**Status:** Not uploaded - can't review

**Expected purpose:** Probably merges different injury sources?

If you upload this, I can review it too!

---

## 🎯 Recommended Actions

### Immediate:
1. ✅ **Keep** qepc_project_backup.ipynb (perfect!)
2. ✅ **Keep** injury_data_fetch.ipynb (excellent!)
3. ✅ **Keep** balldontlie_sync.ipynb (good!)
4. 🔨 **Replace** Injury_impact_analysis.ipynb with fixed version

### Optional:
5. 📤 **Upload** injuries_merge.ipynb if you want it reviewed

---

## 📥 Files to Download

### Essential:
- **injury_impact_analysis_fixed.ipynb** - Working version
  - Replace your broken `Injury_impact_analysis.ipynb` with this

### Already Good:
- `qepc_project_backup.ipynb` ✅ Keep as-is
- `injury_data_fetch.ipynb` ✅ Keep as-is  
- `balldontlie_sync.ipynb` ✅ Keep as-is

---

## 🔄 How to Update

```bash
# Download the fixed notebook
# Rename to match original
mv injury_impact_analysis_fixed.ipynb Injury_impact_analysis.ipynb

# Replace in your project
# Test it by running all cells
```

---

## 🎨 What Makes the Fixed Version Better

### Original (Broken):
- ❌ Imported non-existent module
- ❌ Couldn't run at all
- ❌ ModuleNotFoundError immediately

### Fixed Version:
- ✅ Self-contained (no external modules)
- ✅ Calculates impacts from player stats
- ✅ Clear documentation
- ✅ Interactive analysis
- ✅ Comparison functionality
- ✅ Impact level guide

---

## 💡 Usage Tips

### For qepc_project_backup.ipynb:
- Run before making big changes
- Download the ZIP file immediately
- Store backups off-system (Google Drive, Dropbox)

### For injury_data_fetch.ipynb:
- Run daily during season
- Prioritizes official > ESPN > balldontlie
- Creates master file automatically
- Handles missing sources gracefully

### For balldontlie_sync.ipynb:
- Add your API key first!
- Adjust DAYS_BACK/DAYS_FORWARD as needed
- Good for historical game data
- Syncs teams (rarely changes)

### For Injury_impact_analysis.ipynb (NEW):
- Run after updating PlayerStatistics.csv
- Generates data-driven injury impacts
- Compare with manual overrides
- Analyze specific players
- Creates reference file for predictions

---

## 📊 Data Flow

```
injury_data_fetch.ipynb
    ↓
Fetches from APIs (ESPN, NBA, balldontlie)
    ↓
Creates individual source files:
- Injury_Overrides_live_espn.csv
- Injury_Overrides_live_official.csv
- Injury_Overrides_live_balldontlie.csv
    ↓
Merges with priority → Injury_Overrides_MASTER.csv

MEANWHILE:

Injury_impact_analysis.ipynb
    ↓
Reads PlayerStatistics.csv
    ↓
Calculates impact factors
    ↓
Creates Injury_Overrides_data_driven.csv
    ↓
Merges into final MASTER file
```

---

## 🎯 Integration with QEPC

These utilities feed into your prediction engine:

1. **injury_data_fetch** → Current injuries (who's out)
2. **Injury_impact_analysis** → Impact factors (how much it matters)
3. **Lambda engine** uses both to adjust team strength

**Example:**
```
Lakers without LeBron:
- Base ORtg: 115.0
- LeBron Impact: 0.70
- Adjusted ORtg: 115.0 × 0.70 = 80.5
- Used in predictions
```

---

## ✅ 02_utilities Summary

**Status:** 4 out of 5 notebooks are excellent!

**Working notebooks:**
- ✅ Project backup (perfect)
- ✅ Injury fetching (comprehensive)  
- ✅ Balldontlie sync (functional)
- ✅ Impact analysis (now fixed!)

**Actions:**
1. Download fixed Injury_impact_analysis.ipynb
2. Replace the broken version
3. Test by running all cells
4. Optional: Upload injuries_merge.ipynb for review

---

## 🚀 Ready for 03_dev?

Once you've updated the broken notebook, we can review your development/sandbox notebooks!

**Progress so far:**
- ✅ 00_setup - Reviewed & refined
- ✅ 01_core - Reviewed & enhanced
- ✅ 02_utilities - Reviewed & fixed
- ⏳ 03_dev - Ready when you are!

Upload your 03_dev notebooks or let me know if you want to test these updates first! 🎉
