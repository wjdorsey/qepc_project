# 🛠️ QEPC Scripts

Utility scripts for data fetching, processing, and automation.

## 📥 Data Fetching

### `fetch_injuries.py` - Comprehensive Injury Fetcher

Fetches NBA injury data from multiple sources with intelligent merging.

**Features:**
- Multi-source fetching (ESPN, NBA.com, BallDontLie)
- Source prioritization (Official > ESPN > BallDontLie)
- Automatic deduplication
- Timestamped data
- Team name standardization

**Usage:**
```bash
# Fetch from all sources
python scripts/fetch_injuries.py

# Fetch from specific source
python scripts/fetch_injuries.py --source espn

# Custom output location
python scripts/fetch_injuries.py --output data/custom_injuries.csv

# Quiet mode (no logs)
python scripts/fetch_injuries.py --quiet
```

**Output:**
- `data/Injury_Overrides.csv` (default)

---

### `fetch_injuries_simple.py` - ESPN-Only Quick Fetcher

Simplified version that focuses on ESPN's reliable API.

**Usage:**
```bash
python scripts/fetch_injuries_simple.py
```

**Best for:**
- Quick updates
- When other APIs are unavailable
- Daily automated runs

---

## 🔄 Data Processing

### `merge_schedules.py` - Schedule Consolidation

Merges multiple NBA schedule files into one unified schedule.

**Usage:**
```bash
python scripts/merge_schedules.py
```

**Input:**
- `LeagueSchedule24_25.csv`
- `Games.csv`

**Output:**
- `Games_Merged.csv`

---

## 🧪 Validation

### `validate_data.py` - Data Quality Checks

*(Coming Soon)*

Validates data integrity and completeness:
- Check for missing games
- Verify team names
- Detect data gaps
- Validate file formats

---

## 🤖 Automation

### Future Scripts

These scripts are planned for future development:

- `fetch_all_data.py` - One command to update everything
- `update_schedules.py` - Automated schedule updates
- `generate_daily_report.py` - Create prediction reports
- `run_backtest.py` - CLI backtest runner

---

## 📦 Requirements

All scripts require:
```bash
pip install pandas requests
```

For full functionality:
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

1. **Create scripts folder** (if it doesn't exist)
   ```bash
   mkdir -p scripts
   ```

2. **Make scripts executable** (Linux/Mac)
   ```bash
   chmod +x scripts/*.py
   ```

3. **Run your first fetch**
   ```bash
   python scripts/fetch_injuries_simple.py
   ```

---

## 🔧 Development

### Adding a New Script

1. Create file in `scripts/` folder
2. Add shebang: `#!/usr/bin/env python3`
3. Add docstring explaining purpose
4. Include usage examples
5. Add error handling
6. Update this README

### Script Template

```python
#!/usr/bin/env python3
"""
Script Name - Brief Description
================================

Longer description of what the script does.

Usage:
    python scripts/script_name.py [options]
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def main():
    print("Script running...")
    # Your code here

if __name__ == "__main__":
    main()
```

---

## 📝 Notes

- All scripts should be runnable from project root: `python scripts/script.py`
- Scripts should create necessary directories if they don't exist
- Use `PROJECT_ROOT` for absolute paths
- Include error handling and user-friendly messages
- Log progress for long-running operations

---

## 🤝 Contributing

When adding scripts:
1. Follow the template above
2. Add comprehensive docstrings
3. Include usage examples
4. Update this README
5. Test from project root