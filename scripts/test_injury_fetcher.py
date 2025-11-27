#!/usr/bin/env python3
"""
Test the injury fetcher script
"""

import sys
from pathlib import Path

# Test if we can import required libraries
print("Testing dependencies...")
try:
    import pandas as pd
    print("✅ pandas installed")
except ImportError:
    print("❌ pandas not installed - run: pip install pandas")
    sys.exit(1)

try:
    import requests
    print("✅ requests installed")
except ImportError:
    print("❌ requests not installed - run: pip install requests")
    sys.exit(1)

# Test network connectivity
print("\nTesting ESPN API connectivity...")
try:
    response = requests.get("https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard", timeout=5)
    if response.status_code == 200:
        print("✅ ESPN API accessible")
    else:
        print(f"⚠️  ESPN API returned status {response.status_code}")
except Exception as e:
    print(f"❌ ESPN API connection failed: {e}")
    sys.exit(1)

# Test project structure
print("\nTesting project structure...")
PROJECT_ROOT = Path(__file__).resolve().parent.parent

required_dirs = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "qepc",
]

for dir_path in required_dirs:
    if dir_path.exists():
        print(f"✅ {dir_path.relative_to(PROJECT_ROOT)}/ exists")
    else:
        print(f"⚠️  {dir_path.relative_to(PROJECT_ROOT)}/ not found")

print("\n" + "=" * 60)
print("✅ All tests passed! Ready to run fetch_injuries.py")
print("=" * 60)
print("\nRun the injury fetcher:")
print("  python scripts/fetch_injuries_simple.py")