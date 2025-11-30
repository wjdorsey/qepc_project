# notebook_context.py  –  FINAL VERSION 2025
# Works with every CSV format ever shipped with the repo
# Copy-paste this exactly, save, restart kernel → everything works forever

import sys
from pathlib import Path
import pandas as pd

# ────────────────────────────── 1. Find project root ──────────────────────────────
current = Path.cwd()
for p in [current] + list(current.parents):
    if (p / "notebook_context.py").exists() and (p / "qepc").exists():
        project_root = p.resolve()
        break
else:
    raise RuntimeError("Could not locate project root!")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print(f"QEPC project root locked in: {project_root}\n")

# ────────────────────────────── 2. Paths ──────────────────────────────
data_dir = project_root / "data"
raw_dir  = data_dir / "raw"

# ────────────────────────────── 3. Safe CSV loader ──────────────────────────────
def load_csv_safe(filename):
    path = raw_dir / filename
    if not path.exists():
        print(f"Warning: Missing {path.name}")
        return None
    df = pd.read_csv(path)
    # Force ALL columns to lowercase snake_case → no more KeyErrors ever
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

# ────────────────────────────── 4. Load the four core files ──────────────────────────────
print("Loading QEPC data files...\n")

team_stats   = load_csv_safe("Team_Stats.csv")
player_stats = load_csv_safe("PlayerStatistics.csv")
games        = load_csv_safe("Games.csv")
injuries     = load_csv_safe("Injury_Overrides.csv")

# ────────────────────────────── 5. Pretty status report ──────────────────────────────
def nice_print(df, name):
    if df is None:
        return
    size_mb = Path(raw_dir / f"{name}.csv").stat().st_size / (1024**2)
    if name == "Team_Stats":
        print(f"   {name}.csv → {len(df):,} team-seasons | {size_mb:.1f} MB")
    elif name == "PlayerStatistics":
        players = df["player_name"].nunique() if "player_name" in df.columns else "?"
        print(f"   {name}.csv → {len(df):,} rows | {players} players | {size_mb:.1f} MB")
    elif name == "Games":
        print(f"   {name}.csv → {len(df):,} games | {size_mb:.1f} MB")
    elif name == "Injury_Overrides":
        print(f"   {name}.csv → {len(df):,} overrides | {size_mb:.1f} MB")

nice_print(team_stats,   "Team_Stats")
nice_print(player_stats, "PlayerStatistics")
nice_print(games,        "Games")
nice_print(injuries,     "Injury_Overrides")

print("\nQEPC environment fully loaded — ready for real quantum predictions!\n")

# ────────────────────────────── 6. Export for notebooks ──────────────────────────────
__all__ = [
    "project_root", "data_dir", "raw_dir",
    "team_stats", "player_stats", "games", "injuries"
]