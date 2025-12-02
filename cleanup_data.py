"""
QEPC Data Cleanup
=================
Reorganizes your data folder from the old messy structure to the new clean one.

Run this ONCE to:
1. Create clean folder structure
2. Move essential files to correct locations
3. Delete redundant/duplicate files
4. Clean up checkpoint folders

Usage:
    python cleanup_data.py           # Show what would be done (dry run)
    python cleanup_data.py --execute # Actually do it
"""

import os
import shutil
from pathlib import Path
from datetime import datetime
import argparse


def find_project_root() -> Path:
    """Find project root by looking for data folder."""
    current = Path.cwd()
    for p in [current] + list(current.parents)[:5]:
        if (p / "data").exists():
            return p
    return current


PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"


# ============================================================================
# FILES TO KEEP (map old location -> new location)
# ============================================================================

ESSENTIAL_FILES = {
    # Team statistics (for volatility)
    "raw/TeamStatistics.csv": "raw/TeamStatistics.csv",
    "TeamStatistics.csv": "raw/TeamStatistics.csv",
    
    # Live ratings
    "live/team_stats_live_nba_api.csv": "live/team_ratings.csv",
    
    # Game results (for backtesting)
    "GameResults_2025.csv": "raw/GameResults_2025.csv",
    
    # Schedule
    "Schedule_with_Rest.csv": "raw/Schedule_with_Rest.csv",
    "raw/LeagueSchedule25_26.csv": "raw/LeagueSchedule.csv",
    
    # Team form
    "TeamForm.csv": "raw/TeamForm.csv",
    
    # Player data (keep most useful)
    "raw/Player_Game_Logs_All_Seasons.csv": "raw/Player_Game_Logs.csv",
    "props/Player_Season_Averages.csv": "props/Player_Season_Averages.csv",
    
    # Injuries (consolidate)
    "Injury_Overrides_live_espn.csv": "injuries/current_injuries.csv",
    "injuries/Injury_Overrides_MASTER.csv": "injuries/Injury_Overrides_MASTER.csv",
}


# ============================================================================
# FILES TO DELETE (redundant/duplicate)
# ============================================================================

DELETE_PATTERNS = [
    # Checkpoint folders (Jupyter creates these)
    "**/.ipynb_checkpoints",
    "**/__pycache__",
    
    # Duplicate team stats
    "Team_Stats.csv",
    "TeamStatistics_Extended.csv",
    "NBA_API_Raw_Data.csv",
    "NBA_API_QEPC_Format.csv",
    
    # Old/duplicate files
    "raw/Team_Stats.csv",
    "raw/NBA_games.csv",
    
    # Multiple player files (keep only essentials)
    "props/Player_Recent_Form_L5.csv",
    "props/Player_Recent_Form_L10.csv",
    "props/Player_Recent_Form_L15.csv",
    "props/Player_Home_Away_Splits.csv",
    "raw/Player_Props_Averages.csv",
]


# ============================================================================
# CLEAN FOLDER STRUCTURE
# ============================================================================

CLEAN_STRUCTURE = [
    "live",           # Refreshed daily (todays_games, todays_odds, team_ratings)
    "raw",            # Historical data (TeamStatistics, GameResults, Schedule)
    "injuries",       # Injury data
    "props",          # Player props data
    "results/predictions",  # Saved predictions
    "results/backtests",    # Backtest results
]


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0


def analyze_current_state():
    """Analyze current data folder state."""
    print("\n📊 CURRENT DATA FOLDER ANALYSIS")
    print("=" * 60)
    
    if not DATA_DIR.exists():
        print(f"❌ Data directory not found: {DATA_DIR}")
        return
    
    # Count files by type
    csv_files = list(DATA_DIR.rglob("*.csv"))
    checkpoint_dirs = list(DATA_DIR.rglob(".ipynb_checkpoints"))
    
    total_size = sum(f.stat().st_size for f in csv_files) / (1024 * 1024)
    
    print(f"\n📁 Location: {DATA_DIR}")
    print(f"📄 Total CSV files: {len(csv_files)}")
    print(f"📦 Total size: {total_size:.1f} MB")
    print(f"🗑️  Checkpoint folders: {len(checkpoint_dirs)}")
    
    # List largest files
    print(f"\n📈 Largest files:")
    sorted_files = sorted(csv_files, key=lambda f: f.stat().st_size, reverse=True)
    for f in sorted_files[:10]:
        size = f.stat().st_size / (1024 * 1024)
        rel_path = f.relative_to(DATA_DIR)
        print(f"   {size:6.1f} MB  {rel_path}")
    
    # Check for duplicates
    print(f"\n🔍 Potential duplicates/redundant files:")
    for pattern in DELETE_PATTERNS[:5]:
        if not pattern.startswith("**"):
            path = DATA_DIR / pattern
            if path.exists():
                size = get_file_size_mb(path)
                print(f"   {size:6.1f} MB  {pattern}")


def create_clean_structure(dry_run: bool = True):
    """Create the clean folder structure."""
    print("\n📁 CREATING CLEAN FOLDER STRUCTURE")
    print("=" * 60)
    
    for folder in CLEAN_STRUCTURE:
        path = DATA_DIR / folder
        if path.exists():
            print(f"   ✅ {folder}/ (exists)")
        else:
            if dry_run:
                print(f"   📁 {folder}/ (would create)")
            else:
                path.mkdir(parents=True, exist_ok=True)
                print(f"   ✅ {folder}/ (created)")


def move_essential_files(dry_run: bool = True):
    """Move essential files to their correct locations."""
    print("\n📦 MOVING ESSENTIAL FILES")
    print("=" * 60)
    
    for old_path, new_path in ESSENTIAL_FILES.items():
        src = DATA_DIR / old_path
        dst = DATA_DIR / new_path
        
        if src.exists():
            if src == dst:
                print(f"   ✅ {old_path} (already in place)")
            elif dst.exists():
                print(f"   ⚠️  {old_path} -> {new_path} (destination exists)")
            else:
                if dry_run:
                    print(f"   📦 {old_path} -> {new_path}")
                else:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dst)
                    print(f"   ✅ {old_path} -> {new_path}")


def delete_redundant_files(dry_run: bool = True):
    """Delete redundant/duplicate files."""
    print("\n🗑️  CLEANING UP REDUNDANT FILES")
    print("=" * 60)
    
    total_freed = 0
    
    for pattern in DELETE_PATTERNS:
        if pattern.startswith("**"):
            # Glob pattern
            matches = list(DATA_DIR.glob(pattern))
            for path in matches:
                if path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / (1024 * 1024)
                    if dry_run:
                        print(f"   🗑️  {path.relative_to(DATA_DIR)}/ ({size:.1f} MB)")
                    else:
                        shutil.rmtree(path)
                        print(f"   ✅ Deleted {path.relative_to(DATA_DIR)}/")
                    total_freed += size
        else:
            # Direct path
            path = DATA_DIR / pattern
            if path.exists():
                size = get_file_size_mb(path)
                if dry_run:
                    print(f"   🗑️  {pattern} ({size:.1f} MB)")
                else:
                    path.unlink()
                    print(f"   ✅ Deleted {pattern}")
                total_freed += size
    
    print(f"\n   💾 Space {'would be ' if dry_run else ''}freed: {total_freed:.1f} MB")


def show_final_structure():
    """Show what the final structure looks like."""
    print("\n📁 FINAL CLEAN STRUCTURE")
    print("=" * 60)
    
    print("""
data/
├── live/                    # Refreshed daily via API
│   ├── todays_games.csv     # Today's schedule
│   ├── todays_odds.csv      # Vegas lines (NEW!)
│   └── team_ratings.csv     # Current ORtg/DRtg/Pace
│
├── raw/                     # Historical data
│   ├── TeamStatistics.csv   # Game-by-game stats (for volatility)
│   ├── GameResults_2025.csv # Results (for backtesting)
│   ├── LeagueSchedule.csv   # Full season schedule
│   └── Player_Game_Logs.csv # Player data (for props)
│
├── injuries/
│   └── current_injuries.csv # Consolidated injury data
│
├── props/                   # Player props
│   └── Player_Season_Averages.csv
│
└── results/
    ├── predictions/         # Your saved predictions
    └── backtests/           # Backtest results
""")


def main():
    parser = argparse.ArgumentParser(description='QEPC Data Cleanup')
    parser.add_argument('--execute', action='store_true', 
                       help='Actually perform the cleanup (default is dry run)')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze current state, no cleanup')
    
    args = parser.parse_args()
    
    dry_run = not args.execute
    
    print("=" * 60)
    print("🧹 QEPC DATA CLEANUP")
    print("=" * 60)
    print(f"📁 Project Root: {PROJECT_ROOT}")
    print(f"📂 Data Directory: {DATA_DIR}")
    
    if dry_run:
        print("\n⚠️  DRY RUN MODE - No changes will be made")
        print("   Run with --execute to actually perform cleanup")
    
    # Always analyze first
    analyze_current_state()
    
    if args.analyze_only:
        return
    
    # Create structure
    create_clean_structure(dry_run)
    
    # Move files
    move_essential_files(dry_run)
    
    # Delete redundant
    delete_redundant_files(dry_run)
    
    # Show final structure
    show_final_structure()
    
    if dry_run:
        print("\n" + "=" * 60)
        print("⚠️  This was a DRY RUN - no changes were made")
        print("   Run with --execute to actually perform cleanup:")
        print("   python cleanup_data.py --execute")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✅ CLEANUP COMPLETE!")
        print("=" * 60)


if __name__ == "__main__":
    main()
