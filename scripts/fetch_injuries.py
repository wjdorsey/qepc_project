#!/usr/bin/env python3
"""
QEPC Unified Injury Data Pipeline
==================================

Fetches injury data from multiple sources and intelligently merges them.

Sources:
- ESPN API
- NBA.com Official Injury Report
- BallDontLie API

Usage:
    python scripts/fetch_injuries.py
    python scripts/fetch_injuries.py --source espn
    python scripts/fetch_injuries.py --output custom_injuries.csv
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
OUTPUT_FILE = PROJECT_ROOT / "data" / "Injury_Overrides.csv"
TIMEOUT = 10  # seconds


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def log(message: str, level: str = "INFO"):
    """Simple logging function"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def standardize_team_name(team: str) -> str:
    """Standardize team names to full names"""
    team_map = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards",
    }
    
    # Return as-is if already full name
    if team in team_map.values():
        return team
    
    # Try abbreviation lookup
    return team_map.get(team.upper(), team)


def standardize_status(status: str) -> str:
    """Standardize injury status codes"""
    status = status.upper().strip()
    
    status_map = {
        "OUT": "OUT",
        "QUESTIONABLE": "QUESTIONABLE",
        "DOUBTFUL": "DOUBTFUL",
        "DAY-TO-DAY": "DAY-TO-DAY",
        "GTD": "QUESTIONABLE",
        "PROBABLE": "QUESTIONABLE",
        "D2D": "DAY-TO-DAY",
    }
    
    return status_map.get(status, status)


# ==============================================================================
# DATA SOURCE FETCHERS
# ==============================================================================

def fetch_espn_injuries() -> pd.DataFrame:
    """
    Fetch injury data from ESPN API
    
    Returns:
        DataFrame with columns: PlayerName, Team, Status, Injury, Source, Timestamp
    """
    log("Fetching from ESPN API...")
    
    try:
        # ESPN NBA injuries endpoint
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams"
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        injuries = []
        
        # Parse team rosters and injuries
        for team_data in data.get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", []):
            team = team_data.get("team", {})
            team_name = team.get("displayName", "Unknown")
            
            # Get roster with injury info
            roster_url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team.get('id')}/roster"
            roster_response = requests.get(roster_url, timeout=TIMEOUT)
            
            if roster_response.status_code == 200:
                roster_data = roster_response.json()
                
                for athlete in roster_data.get("athletes", []):
                    injury_status = athlete.get("injuries", [])
                    
                    if injury_status:
                        for injury in injury_status:
                            injuries.append({
                                "PlayerName": athlete.get("displayName", "Unknown"),
                                "Team": standardize_team_name(team_name),
                                "Status": standardize_status(injury.get("status", "OUT")),
                                "Injury": injury.get("longComment", injury.get("type", "Unknown")),
                                "Source": "ESPN",
                                "Timestamp": datetime.now().isoformat()
                            })
        
        df = pd.DataFrame(injuries)
        log(f"ESPN: Found {len(df)} injuries")
        return df
        
    except Exception as e:
        log(f"ESPN fetch failed: {e}", "ERROR")
        return pd.DataFrame()


def fetch_nba_official_injuries() -> pd.DataFrame:
    """
    Fetch injury data from NBA.com Official Injury Report
    
    Returns:
        DataFrame with columns: PlayerName, Team, Status, Injury, Source, Timestamp
    """
    log("Fetching from NBA.com Official Injury Report...")
    
    try:
        # NBA.com injury report endpoint
        url = "https://www.nba.com/stats/js/data/widgets/home_injury.json"
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        
        data = response.json()
        injuries = []
        
        # Parse injury data
        for team_injuries in data.get("data", {}).get("items", []):
            team_name = team_injuries.get("teamName", "Unknown")
            
            for player in team_injuries.get("players", []):
                injuries.append({
                    "PlayerName": player.get("playerName", "Unknown"),
                    "Team": standardize_team_name(team_name),
                    "Status": standardize_status(player.get("status", "OUT")),
                    "Injury": player.get("injury", "Unknown"),
                    "Source": "NBA_Official",
                    "Timestamp": datetime.now().isoformat()
                })
        
        df = pd.DataFrame(injuries)
        log(f"NBA Official: Found {len(df)} injuries")
        return df
        
    except Exception as e:
        log(f"NBA Official fetch failed: {e}", "ERROR")
        return pd.DataFrame()


def fetch_balldontlie_injuries() -> pd.DataFrame:
    """
    Fetch injury data from BallDontLie API
    
    Returns:
        DataFrame with columns: PlayerName, Team, Status, Injury, Source, Timestamp
    """
    log("Fetching from BallDontLie API...")
    
    try:
        # Note: BallDontLie doesn't have a direct injury endpoint
        # This is a placeholder - would need actual implementation
        log("BallDontLie: No direct injury endpoint available", "WARNING")
        return pd.DataFrame()
        
    except Exception as e:
        log(f"BallDontLie fetch failed: {e}", "ERROR")
        return pd.DataFrame()


# ==============================================================================
# MERGING AND DEDUPLICATION
# ==============================================================================

def merge_injury_data(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge injury data from multiple sources with intelligent prioritization
    
    Priority: NBA_Official > ESPN > BallDontLie
    
    Args:
        dataframes: List of DataFrames from different sources
        
    Returns:
        Merged and deduplicated DataFrame
    """
    log("Merging injury data...")
    
    # Concatenate all sources
    all_injuries = pd.concat(dataframes, ignore_index=True)
    
    if all_injuries.empty:
        log("No injury data to merge!", "WARNING")
        return pd.DataFrame()
    
    # Define source priority
    source_priority = {
        "NBA_Official": 1,
        "ESPN": 2,
        "BallDontLie": 3
    }
    
    all_injuries["Priority"] = all_injuries["Source"].map(source_priority).fillna(99)
    
    # Sort by priority (lower number = higher priority)
    all_injuries = all_injuries.sort_values("Priority")
    
    # Remove duplicates, keeping highest priority source
    # Duplicates are identified by PlayerName + Team combination
    all_injuries["UniqueKey"] = all_injuries["PlayerName"] + "_" + all_injuries["Team"]
    deduplicated = all_injuries.drop_duplicates(subset=["UniqueKey"], keep="first")
    
    # Clean up helper columns
    deduplicated = deduplicated.drop(columns=["Priority", "UniqueKey"])
    
    log(f"Merged: {len(all_injuries)} total → {len(deduplicated)} unique injuries")
    
    # Log source breakdown
    source_counts = deduplicated["Source"].value_counts()
    for source, count in source_counts.items():
        log(f"  {source}: {count} injuries")
    
    return deduplicated


def save_injury_data(df: pd.DataFrame, output_path: Path):
    """Save injury data to CSV"""
    log(f"Saving to {output_path}...")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Sort by team and player name
    df = df.sort_values(["Team", "PlayerName"])
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    log(f"✅ Saved {len(df)} injuries to {output_path}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Fetch and merge NBA injury data from multiple sources"
    )
    parser.add_argument(
        "--source",
        choices=["all", "espn", "nba", "balldontlie"],
        default="all",
        help="Which source(s) to fetch from (default: all)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUTPUT_FILE),
        help=f"Output CSV file (default: {OUTPUT_FILE})"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress log messages"
    )
    
    args = parser.parse_args()
    
    # Override logging if quiet mode
    global log
    if args.quiet:
        log = lambda *args, **kwargs: None
    
    print("=" * 60)
    print("🏥 QEPC Unified Injury Data Pipeline")
    print("=" * 60)
    print()
    
    # Fetch from selected sources
    dataframes = []
    
    if args.source in ["all", "espn"]:
        df_espn = fetch_espn_injuries()
        if not df_espn.empty:
            dataframes.append(df_espn)
    
    if args.source in ["all", "nba"]:
        df_nba = fetch_nba_official_injuries()
        if not df_nba.empty:
            dataframes.append(df_nba)
    
    if args.source in ["all", "balldontlie"]:
        df_bdl = fetch_balldontlie_injuries()
        if not df_bdl.empty:
            dataframes.append(df_bdl)
    
    # Check if we got any data
    if not dataframes:
        log("No injury data fetched from any source!", "ERROR")
        print("\n❌ Failed to fetch injury data. Check your internet connection.")
        sys.exit(1)
    
    # Merge and save
    merged_df = merge_injury_data(dataframes)
    
    if merged_df.empty:
        log("Merged data is empty!", "ERROR")
        sys.exit(1)
    
    output_path = Path(args.output)
    save_injury_data(merged_df, output_path)
    
    print()
    print("=" * 60)
    print("✅ Injury Data Pipeline Complete!")
    print("=" * 60)
    print(f"\n📊 Summary:")
    print(f"   Total Injuries: {len(merged_df)}")
    print(f"   Teams Affected: {merged_df['Team'].nunique()}")
    print(f"   Output File: {output_path}")
    print()
    print("Status Breakdown:")
    for status, count in merged_df["Status"].value_counts().items():
        print(f"   {status}: {count}")
    print()


if __name__ == "__main__":
    main()