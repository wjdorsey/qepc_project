#!/usr/bin/env python3
"""
QEPC Injury Data Fetcher - ESPN Focus
======================================

Simple, working injury fetcher that focuses on ESPN's reliable API.

Usage:
    python scripts/fetch_injuries_simple.py
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_FILE = PROJECT_ROOT / "data" / "Injury_Overrides.csv"


def fetch_espn_injuries():
    """Fetch injury data from ESPN's public scoreboard API"""
    print("🏥 Fetching NBA injury data from ESPN...")
    
    injuries = []
    
    try:
        # ESPN's scoreboard endpoint includes injury info
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract teams and their rosters
        for event in data.get("events", []):
            for competition in event.get("competitions", []):
                for competitor in competition.get("competitors", []):
                    team = competitor.get("team", {})
                    team_name = team.get("displayName", "Unknown")
                    
                    # Check roster for injuries
                    for athlete in competitor.get("roster", []):
                        # Check if player has injury status
                        if athlete.get("injured", False):
                            injuries.append({
                                "PlayerName": athlete.get("displayName", "Unknown"),
                                "Team": team_name,
                                "Status": "OUT",  # Default to OUT if injured flag is set
                                "Injury": athlete.get("injuryStatus", "Unknown"),
                                "Source": "ESPN",
                                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
        
        # If no games today, fetch from team pages
        if not injuries:
            print("   No injuries found in scoreboard, fetching from team rosters...")
            injuries = fetch_from_team_rosters()
        
        df = pd.DataFrame(injuries)
        print(f"   ✅ Found {len(df)} injuries")
        return df
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return pd.DataFrame()


def fetch_from_team_rosters():
    """Backup method: fetch from individual team pages"""
    injuries = []
    
    # NBA team IDs from ESPN
    team_ids = [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # ATL to GSW
        11, 14, 15, 16, 17, 18, 19, 20, 21, 22,  # HOU to MIA
        23, 24, 25, 26, 27, 28, 29, 30  # MIL to WAS
    ]
    
    for team_id in team_ids:
        try:
            url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/nba/teams/{team_id}"
            response = requests.get(url, timeout=5)
            
            if response.status_code != 200:
                continue
                
            data = response.json()
            team_name = data.get("team", {}).get("displayName", "Unknown")
            
            # Check roster
            for athlete in data.get("team", {}).get("athletes", []):
                if athlete.get("injured", False) or athlete.get("status", {}).get("type") != "active":
                    injuries.append({
                        "PlayerName": athlete.get("displayName", "Unknown"),
                        "Team": team_name,
                        "Status": "OUT",
                        "Injury": athlete.get("status", {}).get("type", "Unknown"),
                        "Source": "ESPN",
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        except:
            continue
    
    return injuries


def main():
    print("=" * 60)
    print("🏀 QEPC Injury Data Fetcher")
    print("=" * 60)
    print()
    
    # Fetch injuries
    df = fetch_espn_injuries()
    
    if df.empty:
        print("\n⚠️  No injury data found. This could mean:")
        print("   • No games scheduled today")
        print("   • API might be temporarily unavailable")
        print("   • No players currently injured")
        return
    
    # Save to CSV
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df = df.sort_values(["Team", "PlayerName"])
    df.to_csv(OUTPUT_FILE, index=False)
    
    print()
    print("=" * 60)
    print(f"✅ Saved {len(df)} injuries to:")
    print(f"   {OUTPUT_FILE}")
    print("=" * 60)
    print()
    print("📊 Injury Breakdown by Team:")
    for team, count in df["Team"].value_counts().items():
        print(f"   {team}: {count}")
    print()


if __name__ == "__main__":
    main()