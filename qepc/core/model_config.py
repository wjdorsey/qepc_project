"""
QEPC Module: model_config.py
Real NBA settings with expanded team-specific adjustments.
(Current season determined dynamically, e.g. 2025-26.)
"""

from datetime import datetime

# =============================================================================
# LEAGUE SETTINGS
# =============================================================================

LEAGUE_AVG_POINTS = 115.0  # modern scoring average

def get_current_season() -> str:
    """Dynamically determine current NBA season."""
    now = datetime.now()
    year = now.year
    if now.month >= 10:  # NBA season starts in October
        return f"{year}-{str(year+1)[2:]}"
    return f"{year-1}-{str(year)[2:]}"

CURRENT_SEASON = get_current_season()


# =============================================================================
# HOME COURT ADVANTAGE
# =============================================================================

BASE_HCA = 1.03  # 3% base home court advantage

# Team-specific HCA adjustments (multiplied by BASE_HCA)
# Values based on historical home/away splits and venue factors
TEAM_HCA_BOOST = {
    # Altitude advantage
    "Denver Nuggets": 1.020,        # 5,280 ft elevation - significant
    "Utah Jazz": 1.015,             # 4,226 ft elevation
    
    # Strong home crowds / venues
    "Boston Celtics": 1.015,        # TD Garden atmosphere
    "Golden State Warriors": 1.012, # Chase Center energy
    "Miami Heat": 1.010,            # Heat culture + crowd
    "Philadelphia 76ers": 1.012,    # Passionate fanbase
    "New York Knicks": 1.010,       # MSG factor
    "Phoenix Suns": 1.008,          # Footprint Center
    "Memphis Grizzlies": 1.010,     # FedExForum energy
    "Cleveland Cavaliers": 1.008,
    "Oklahoma City Thunder": 1.010, # Loud City
    "Indiana Pacers": 1.008,
    
    # Neutral / standard venues
    "Los Angeles Lakers": 1.005,    # Split arena, tourist crowd
    "Dallas Mavericks": 1.005,
    "Milwaukee Bucks": 1.005,
    "Sacramento Kings": 1.005,
    "Minnesota Timberwolves": 1.005,
    "New Orleans Pelicans": 1.003,
    "Atlanta Hawks": 1.003,
    "Toronto Raptors": 1.003,       # Cross-border travel helps
    "Chicago Bulls": 1.003,
    "Houston Rockets": 1.003,
    "Detroit Pistons": 1.000,
    "San Antonio Spurs": 1.000,
    "Charlotte Hornets": 1.000,
    "Portland Trail Blazers": 1.000,
    "Orlando Magic": 1.000,
    "Washington Wizards": 0.998,
    
    # Weaker home courts (shared arenas, poor attendance)
    "Los Angeles Clippers": 0.995,  # Intuit Dome is new, but historically weak
    "Brooklyn Nets": 0.990,         # Historically weak home court
}


# =============================================================================
# REST & FATIGUE ADJUSTMENTS
# =============================================================================

# Rest advantage per day of rest difference
REST_ADVANTAGE_PER_DAY = 1.5  # Points per day of rest difference

# Maximum rest advantage (caps the effect)
MAX_REST_ADVANTAGE = 4.0  # Max 4 point swing from rest

# Back-to-back penalty (multiplier on expected score)
B2B_PENALTY = 0.97  # 3% reduction for back-to-backs

# Travel penalty per 1000 miles
TRAVEL_PENALTY_PER_1000MI = 0.008  # 0.8% penalty per 1000 miles


# =============================================================================
# SIMULATION SETTINGS
# =============================================================================

# Default Monte Carlo trials
DEFAULT_NUM_TRIALS = 20000

# Quantum noise for lambda calculations
QUANTUM_NOISE_STD = 0.02  # 2% noise

# Score correlation between teams (pace entanglement)
SCORE_CORRELATION = 0.35

# Minimum realistic score floor
MIN_SCORE_FLOOR = 70  # No NBA team scores below 70


# =============================================================================
# RECENCY WEIGHTING
# =============================================================================

RECENCY_HALF_LIFE_DAYS = 30  # 30 days for 50% weight decay
MIN_GAMES_REQUIRED = 5       # Minimum games for stable strength rating


# =============================================================================
# CITY DISTANCES (for travel penalty)
# =============================================================================

# Major city pairs with approximate distances in miles
# Add more as needed
CITY_DISTANCES = {
    "Denver Nuggets": {
        "Los Angeles Lakers": 831,
        "Los Angeles Clippers": 831,
        "Golden State Warriors": 949,
        "Phoenix Suns": 586,
        "Portland Trail Blazers": 1238,
        "Utah Jazz": 371,
        "Sacramento Kings": 1013,
        "Miami Heat": 2065,
        "Boston Celtics": 1769,
        "New York Knicks": 1631,
        "Brooklyn Nets": 1631,
    },
    "Miami Heat": {
        "Boston Celtics": 1499,
        "New York Knicks": 1280,
        "Brooklyn Nets": 1280,
        "Philadelphia 76ers": 1199,
        "Washington Wizards": 1057,
        "Toronto Raptors": 1380,
        "Los Angeles Lakers": 2752,
        "Los Angeles Clippers": 2752,
        "Portland Trail Blazers": 3050,
    },
    "Los Angeles Lakers": {
        "Golden State Warriors": 347,
        "Sacramento Kings": 373,
        "Phoenix Suns": 357,
        "Portland Trail Blazers": 834,
        "Utah Jazz": 579,
        "Denver Nuggets": 831,
        "Boston Celtics": 2596,
        "New York Knicks": 2451,
        "Brooklyn Nets": 2451,
        "Miami Heat": 2752,
    },
    # Add more teams as needed
}


# =============================================================================
# PLAYOFF ADJUSTMENTS (for future use)
# =============================================================================

PLAYOFF_HCA_BOOST = 1.02      # Extra 2% home advantage in playoffs
PLAYOFF_VARIANCE_REDUCTION = 0.85  # Less variance in playoff games (tighter D)
ELIMINATION_GAME_BOOST = 1.03  # Teams facing elimination often overperform


# =============================================================================
# INJURY IMPACT DEFAULTS
# =============================================================================

# Default impact if player data not available
DEFAULT_STAR_IMPACT = 4.0      # Points impact for star player out
DEFAULT_STARTER_IMPACT = 2.0   # Points impact for starter out
DEFAULT_ROTATION_IMPACT = 0.5  # Points impact for rotation player out
