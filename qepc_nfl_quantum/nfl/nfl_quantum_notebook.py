"""
QEPC NFL Quantum Prediction Model - Interactive Notebook
=========================================================

This notebook demonstrates the quantum-inspired NFL prediction model.

Run each cell in order, or execute the whole file.
"""

# =============================================================================
# CELL 1: Setup & Imports
# =============================================================================

import sys
from pathlib import Path

# Add NFL folder to path
for folder in [Path.cwd(), Path.cwd() / "nfl", Path.cwd().parent]:
    if (folder / "nfl_quantum_engine.py").exists():
        sys.path.insert(0, str(folder))
        break

import pandas as pd
import numpy as np

from nfl_quantum_engine import (
    NFLQuantumEngine, 
    NFLTeamStrength, 
    NFLQuantumConfig,
    QuantumState,
    TeamState,
    create_sample_teams,
)
from nfl_strengths import NFLStrengthCalculator, generate_sample_nfl_games
from nfl_backtest_engine import NFLBacktestEngine

print("✅ NFL Quantum Engine loaded!")
print("\n🌌 Quantum Concepts Active:")
print("   • Superposition - Teams exist in multiple states")
print("   • Entanglement - Performances are correlated")
print("   • Interference - Matchup effects amplify/cancel")
print("   • Tunneling - Upset probability floors")

# =============================================================================
# CELL 2: Create Sample Data (Skip if you have real data)
# =============================================================================

print("\n" + "=" * 60)
print("📊 GENERATING SAMPLE NFL DATA")
print("=" * 60)

# Generate 10 weeks of sample games
sample_games = generate_sample_nfl_games(n_weeks=10)
sample_games.to_csv("sample_nfl_games.csv", index=False)

print(f"Created sample_nfl_games.csv with {len(sample_games)} games")
print("\nSample games:")
print(sample_games.head(10).to_string(index=False))

# =============================================================================
# CELL 3: Calculate Team Strengths
# =============================================================================

print("\n" + "=" * 60)
print("💪 CALCULATING TEAM STRENGTHS")
print("=" * 60)

# Use your real data path here, or the sample data
GAMES_PATH = "sample_nfl_games.csv"  # Change to your data path

calc = NFLStrengthCalculator()
calc.load_games(GAMES_PATH)
team_strengths = calc.calculate_all_strengths()

print("\nTop 10 Teams by Offensive Efficiency:")
print(team_strengths.head(10)[['team', 'off_efficiency', 'def_efficiency', 
                                'momentum', 'point_differential']].to_string(index=False))

print("\nPower Rankings:")
rankings = calc.power_rankings()
print(rankings.head(15).to_string(index=False))

# =============================================================================
# CELL 4: Initialize Quantum Engine
# =============================================================================

print("\n" + "=" * 60)
print("🌌 INITIALIZING QUANTUM ENGINE")
print("=" * 60)

# Create engine with custom config
config = NFLQuantumConfig()
config.N_SIMULATIONS = 10000        # More sims = more accurate
config.ENTANGLEMENT_STRENGTH = 0.35  # How correlated performances are
config.TUNNELING_PROBABILITY = 0.08  # Upset floor

engine = NFLQuantumEngine(config)

# Load team strengths into engine
for _, row in team_strengths.iterrows():
    strength = NFLTeamStrength(
        team=row['team'],
        off_efficiency=row['off_efficiency'],
        def_efficiency=row['def_efficiency'],
        off_explosiveness=row.get('off_explosiveness', 1.0),
        turnover_rate=row.get('turnover_rate', 0.12),
        takeaway_rate=row.get('takeaway_rate', 0.12),
        momentum=row.get('momentum', 0.0),
        home_boost=row.get('home_boost', 0.03),
    )
    engine.add_team(strength)

print(f"Loaded {len(engine.teams)} teams into quantum engine")

# =============================================================================
# CELL 5: Single Game Prediction
# =============================================================================

print("\n" + "=" * 60)
print("🏈 SINGLE GAME PREDICTION")
print("=" * 60)

# Pick two teams (use teams from your data)
HOME_TEAM = "Kansas City Chiefs"
AWAY_TEAM = "Buffalo Bills"

# Check if teams exist
available_teams = list(engine.teams.keys())
if HOME_TEAM not in available_teams:
    HOME_TEAM = available_teams[0]
if AWAY_TEAM not in available_teams:
    AWAY_TEAM = available_teams[1]

print(f"\n🏟️ {AWAY_TEAM} @ {HOME_TEAM}")
print("-" * 50)

# Get prediction
prediction = engine.predict_game(
    home_team=HOME_TEAM,
    away_team=AWAY_TEAM,
    n_simulations=10000,
    weather='clear',
    primetime=False,
)

print(f"""
┌─────────────────────────────────────────────────┐
│              QUANTUM PREDICTION                 │
├─────────────────────────────────────────────────┤
│  {HOME_TEAM:20} Win Prob: {prediction['home_win_prob']:.1%}     │
│  {AWAY_TEAM:20} Win Prob: {prediction['away_win_prob']:.1%}     │
├─────────────────────────────────────────────────┤
│  Predicted Spread: {prediction['predicted_spread']:+.1f}                     │
│  Spread Std Dev:   {prediction['spread_std']:.1f}                      │
├─────────────────────────────────────────────────┤
│  Predicted Total:  {prediction['predicted_total']:.1f}                     │
│  Total Std Dev:    {prediction['total_std']:.1f}                      │
├─────────────────────────────────────────────────┤
│  Home Score:       {prediction['home_score_avg']:.1f} (median: {prediction['home_score_median']:.0f})   │
│  Away Score:       {prediction['away_score_avg']:.1f} (median: {prediction['away_score_median']:.0f})   │
└─────────────────────────────────────────────────┘
""")

print("🎲 Quantum State Distribution (Home Team):")
for state, prob in prediction['state_distribution'].items():
    bar = "█" * int(prob * 50)
    print(f"   {state:12}: {bar} {prob:.1%}")

# =============================================================================
# CELL 6: Spread and Total Analysis
# =============================================================================

print("\n" + "=" * 60)
print("📊 SPREAD & TOTAL ANALYSIS")
print("=" * 60)

# Analyze different spreads
spreads_to_check = [-7.0, -3.5, -3.0, 0, 3.0, 3.5, 7.0]

print(f"\n{AWAY_TEAM} @ {HOME_TEAM}")
print("\nSpread Analysis (negative = home favored):")
print("-" * 50)

for spread in spreads_to_check:
    home_cover_prob, away_cover_prob = engine.spread_probability(
        HOME_TEAM, AWAY_TEAM, spread, n_simulations=5000
    )
    
    favored = "HOME" if spread < 0 else "AWAY" if spread > 0 else "PICK"
    line_str = f"{spread:+.1f}" if spread != 0 else "PK"
    
    print(f"  {line_str:6} | Home covers: {home_cover_prob:.1%} | Away covers: {away_cover_prob:.1%}")

# Totals analysis
totals_to_check = [41.5, 44.5, 47.5, 50.5, 53.5]

print("\nTotal Analysis:")
print("-" * 50)

for total in totals_to_check:
    over_prob, under_prob = engine.over_under_probability(
        HOME_TEAM, AWAY_TEAM, total, n_simulations=5000
    )
    
    print(f"  {total:5.1f} | Over: {over_prob:.1%} | Under: {under_prob:.1%}")

# =============================================================================
# CELL 7: Weather Impact Analysis
# =============================================================================

print("\n" + "=" * 60)
print("🌦️ WEATHER IMPACT ANALYSIS")
print("=" * 60)

weather_conditions = ['clear', 'rain', 'snow', 'wind']

print(f"\n{AWAY_TEAM} @ {HOME_TEAM}")
print("\nHow weather affects the game:")
print("-" * 60)

for weather in weather_conditions:
    pred = engine.predict_game(HOME_TEAM, AWAY_TEAM, n_simulations=3000, weather=weather)
    
    emoji = {'clear': '☀️', 'rain': '🌧️', 'snow': '❄️', 'wind': '💨'}[weather]
    
    print(f"  {emoji} {weather:6} | Total: {pred['predicted_total']:5.1f} | "
          f"Spread: {pred['predicted_spread']:+5.1f} | Home: {pred['home_win_prob']:.1%}")

# =============================================================================
# CELL 8: Weekly Predictions (Multiple Games)
# =============================================================================

print("\n" + "=" * 60)
print("📅 WEEKLY PREDICTIONS")
print("=" * 60)

# Define this week's matchups (adjust to your actual games)
# Format: (home_team, away_team, spread_line, total_line)
WEEKLY_MATCHUPS = [
    (available_teams[0], available_teams[1], -3.0, 47.5),
    (available_teams[2], available_teams[3], -7.0, 44.5),
    (available_teams[4], available_teams[5], 2.5, 50.5),
]

print("\n📋 Weekly Picks:")
print("-" * 80)

for home, away, spread, total in WEEKLY_MATCHUPS:
    if home not in engine.teams or away not in engine.teams:
        continue
    
    pred = engine.predict_game(home, away, n_simulations=5000)
    
    # Spread pick
    home_cover, _ = engine.spread_probability(home, away, spread, n_simulations=3000)
    spread_pick = "HOME" if home_cover > 0.52 else "AWAY" if home_cover < 0.48 else "PASS"
    spread_edge = abs(home_cover - 0.5) * 2
    
    # Total pick
    over_prob, _ = engine.over_under_probability(home, away, total, n_simulations=3000)
    total_pick = "OVER" if over_prob > 0.52 else "UNDER" if over_prob < 0.48 else "PASS"
    total_edge = abs(over_prob - 0.5) * 2
    
    print(f"\n🏈 {away} @ {home}")
    print(f"   Win: {pred['home_win_prob']:.1%} home | Pred Spread: {pred['predicted_spread']:+.1f}")
    print(f"   📍 Spread {spread:+.1f}: {spread_pick} ({home_cover:.1%} home covers) [{spread_edge:.0%} edge]")
    print(f"   📍 Total {total}: {total_pick} ({over_prob:.1%} over) [{total_edge:.0%} edge]")

# =============================================================================
# CELL 9: Quantum State Deep Dive
# =============================================================================

print("\n" + "=" * 60)
print("🔬 QUANTUM STATE DEEP DIVE")
print("=" * 60)

# Analyze how often teams collapse into each state
print(f"\nAnalyzing quantum states for {HOME_TEAM}...")

state_counts = {s.value: 0 for s in TeamState}
n_samples = 1000

home_strength = engine.teams[HOME_TEAM]

for _ in range(n_samples):
    state, multiplier = home_strength.quantum_state.collapse()
    state_counts[state.value] += 1

print("\nState Probability Distribution:")
for state, count in sorted(state_counts.items(), key=lambda x: -x[1]):
    prob = count / n_samples
    bar = "█" * int(prob * 40)
    mult = home_strength.quantum_state.multipliers[TeamState(state)]
    print(f"  {state:12}: {bar} {prob:.1%} (×{mult:.2f})")

print("""
💡 Interpretation:
   • DOMINANT: Team plays at peak level (~125% efficiency)
   • ELEVATED: Above average performance (~110%)
   • BASELINE: Normal performance (100%)
   • DIMINISHED: Below average (~90%)
   • STRUGGLING: Poor performance (~75%)
   
   The model samples from this distribution each simulation,
   capturing the inherent uncertainty in how a team will perform.
""")

# =============================================================================
# CELL 10: Run Backtest
# =============================================================================

print("\n" + "=" * 60)
print("🔄 RUNNING BACKTEST")
print("=" * 60)

# Run backtest on sample data
backtest = NFLBacktestEngine(GAMES_PATH)
backtest.config.N_SIMULATIONS = 3000  # Fewer sims for speed

# Use later games as test set
results = backtest.run_backtest(
    start_date="2024-10-01",  # Adjust based on your data
    end_date="2024-12-01",
    verbose=True
)

if results:
    backtest.print_summary()
    
    # Show some individual results
    print("\n📋 Sample Predictions vs Actuals:")
    df = backtest.results_to_dataframe()
    print(df.head(10).to_string(index=False))
else:
    print("⚠️ No games in date range for backtest")

# =============================================================================
# CELL 11: Export Predictions
# =============================================================================

print("\n" + "=" * 60)
print("💾 EXPORT PREDICTIONS")
print("=" * 60)

from datetime import datetime

# Create predictions for all possible matchups
all_predictions = []

teams_list = list(engine.teams.keys())

for i, home in enumerate(teams_list[:8]):  # Limit for speed
    for away in teams_list[:8]:
        if home == away:
            continue
        
        pred = engine.predict_game(home, away, n_simulations=2000)
        
        all_predictions.append({
            'home_team': home,
            'away_team': away,
            'home_win_prob': pred['home_win_prob'],
            'predicted_spread': pred['predicted_spread'],
            'predicted_total': pred['predicted_total'],
            'spread_std': pred['spread_std'],
            'total_std': pred['total_std'],
        })

pred_df = pd.DataFrame(all_predictions)

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = f"nfl_predictions_{timestamp}.csv"
pred_df.to_csv(output_path, index=False)

print(f"✅ Saved {len(pred_df)} predictions to {output_path}")
print("\nSample predictions:")
print(pred_df.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("🏁 NOTEBOOK COMPLETE")
print("=" * 60)
print("""
Next Steps:
1. Replace sample_nfl_games.csv with real NFL data
2. Add actual betting lines for more accurate backtesting  
3. Tune quantum parameters (entanglement, tunneling, etc.)
4. Add player-level injury adjustments
5. Integrate weather data from API
""")
