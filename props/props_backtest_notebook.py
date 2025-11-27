"""
QEPC Player Props Backtest Notebook
====================================

Copy each cell into a Jupyter notebook, or run this file directly.

Requirements:
- pandas
- numpy  
- scipy (optional, for better probability calculations)

Data Required:
- PlayerStatistics.csv in data/raw/ folder
"""

# =============================================================================
# CELL 1: Setup and Imports
# =============================================================================

import sys
from pathlib import Path

# Add the props folder to path
props_path = Path.cwd() / "props"
if props_path.exists() and str(props_path) not in sys.path:
    sys.path.insert(0, str(props_path))

# Also try parent paths
for parent in [Path.cwd(), Path.cwd().parent]:
    props_folder = parent / "props"
    if props_folder.exists() and str(props_folder) not in sys.path:
        sys.path.insert(0, str(props_folder))

import pandas as pd
import numpy as np
from datetime import date, timedelta

# Import our engines
try:
    from player_props_engine import PlayerPropsEngine, PropPrediction
    from props_backtest_engine import PropsBacktestEngine, BacktestSummary
    print("✅ Props engines loaded successfully!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure player_props_engine.py and props_backtest_engine.py are in the props/ folder")

# =============================================================================
# CELL 2: Configuration
# =============================================================================

# Path to your player statistics file
DATA_PATH = "data/raw/PlayerStatistics.csv"  # Adjust if needed

# Backtest date range
BACKTEST_START = "2024-11-01"  # Start date
BACKTEST_END = "2024-11-15"    # End date

# Props to test
PROPS_TO_TEST = ['PTS', 'REB', 'AST', '3PM', 'PRA']

# Minimum player minutes to include
MIN_MINUTES = 15.0

print(f"📊 Configuration:")
print(f"   Data: {DATA_PATH}")
print(f"   Date Range: {BACKTEST_START} → {BACKTEST_END}")
print(f"   Props: {PROPS_TO_TEST}")

# =============================================================================
# CELL 3: Quick Test - Single Player Prediction
# =============================================================================

print("=" * 60)
print("🧪 QUICK TEST: Single Player Prediction")
print("=" * 60)

# Load engine (no cutoff = use all available data)
engine = PlayerPropsEngine(DATA_PATH)
engine.load_data()

# Test a prediction
test_player = "LeBron James"  # Change to any player in your data

print(f"\n📈 Predictions for {test_player}:")
print("-" * 50)

for prop in ['PTS', 'REB', 'AST', 'PRA']:
    pred = engine.predict(test_player, prop, opponent="Warriors", is_home=True)
    
    if pred:
        print(f"\n{prop}:")
        print(f"  Projection: {pred.projection}")
        print(f"  Range: {pred.floor} - {pred.ceiling}")
        print(f"  Confidence: {pred.confidence}")
        print(f"  Over 25.5: {pred.over_prob(25.5):.1%}" if prop == 'PTS' else "")
    else:
        print(f"\n{prop}: Player not found")

# =============================================================================
# CELL 4: Run Full Backtest
# =============================================================================

print("\n" + "=" * 60)
print("🚀 RUNNING FULL BACKTEST")
print("=" * 60)

# Initialize backtest engine
backtest = PropsBacktestEngine(DATA_PATH)

# Run backtest
results = backtest.run_backtest(
    start_date=BACKTEST_START,
    end_date=BACKTEST_END,
    props=PROPS_TO_TEST,
    min_minutes=MIN_MINUTES,
    verbose=True,
)

print(f"\n✅ Backtest complete: {len(results)} predictions generated")

# =============================================================================
# CELL 5: Backtest Summary
# =============================================================================

print("\n" + "=" * 60)
print("📊 BACKTEST SUMMARY")
print("=" * 60)

if results:
    summary = backtest.get_summary()
    
    print(f"""
┌─────────────────────────────────────────────┐
│           OVERALL PERFORMANCE               │
├─────────────────────────────────────────────┤
│  Total Predictions:    {summary.total_predictions:>6}              │
│  Mean Absolute Error:  {summary.mean_absolute_error:>6.2f} pts          │
│  Median Abs Error:     {summary.median_absolute_error:>6.2f} pts          │
│  Mean % Error:         {summary.mean_pct_error*100:>6.1f}%             │
├─────────────────────────────────────────────┤
│           DIRECTIONAL ACCURACY              │
├─────────────────────────────────────────────┤
│  Overall Accuracy:     {summary.overall_accuracy*100:>6.1f}%             │
│  Over Accuracy:        {summary.over_accuracy*100:>6.1f}%             │
│  Under Accuracy:       {summary.under_accuracy*100:>6.1f}%             │
├─────────────────────────────────────────────┤
│           CALIBRATION                       │
├─────────────────────────────────────────────┤
│  Brier Score:          {summary.brier_score:>6.4f}             │
│  (Lower is better, 0.25 = random)           │
├─────────────────────────────────────────────┤
│           BY CONFIDENCE                     │
├─────────────────────────────────────────────┤
│  HIGH Confidence:      {summary.high_conf_accuracy*100:>6.1f}%             │
│  MEDIUM Confidence:    {summary.medium_conf_accuracy*100:>6.1f}%             │
│  LOW Confidence:       {summary.low_conf_accuracy*100:>6.1f}%             │
├─────────────────────────────────────────────┤
│           SIMULATED BETTING                 │
├─────────────────────────────────────────────┤
│  Bets Placed:          {summary.simulated_bets:>6}              │
│  Bets Won:             {summary.simulated_wins:>6}              │
│  ROI:                  {summary.simulated_roi*100:>6.1f}%             │
└─────────────────────────────────────────────┘
    """)
else:
    print("❌ No results to summarize")

# =============================================================================
# CELL 6: Breakdown by Prop Type
# =============================================================================

print("\n" + "=" * 60)
print("📈 BREAKDOWN BY PROP TYPE")
print("=" * 60)

if results:
    prop_breakdown = backtest.breakdown_by_prop()
    print("\n")
    print(prop_breakdown.to_string())
    
    print("\n💡 Interpretation:")
    print("   MAE = Mean Absolute Error (lower is better)")
    print("   MAPE = Mean Absolute Percentage Error")
    print("   Accuracy = % of correct over/under predictions")
    print("   Brier = Probability calibration (lower is better, 0.25 = random)")

# =============================================================================
# CELL 7: Breakdown by Confidence Level
# =============================================================================

print("\n" + "=" * 60)
print("🎯 BREAKDOWN BY CONFIDENCE LEVEL")
print("=" * 60)

if results:
    conf_breakdown = backtest.breakdown_by_confidence()
    print("\n")
    print(conf_breakdown.to_string())
    
    print("\n💡 Interpretation:")
    print("   HIGH confidence predictions should have higher accuracy")
    print("   If LOW > HIGH, the confidence scoring needs adjustment")

# =============================================================================
# CELL 8: Detailed Results DataFrame
# =============================================================================

print("\n" + "=" * 60)
print("📋 DETAILED RESULTS")
print("=" * 60)

if results:
    df_results = backtest.results_to_dataframe()
    
    print(f"\nTotal records: {len(df_results)}")
    print("\nSample of results:")
    display_cols = ['player', 'date', 'prop', 'projection', 'actual', 'line', 
                    'over_prob', 'correct_side', 'abs_error']
    print(df_results[display_cols].head(15).to_string(index=False))

# =============================================================================
# CELL 9: Biggest Misses Analysis
# =============================================================================

print("\n" + "=" * 60)
print("🎯 BIGGEST MISSES (Largest Errors)")
print("=" * 60)

if results:
    df_results = backtest.results_to_dataframe()
    
    biggest_misses = df_results.nlargest(10, 'abs_error')
    
    print("\n⚠️ Top 10 Largest Prediction Errors:")
    print("-" * 80)
    
    for _, row in biggest_misses.iterrows():
        direction = "OVER" if row['actual'] > row['projection'] else "UNDER"
        print(f"{row['player']:20} | {row['prop']:4} | "
              f"Proj: {row['projection']:5.1f} | Actual: {row['actual']:5.1f} | "
              f"Error: {row['abs_error']:5.1f} | Went {direction}")

# =============================================================================
# CELL 10: Best Predictions Analysis
# =============================================================================

print("\n" + "=" * 60)
print("✅ BEST PREDICTIONS (Most Accurate)")
print("=" * 60)

if results:
    df_results = backtest.results_to_dataframe()
    
    # Filter to confident predictions
    confident = df_results[df_results['confidence'].isin(['HIGH', 'MEDIUM'])]
    
    # Find most accurate (lowest error)
    best_preds = confident.nsmallest(10, 'abs_error')
    
    print("\n🎯 Top 10 Most Accurate Predictions:")
    print("-" * 80)
    
    for _, row in best_preds.iterrows():
        print(f"{row['player']:20} | {row['prop']:4} | "
              f"Proj: {row['projection']:5.1f} | Actual: {row['actual']:5.1f} | "
              f"Error: {row['abs_error']:4.1f} | {row['confidence']}")

# =============================================================================
# CELL 11: Calibration Analysis
# =============================================================================

print("\n" + "=" * 60)
print("📐 PROBABILITY CALIBRATION ANALYSIS")
print("=" * 60)

if results:
    calibration = backtest.calibration_analysis(bins=5)
    
    print("\n")
    print(calibration.to_string())
    
    print("""
💡 Interpretation:
   'Predicted' = Average predicted probability in this bin
   'ActualOverRate' = Actual rate the over hit
   
   Well-calibrated model: Predicted ≈ ActualOverRate
   - If ActualOverRate > Predicted → Model is under-confident
   - If ActualOverRate < Predicted → Model is over-confident
    """)

# =============================================================================
# CELL 12: Export Results
# =============================================================================

print("\n" + "=" * 60)
print("💾 EXPORT RESULTS")
print("=" * 60)

if results:
    from datetime import datetime
    
    # Create results folder
    results_dir = Path("data/results/props_backtests")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"props_backtest_{BACKTEST_START}_to_{BACKTEST_END}_{timestamp}.csv"
    output_path = results_dir / filename
    
    # Export
    df_results = backtest.results_to_dataframe()
    df_results.to_csv(output_path, index=False)
    
    print(f"✅ Results exported to: {output_path}")
    print(f"   Total records: {len(df_results)}")

# =============================================================================
# CELL 13: Edge Analysis (Find Best Betting Opportunities)
# =============================================================================

print("\n" + "=" * 60)
print("💰 EDGE ANALYSIS (Best Betting Opportunities)")
print("=" * 60)

if results:
    df_results = backtest.results_to_dataframe()
    
    # Calculate edge
    df_results['edge'] = abs(df_results['over_prob'] - 0.5)
    df_results['predicted_side'] = df_results['over_prob'].apply(
        lambda x: 'OVER' if x > 0.5 else 'UNDER'
    )
    
    # Find high-edge correct predictions
    high_edge = df_results[df_results['edge'] >= 0.10]  # 10%+ edge
    
    if not high_edge.empty:
        accuracy = high_edge['correct_side'].mean()
        
        print(f"\n🎯 High-Edge Predictions (10%+ edge):")
        print(f"   Total: {len(high_edge)}")
        print(f"   Accuracy: {accuracy:.1%}")
        
        if accuracy > 0.55:
            print("   ✅ Strong edge! Model performs well on confident predictions.")
        elif accuracy > 0.52:
            print("   👍 Slight edge. Model beats random on confident picks.")
        else:
            print("   ⚠️ No clear edge. Consider adjusting confidence thresholds.")
        
        print("\n📋 Sample high-edge predictions:")
        sample = high_edge.nlargest(10, 'edge')
        for _, row in sample.iterrows():
            hit = "✓" if row['correct_side'] else "✗"
            print(f"   {hit} {row['player']:18} {row['prop']:4} | "
                  f"Line: {row['line']:5.1f} | {row['predicted_side']:5} @ {row['over_prob']:.1%}")
    else:
        print("No high-edge predictions found in this sample.")

# =============================================================================
# CELL 14: Player-Level Analysis
# =============================================================================

print("\n" + "=" * 60)
print("👤 PLAYER-LEVEL ANALYSIS")
print("=" * 60)

if results:
    df_results = backtest.results_to_dataframe()
    
    # Group by player
    player_stats = df_results.groupby('player').agg({
        'abs_error': 'mean',
        'correct_side': ['mean', 'count'],
    }).round(3)
    
    player_stats.columns = ['MAE', 'Accuracy', 'Predictions']
    player_stats = player_stats[player_stats['Predictions'] >= 5]  # Min 5 predictions
    
    print("\n🏆 Most Predictable Players (Lowest Error):")
    most_predictable = player_stats.nsmallest(10, 'MAE')
    print(most_predictable.to_string())
    
    print("\n⚠️ Least Predictable Players (Highest Error):")
    least_predictable = player_stats.nlargest(10, 'MAE')
    print(least_predictable.to_string())

print("\n" + "=" * 60)
print("🏁 BACKTEST COMPLETE")
print("=" * 60)
