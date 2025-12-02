# 🔮 QEPC v2.0 - Quantum Enhanced Prediction Calculator

A quantum-inspired sports prediction engine that models uncertainty using principles from quantum mechanics.

## 🎯 Core Philosophy

1. **Quantum-Inspired** - Not just labels, but actual mathematical models:
   - **Superposition**: Teams exist in multiple performance states until game time
   - **Entanglement**: Team performances are correlated (pace, game flow)
   - **Interference**: Matchup effects amplify or cancel
   - **Tunneling**: Ensures minimum upset probability (any given Sunday)

2. **Maximum Accuracy** - Designed to beat the spread:
   - Recency-weighted team strengths (recent games matter more)
   - Real volatility from game-to-game variance
   - Situational adjustments (home court, rest, injuries)
   - Proper backtesting with time-travel (no lookahead bias)

## 📁 Project Structure

```
qepc_v2/
├── qepc/                      # Main package
│   ├── core/                  # Quantum engine
│   │   ├── quantum.py         # Superposition, entanglement, tunneling
│   │   └── config.py          # All tunable parameters
│   ├── data/
│   │   └── loader.py          # CSV data loading
│   └── sports/nba/
│       ├── strengths.py       # Team ratings with recency
│       ├── predictor.py       # Main prediction engine
│       └── backtest.py        # Validation engine
├── notebooks/
│   ├── 01_daily_predictions.ipynb   # Today's picks
│   ├── 02_backtest.ipynb            # Validate accuracy
│   └── 03_data_refresh.ipynb        # Update CSVs (local only)
├── data/                      # Your existing data folder
│   ├── raw/                   # Historical data
│   ├── live/                  # Current season data
│   ├── props/                 # Player props data
│   └── results/               # Predictions & backtests
└── README.md
```

## 🚀 Quick Start

### 1. Setup

Copy the `qepc/` folder into your existing project:
```bash
# Your project should look like:
qepc_project/
├── qepc/          # NEW - copy this folder
├── data/          # Your existing data
├── notebooks/     # NEW - copy these notebooks
└── ...
```

### 2. Make Predictions

```python
from qepc import quick_predict, predict_today

# Predict a single game
pred = quick_predict("Boston Celtics", "Los Angeles Lakers")

# Predict all games today
predictions = predict_today()
```

### 3. Run Backtest

```python
from qepc import run_backtest

# Test accuracy over last 30 days
results = run_backtest(n_days=30)
```

## 📊 Data Requirements

QEPC works with your existing CSV files:

| File | Purpose | Location |
|------|---------|----------|
| `team_stats_live_nba_api.csv` | Current ORtg/DRtg/Pace | `data/live/` |
| `TeamStatistics.csv` | Game-by-game stats | `data/raw/` |
| `espn_scoreboard_today.csv` | Today's schedule | `data/live/` |
| `Injury_Overrides*.csv` | Injury impacts | `data/` |
| `GameResults_2025.csv` | Results for backtesting | `data/` |

## ⚙️ Configuration

All parameters are in `qepc/core/config.py`:

```python
# Key settings to tune:
home_court.base_advantage = 3.2      # Points
rest.b2b_penalty = 2.8               # Points off for back-to-back
quantum.entanglement_strength = 0.35 # Score correlation
quantum.tunneling_min_prob = 0.02    # Minimum upset chance (2%)
recency.half_life_days = 21          # Recent games weight decay
```

## 🔬 The Quantum Model

### Superposition
Teams exist in multiple performance states simultaneously:
- DOMINANT (+12%)
- ELEVATED (+5%)
- BASELINE (expected)
- DIMINISHED (-5%)
- STRUGGLING (-12%)

The probability of each state depends on team volatility, momentum, and situation.

### Entanglement
Team performances are negatively correlated - when one team exceeds expectations, the other tends to underperform. This is modeled with multivariate normal distributions.

### Interference
Matchup effects compound or cancel:
- Great offense vs bad defense = constructive interference (amplified)
- Great offense vs great defense = destructive interference (cancelled)

### Tunneling
Ensures minimum upset probability. Even 20-point favorites can lose - this prevents overconfident predictions.

## 📈 Expected Accuracy

Based on proper implementation:

| Metric | Target |
|--------|--------|
| Win Accuracy | 55-58% |
| Spread MAE | 9-11 pts |
| Brier Score | 0.20-0.22 |

## 🔧 Tuning for Accuracy

1. **Run backtest** to see current accuracy
2. **Check calibration** - do 60% picks win 60%?
3. **Adjust parameters** based on:
   - Spread bias → adjust home court advantage
   - Overconfidence → increase tunneling
   - Missing trends → decrease recency half-life

## 📱 iPad/Cloud Usage

The system is designed to work offline:
1. Run `03_data_refresh.ipynb` on your **local machine** to update CSVs
2. Sync files to cloud (GitHub, Dropbox, etc.)
3. Run predictions on iPad using the updated CSVs

## 🆚 What's New in v2.0

| Feature | v1.0 | v2.0 |
|---------|------|------|
| Volatility | Synthetic (constant) | Real (game-to-game std) |
| Recency | None | Exponential decay weighting |
| Quantum States | Labels only | Mathematical model |
| Entanglement | Basic correlation | Multivariate normal |
| Tunneling | None | Guaranteed upset floor |
| Backtesting | Lookahead bias | Proper time-travel |

## 📚 Files Reference

### Core Engine
- `quantum.py` - QuantumState, EntanglementEngine, TunnelingModel
- `config.py` - All tunable parameters with defaults

### NBA Module
- `strengths.py` - Team ratings with recency weighting
- `predictor.py` - Main GamePredictor class
- `backtest.py` - BacktestEngine for validation

### Notebooks
- `01_daily_predictions.ipynb` - Generate today's picks
- `02_backtest.ipynb` - Validate and analyze accuracy
- `03_data_refresh.ipynb` - Update data from APIs

---

Built with 🔮 quantum inspiration and 📊 rigorous validation.
