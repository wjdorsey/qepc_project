# 🏈 QEPC NFL Quantum Prediction Model

A quantum-inspired NFL prediction model that goes beyond basic Poisson distributions.

## 🌌 Quantum Concepts Applied

This isn't just another statistical model - it incorporates quantum mechanical analogies to better capture the inherent uncertainty and chaos of NFL games.

### 1. **Superposition**
Teams exist in multiple "states" simultaneously until gametime:
- **DOMINANT** (×1.25) - Playing at peak level
- **ELEVATED** (×1.10) - Above average 
- **BASELINE** (×1.00) - Normal performance
- **DIMINISHED** (×0.90) - Below average
- **STRUGGLING** (×0.75) - Poor performance

Each simulation "collapses" the team into one of these states based on probability amplitudes.

### 2. **Entanglement**
Performances are correlated:
- When offense excels, defense may tire (negative correlation)
- Turnovers affect both teams' scoring
- Game flow creates interdependencies

### 3. **Interference**
Matchup effects create constructive or destructive interference:
- **Constructive**: Elite offense vs. bad defense = amplified scoring
- **Destructive**: Elite offense vs. elite defense = effects cancel

### 4. **Tunneling**
Even heavy underdogs have a non-zero upset probability:
- Models the "any given Sunday" phenomenon
- Creates fat tails in the distribution
- Base floor: ~8% upset probability

### 5. **Decoherence**
Environmental factors reduce quantum effects:
- Weather reduces passing efficiency
- Dome teams have different state distributions
- Prime time games increase variance

## 📁 Files Included

| File | Description |
|------|-------------|
| `nfl_quantum_engine.py` | Core quantum prediction engine |
| `nfl_strengths.py` | Team strength calculator |
| `nfl_backtest_engine.py` | Backtesting framework |
| `nfl_quantum_notebook.py` | Interactive notebook (Python) |
| `qepc_nfl_quantum.ipynb` | Jupyter notebook version |

## 🚀 Quick Start

### 1. Setup

```
your_project/
├── nfl/
│   ├── nfl_quantum_engine.py
│   ├── nfl_strengths.py
│   ├── nfl_backtest_engine.py
│   └── qepc_nfl_quantum.ipynb
└── data/
    └── nfl_games.csv  ← Your game data
```

### 2. Data Format

Your `nfl_games.csv` should have:
```csv
game_date,home_team,away_team,home_score,away_score
2024-09-08,Kansas City Chiefs,Baltimore Ravens,27,20
2024-09-08,Buffalo Bills,Arizona Cardinals,34,28
...
```

### 3. Run Predictions

```python
from nfl_quantum_engine import NFLQuantumEngine, NFLTeamStrength
from nfl_strengths import NFLStrengthCalculator

# Calculate team strengths from historical data
calc = NFLStrengthCalculator()
calc.load_games("data/nfl_games.csv")
strengths = calc.calculate_all_strengths()

# Initialize engine
engine = NFLQuantumEngine()

# Load teams
for _, row in strengths.iterrows():
    engine.add_team(NFLTeamStrength(
        team=row['team'],
        off_efficiency=row['off_efficiency'],
        def_efficiency=row['def_efficiency'],
        momentum=row['momentum'],
    ))

# Predict a game
result = engine.predict_game(
    home_team="Kansas City Chiefs",
    away_team="Buffalo Bills",
    weather='clear',
    primetime=True,
)

print(f"Chiefs win: {result['home_win_prob']:.1%}")
print(f"Spread: {result['predicted_spread']:+.1f}")
print(f"Total: {result['predicted_total']:.1f}")
```

## 🎛️ Tunable Parameters

```python
from nfl_quantum_engine import NFLQuantumConfig

config = NFLQuantumConfig()

# Quantum parameters
config.ENTANGLEMENT_STRENGTH = 0.35   # 0-1, how correlated performances are
config.TUNNELING_PROBABILITY = 0.08   # Base upset probability floor
config.INTERFERENCE_FACTOR = 0.20     # Matchup amplification strength

# NFL parameters
config.HOME_FIELD_ADVANTAGE = 0.03    # ~3% home boost
config.AVG_DRIVES_PER_TEAM = 11.5     # League average

# Weather decoherence
config.WEATHER_IMPACT = {
    'dome': 1.0,
    'clear': 1.0,
    'rain': 0.85,    # 15% passing reduction
    'snow': 0.75,
    'wind': 0.80,
}
```

## 📊 Drive-Based Simulation

Unlike NBA models that use continuous scoring, this NFL model simulates **individual drives**:

```
Drive Outcomes (baseline):
- Touchdown:     22%  (6-7 pts)
- Field Goal:    18%  (3 pts)
- Punt:          38%  (0 pts)
- Turnover:      12%  (0 pts, possession change)
- Defensive TD:   2%  (7 pts for defense)
- Safety:         1%  (2 pts)
```

This captures the discrete nature of NFL scoring (TDs = 6-7, FGs = 3, etc.)

## 🧪 Backtesting

```python
from nfl_backtest_engine import NFLBacktestEngine

backtest = NFLBacktestEngine("data/nfl_games.csv")
results = backtest.run_backtest("2024-09-01", "2024-11-15")
backtest.print_summary()

# Output:
# Straight Up Accuracy: 62.3%
# ATS Accuracy: 54.2%
# Over/Under Accuracy: 51.8%
# Brier Score: 0.2134
```

## 🎯 Key Features

| Feature | Description |
|---------|-------------|
| **Superposition States** | Teams can be in 5 different performance states |
| **State Collapse** | Each simulation randomly collapses to a state |
| **Entangled Scoring** | Off/def performances are negatively correlated |
| **Tunneling** | Upsets have a probability floor |
| **Weather Effects** | Bad weather compresses scoring distributions |
| **Drive Simulation** | Models actual drive outcomes, not just totals |
| **Momentum** | Recent form shifts state probabilities |

## 💡 Why Quantum?

Traditional sports models treat each variable as independent. But NFL games have:

1. **Inherent uncertainty** - The same team can look elite or terrible week to week
2. **Correlated outcomes** - If one team dominates TOP, the other can't
3. **Non-linear matchups** - Great offense + great defense ≠ average game
4. **Fat tails** - Upsets happen more than normal distributions predict

Quantum-inspired models naturally capture these properties through:
- Superposition (multiple possible team states)
- Entanglement (correlated variables)
- Interference (non-linear combinations)
- Tunneling (fat tails)

## 📈 Sample Output

```
🏈 Buffalo Bills @ Kansas City Chiefs
──────────────────────────────────────
Win Probability:
  Chiefs: 58.2%
  Bills:  41.8%

Predicted Score: Chiefs 27.3, Bills 24.1
Predicted Spread: KC -3.2
Predicted Total: 51.4

Quantum State Distribution (Chiefs):
  dominant:    ████████ 16.2%
  elevated:    ████████████ 24.1%
  baseline:    ████████████████ 32.3%
  diminished:  ██████████ 20.1%
  struggling:  ████ 7.3%
```

## 🛠️ Future Enhancements

- [ ] Player-level injury impacts
- [ ] Real-time weather API integration
- [ ] Vegas line comparison
- [ ] Prop bet predictions (player TDs, passing yards)
- [ ] Live game probability updates
- [ ] Conference/division rivalry adjustments

## 📝 License

Part of the QEPC (Quantum Entangled Poisson Cascade) project.
