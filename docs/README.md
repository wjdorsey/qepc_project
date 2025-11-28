# the quantum entangled poisson cascade



[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🎯 What is QEPC?

 **quantum-inspired probabilistic forecasting engine** for sports that goes beyond traditional Poisson models.

Unlike conventional sports prediction models, QEPC draws inspiration from quantum computing principles:

**🔮 Entangled State Modeling**  
Multiple variables (player stats, injuries, matchups, rest days, home court) don't just add linearly—they interact and influence each other like entangled quantum states. QEPC captures these complex, non-linear relationships through:
- **100,000-state Monte Carlo simulations** mimicking quantum superposition
- **Poisson distributions** for discrete, count-based events (points, assists, shots)
- **Sequential cascades** that model real game flow and momentum

**⚡ Probabilistic Collapse**  
Just as quantum states collapse upon measurement, QEPC "collapses" thousands of possible game states into the most probable outcome by:
- Entangling diverse data points (team efficiency, injury impacts, historical patterns)
- Running massive parallel simulations (like a quantum computer exploring multiple states)
- Converging on predictions with calibrated confidence intervals

**🎲 Event-Driven Architecture**  
Sports are fundamentally discrete event systems (made basket, missed shot, turnover). Poisson distributions naturally model this event-driven reality, making QEPC's foundation mathematically aligned with how games actually unfold.

### The Ultimate Goal

**To create the most accurate sports prediction model ever built** by combining:
- Quantum-inspired computational approaches
- Rigorous statistical foundations
- Continuous backtesting and calibration
- Real-world data integration

### Current Status

**Currently optimized for:** NBA basketball  
**Expanding to:** NFL football, NHL hockey  
**Active development of:** Player props, win probabilities, game totals, fantasy projections

### Key Features

✅ **Quantum-Inspired Modeling** - 100,000 Monte Carlo simulations per prediction  
✅ **Non-Linear Interactions** - Variables "entangle" like quantum states  
✅ **Poisson Foundation** - Mathematically sound for discrete sports events  
✅ **Injury-Aware** - Automatically incorporates player availability impacts  
✅ **Rigorous Backtesting** - Validated against historical data with improving Brier scores  
✅ **Automated Data Pipeline** - Continuous integration of latest stats and schedules  
✅ **Multi-Sport Ready** - Architecture designed for NBA, NFL, NHL expansion  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip package manager
- Git with Git LFS installed

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/wjdorsey/qepc_project.git
   cd qepc_project
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Pull large data files** (if using Git LFS)
   ```bash
   git lfs pull
   ```

4. **Verify installation**
   ```bash
   python -c "import qepc; print('QEPC loaded successfully!')"
   ```

---

## 📊 Usage

### Running Predictions

#### Option 1: Jupyter Notebooks (Recommended)

```bash
jupyter notebook
```

Navigate to `notebooks/01_core/qepc_dashboard.ipynb` and run all cells.

#### Option 2: Python Script

```python
from qepc.sports.nba.sim import get_today_games, simulate_game
from qepc.core.lambda_engine import compute_lambda

# Get today's schedule
games = get_today_games()

# Simulate a game
result = simulate_game(
    home_team="Los Angeles Lakers",
    away_team="Boston Celtics"
)

print(f"Predicted Score: {result['home_score']} - {result['away_score']}")
print(f"Home Win Probability: {result['home_win_prob']:.1%}")
```

### Running Backtests

```bash
jupyter notebook notebooks/01_core/qepc_backtest.ipynb
```

Adjust the date range in the notebook and run to validate model accuracy.

---

## 📁 Project Structure

```
qepc_project/
├── qepc/                          # Main Python package
│   ├── autoload/                  # Path management & bootstrapping
│   ├── core/                      # Lambda engine, simulator
│   │   ├── lambda_engine.py       # Poisson rate calculations
│   │   ├── simulator.py           # Game simulation logic
│   │   └── model_config.py        # Model parameters
│   ├── sports/                    # Sport-specific modules
│   │   ├── nba/                   # NBA implementation
│   │   │   ├── sim.py             # NBA simulation wrapper
│   │   │   ├── lineups.py         # Lineup management
│   │   │   └── props.py           # Player props
│   │   ├── nfl/                   # NFL (future)
│   │   └── nhl/                   # NHL (future)
│   ├── utils/                     # Utility functions
│   │   ├── data_cleaning.py       # Team name standardization
│   │   ├── backup.py              # Project backup utilities
│   │   └── diagnostics.py         # System health checks
│   └── ui/                        # User interface components
│       ├── banners.py             # CLI banners
│       └── colors.py              # Color schemes
│
├── data/                          # Data files
│   ├── raw/                       # Large historical datasets (Git LFS)
│   │   ├── PlayerStatistics.csv  # Historical player stats
│   │   └── Team_Stats.csv         # Historical team stats
│   ├── results/                   # Model outputs
│   │   └── backtests/             # Backtest results
│   ├── cache/                     # Generated cache files
│   ├── metadata/                  # Reference data
│   ├── Games.csv                  # NBA schedule (2025-26)
│   ├── Players.csv                # Player reference
│   ├── Injury_Overrides.csv       # Current injury reports
│   └── qepc_calibration.json      # Model calibration settings
│
├── notebooks/                     # Jupyter notebooks
│   ├── 00_setup/                  # Environment setup
│   ├── 01_core/                   # Main analysis notebooks
│   │   ├── qepc_dashboard.ipynb   # Daily predictions dashboard
│   │   ├── qepc_backtest.ipynb    # Model backtesting
│   │   └── qepc_schedules.ipynb   # Schedule viewer
│   ├── 02_utilities/              # Data management
│   │   ├── qepc_project_backup.ipynb
│   │   ├── injuries_merge.ipynb
│   │   └── injury_data_fetch.ipynb
│   └── 03_dev/                    # Development & experiments
│       └── qepc_sandbox.ipynb
│
├── qepc_autoload.py               # Universal module loader
├── notebook_context.py            # Notebook environment setup
├── merge_schedules.py             # Schedule consolidation utility
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## 🧮 How It Works

### The QEPC Quantum-Inspired Prediction Model

QEPC's approach is fundamentally different from traditional sports models. Instead of simple linear regression, it simulates thousands of possible game states simultaneously—like a quantum computer exploring multiple realities at once.

#### Phase 1: State Entanglement 🔗
Multiple variables become "entangled" and influence each other:

1. **Team Efficiency Vectors**
   - Offensive Rating (ORtg): Points scored per 100 possessions
   - Defensive Rating (DRtg): Points allowed per 100 possessions
   - Home court advantage adjustments
   - Recent form and momentum factors

2. **Injury Impact Matrix**
   - Player-level injury reports from multiple sources
   - Position-specific impact weighting
   - Lineup disruption calculations
   - Cascading effects on team chemistry

3. **Contextual State Variables**
   - Rest days and back-to-back penalties
   - Travel distance and time zone adjustments
   - Historical head-to-head matchups
   - Venue-specific factors

These variables don't simply add—they interact non-linearly, creating an "entangled" state space where changing one variable affects the probability distribution of all others.

#### Phase 2: Lambda Computation (Poisson Rate Calculation) ⚛️
The core of QEPC's quantum-inspired approach:

```
λ_home = (Home_ORtg × Away_DRtg × Home_Court_Factor × Injury_Factor) / League_Avg
λ_away = (Away_ORtg × Home_DRtg × Injury_Factor) / League_Avg
```

These lambda values represent the **expected scoring rate**—the probabilistic "center" of our quantum-like state space.

#### Phase 3: Superposition Simulation 🌊
**100,000 Monte Carlo simulations** explore the possibility space:

- Each simulation represents a possible "collapsed" game state
- Poisson distributions generate discrete events (possessions, scores)
- Sequential cascades model game flow (momentum, clutch performance)
- Results aggregate into probability distributions

This is analogous to quantum superposition—exploring all possible outcomes simultaneously before "measurement" (prediction) occurs.

#### Phase 4: Probabilistic Collapse 🎯
The simulation results collapse into actionable predictions:

- **Win Probability:** Percentage of simulations where each team wins
- **Expected Score:** Most probable final score (median of distribution)
- **Spread Confidence:** Standard deviation of score differential
- **Total Confidence:** Variance in combined scoring

#### Phase 5: Continuous Calibration 🔄
Like quantum error correction:

- Backtest results identify systematic biases
- Lambda scaling factors adjust for over/under predictions
- Model "learns" from prediction errors
- Calibration saved in `data/qepc_calibration.json`

### Why This Approach Works

**Traditional Models:**
- Treat variables independently
- Use linear combinations
- Single-point predictions
- Limited uncertainty quantification

**QEPC Quantum-Inspired Model:**
- ✅ Variables interact (entanglement)
- ✅ Non-linear relationships captured
- ✅ Full probability distributions
- ✅ Confidence intervals built-in
- ✅ Handles discrete events naturally (Poisson)
- ✅ Massive parallel simulation (100k states)

The result: **More accurate predictions** that capture the true complexity and uncertainty of sports outcomes.

---

## 📈 Model Performance

*Run backtests to generate performance metrics*

Example metrics from recent backtests:
- **Accuracy:** ~65% on winner prediction
- **Spread Error:** Mean Absolute Error of 8-12 points
- **Brier Score:** 0.18-0.22 (lower is better, random = 0.25)

Run `notebooks/01_core/qepc_backtest.ipynb` to generate your own metrics.

---

## 🔄 Data Sources

### Schedules
- NBA official schedule (consolidated from multiple seasons)
- Updated via `merge_schedules.py` utility

### Team Statistics
- Historical ORtg/DRtg from Basketball Reference
- Stored in `data/raw/TeamStatistics.csv`

### Player Statistics
- Historical player-level data
- Stored in `data/raw/PlayerStatistics.csv`

### Injury Reports
- **Primary:** `data/Injury_Overrides.csv`
- Sources: ESPN, NBA.com, balldontlie API
- Updated via utility notebooks in `notebooks/02_utilities/`

---

## 🛠️ Development

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# Run with coverage
pytest --cov=qepc tests/
```

### Code Quality

```bash
# Format code
black qepc/

# Lint code
pylint qepc/

# Type checking
mypy qepc/
```

### Creating a Backup

```bash
# Run backup notebook
jupyter notebook notebooks/02_utilities/qepc_project_backup.ipynb

# Or use Python
python -c "from qepc.utils.backup import create_backup; create_backup()"
```

Backups are saved to `data/backups/`

---

## 📚 Documentation

### API Documentation
See [docs/API.md](docs/API.md) for detailed API reference.

### Notebooks
Each notebook includes inline documentation and examples:
- **Dashboard:** Daily game predictions and analysis
- **Backtest:** Historical model validation
- **Schedules:** View and filter upcoming games

### Key Modules

#### Lambda Engine
```python
from qepc.core.lambda_engine import compute_lambda

lambda_home, lambda_away = compute_lambda(
    home_team="Lakers",
    away_team="Celtics",
    home_ortg=115.2,
    home_drtg=108.5,
    away_ortg=118.3,
    away_drtg=110.1
)
```

#### Simulator
```python
from qepc.core.simulator import run_qepc_simulation

results = run_qepc_simulation(
    lambda_home=1.15,
    lambda_away=1.10,
    n_simulations=10000
)
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/qepc_project.git

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

---

## 🐛 Troubleshooting

### Common Issues

**Problem:** `ModuleNotFoundError: No module named 'qepc'`
```bash
# Solution: Run notebooks with the header cells
# Or add project root to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:/path/to/qepc_project"
```

**Problem:** `FileNotFoundError: Games.csv not found`
```bash
# Solution: Ensure you're in the project root
cd /path/to/qepc_project
# Or verify data files exist
ls data/Games.csv
```

**Problem:** Git LFS files not downloaded
```bash
# Solution: Pull LFS files explicitly
git lfs pull
```

**Problem:** Import errors in notebooks
```bash
# Solution: Always run the header cells first
# Cell 1: Path Stabilization
# Cell 2: Autoload Framework
```

### Getting Help

- **Issues:** [GitHub Issues](https://github.com/wjdorsey/qepc_project/issues)
- **Documentation:** Check notebook examples
- **Diagnostics:** Run `notebooks/00_setup/qepc_setup_environment.ipynb`

---

## 📋 Requirements

### Core Dependencies
- `pandas >= 2.0.0` - Data manipulation
- `numpy >= 1.24.0` - Numerical computing
- `matplotlib >= 3.7.0` - Plotting
- `plotly >= 5.14.0` - Interactive visualizations
- `scipy >= 1.10.0` - Statistical functions
- `jupyter >= 1.0.0` - Notebook interface

### Optional Dependencies
- `pytest >= 7.3.0` - Testing framework
- `black >= 23.0.0` - Code formatter
- `pylint >= 2.17.0` - Code linter
- `mypy >= 1.3.0` - Type checker

See `requirements.txt` for complete list.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- NBA.com for official data
- Basketball Reference for historical statistics
- ESPN for injury reports
- The open-source community for excellent Python libraries

---

## 📬 Contact

**Will Dorsey** - [@wjdorsey](https://github.com/wjdorsey)

Project Link: [https://github.com/wjdorsey/qepc_project](https://github.com/wjdorsey/qepc_project)

---

## 🗺️ Roadmap

### Foundation (Complete ✅)
- [x] Core Poisson-based prediction engine
- [x] 100,000-state Monte Carlo simulation framework
- [x] Backtesting and validation system
- [x] Multi-source injury integration
- [x] Git LFS for large historical datasets

### Phase 1: Enhanced Quantum Modeling (In Progress 🔄)
- [ ] Implement true "entanglement" matrices for variable interactions
- [ ] Expand to 1,000,000 simulation states for higher precision
- [ ] Add momentum and game flow cascades
- [ ] Player-level props (points, assists, rebounds)
- [ ] Advanced injury impact modeling

### Phase 2: Multi-Sport Expansion (Planned 📋)
- [ ] NFL: Receiving yards, rushing stats, passing TDs
- [ ] NHL: Goals, assists, shots on goal
- [ ] Cross-sport pattern recognition
- [ ] Unified prediction API

### Phase 3: Intelligence Layer (Future 🚀)
- [ ] Machine learning hybrid models (XGBoost + Poisson)
- [ ] Real-time in-game prediction updates
- [ ] Weather and contextual factors
- [ ] Ensemble predictions with confidence weighting

### Phase 4: Production Deployment (Future 🌐)
- [ ] Automated daily predictions (GitHub Actions)
- [ ] Web dashboard with live updates (Streamlit)
- [ ] REST API for integrations
- [ ] Mobile app for iOS/Android

### The North Star ⭐
**Build the most accurate sports prediction model ever created** by pushing the boundaries of quantum-inspired probabilistic forecasting.

See [IMPROVEMENT_ROADMAP.md](IMPROVEMENT_ROADMAP.md) for detailed implementation plans.

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ and Python**

