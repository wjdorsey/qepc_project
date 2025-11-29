# QEPC Instruction Manual (README)

## Introduction to QEPC
QEPC (Quantum Enhanced Prediction Core) is your quantum-inspired NBA prediction engine, designed to make the most accurate sports model ever. What makes it unique? It's inspired by quantum computers: 
- **Superposition**: Thinks about multiple "what if" game outcomes at once (like random noise in simulations for uncertainties).
- **Entanglement**: Links factors like injuries, lineups, and team strengths in smart ways (e.g., how one player's status affects the whole team).
- **Interference**: Combines predictions from different models (classical + quantum-like) for nuanced results.

This isn't full quantum computing (no special hardware needed)—it's "quantum-inspired" ML running on your laptop, using tools like PennyLane for probabilistic boosts. Goal: 65-70% win accuracy, low spread errors, using 10+ years of data, live APIs, and advanced stats.

Built for beginners: Run notebooks in JupyterLab. Each is like a recipe—follow cells top to bottom (Shift+Enter to run).

## Setup and Installation
1. **Install JupyterLab**: If not installed, open terminal/command prompt: `pip install jupyterlab`.
2. **Clone Repo**: Download from GitHub (green "Code" button > Download ZIP) or `git clone https://github.com/wjdorsey/qepc_project.git`.
3. **Open in JupyterLab**: Navigate to folder, run `jupyter lab`.
4. **Install Libraries**: In a notebook cell: `!pip install nba_api pennylane requests scipy numpy pandas`.
5. **Data**: Put your large CSVs (10-year history) in `data/raw/`. For live, APIs are set up—no extra keys needed except your Balldontlie one (add to code if using).

Run `00_qepc_project_hub.ipynb` first—it checks everything and warns if setup issues.

## Notebook Guide: What to Run, In What Order
Start with basics, then data fetch, then predictions/backtesting. Order for new users:
1. **00_qepc_project_hub.ipynb** (Hub/Setup): Run this EVERY time you start. Checks environment, shows today's games (live API), quick predictions with quantum noise. Order: All cells top to bottom.
   - Purpose: Validate setup, quick tests.
   - Quantum tie-in: Noise in predictions for superposition of outcomes.

2. **01_nba_api_fetch_historical_team_data.ipynb** (Historical Data Fetch): Run once to get 5,000+ games, merges with your 10-year CSVs. Add live updates daily.
   - Purpose: Build dataset for backtesting.
   - Quantum tie-in: Ensembles average fetches with noise for "interference".
   - Order: Run all cells; save output CSV.

3. **injury_data_fetch.ipynb** (Injury Fetch/Merge): Run daily for live injuries (Rotowire/Balldontlie APIs). Merges sources, adds quantum probs for "Questionable".
   - Purpose: Update injuries for accurate adjustments.
   - Quantum tie-in: Beta distributions for superposition of play/out chances.
   - Order: Run all cells; outputs Injury_Overrides.csv.

4. **balldontlie_sync.ipynb** (Balldontlie Sync): Run for backup data sync (games/injuries). Uses your key for live.
   - Purpose: Fallback if nba_api down.
   - Order: Set dates, run cells.

5. **qepc_pipeline_smoketest.ipynb** (Smoketest): Run to test full pipeline (data to prediction) with quantum elements.
   - Purpose: Quick check if everything works.
   - Quantum tie-in: In sim step, noise for entanglement.

6. **qepc_backtest.ipynb** (Backtesting): Run for accuracy tests on history. Uses 10-year data, live recent games, quantum calibration on probs.
   - Purpose: Measure win %, spread error—key for "most accurate" goal.
   - Quantum tie-in: PennyLane circuit calibrates probs (superposition of diffs).
   - Order: Set dates, run all; export results.

7. **04_how_to_use_quantum_core_FIXED.ipynb** (Quantum Core Guide): Run to learn quantum parts (ensembles, consistency).
   - Purpose: Understand/use quantum inspiration.
   - Order: As needed; add PennyLane for ML.

8. **qepc_project_backup.ipynb** (Backup): Run anytime to zip project.
   - Purpose: Save work.

Other notebooks (e.g., 02_nba_api_comprehensive_player_fetcher.ipynb for player data): Run as needed for specific fetches.

**Recommended Order for Full Use**:
- Daily: Hub > Injury fetch > Backtest (for today's predictions).
- Weekly: Historical fetch (update data) > Backtest (test improvements).

## How to Backtest Right Now (Step-by-Step)
Backtesting tests predictions on past games to see accuracy (win %, spread error)—crucial for your goal.

1. Open `qepc_backtest.ipynb` (or adjusted version).
2. Run first cell (setup, imports)—includes live API for recent games.
3. Set dates in second cell (e.g., start='2025-10-01', end='2025-11-29' for recent).
4. Run the backtest loop—uses your 10-year data merged with live.
5. Check summary: Win % (aim 65%+), MAE spread (<8 points good).
6. Inspect worst misses (bottom cells)—see if injuries/lineups caused.
7. Export CSV for records.

To improve: Add your large CSVs to data/raw/, run historical fetch to merge. If PennyLane installed, quantum calibration runs auto for better probs.

## Tips and Troubleshooting
- **Quantum Inspiration**: In sims/preds, noise mimics superposition (multiple realities); PennyLane adds ML learning for entanglement (linked factors).
- **Accuracy Tips**: Run daily for live data; backtest on full history to calibrate.
- **Errors?**: "Module not found"—check imports/installs. "API error"—check internet/key. Share message for help.
- **Expand**: Add more APIs (e.g., for odds) in data_source.py.
- **Resources**: PennyLane docs for quantum ML; nba_api GitHub for endpoints.

This makes QEPC real and unique—quantum edge for top accuracy! Questions? Ask. 🚀