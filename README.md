````markdown
# QEPC – Quantum Entangled Poisson Cascade

QEPC is a **quantum-inspired sports prediction engine** for the NBA.

At its core, QEPC:

1. Builds **team strength ratings** (ORtg / DRtg / Pace / Volatility) from massive game logs.  
2. Converts those into **per-game scoring lambdas** (expected points) with situational adjustments (rest, travel, home court, form, etc).  
3. Runs a **Monte Carlo multiverse** of simulated games to estimate:
   - Win probabilities  
   - Expected scores & spreads  
   - Distribution of outcomes for further analysis (props, tails, etc)

The long-term goal is simple and ambitious:

> **Build the most accurate real-world sports prediction model ever created, without losing the “quantum computer” identity.**

---
# QEPC – Quantum Entangled Poisson Cascade (NBA Prediction Engine)

QEPC is a **quantum-inspired sports prediction engine** for the NBA.

At its core, QEPC:

1. Builds **team strength ratings** (ORtg / DRtg / Pace / Volatility) from massive game logs.  
2. Converts those into **per-game scoring lambdas** (expected points) with situational adjustments (rest, travel, home court, form, etc).  
3. Runs a **Monte Carlo multiverse** of simulated games to estimate:
   - Win probabilities  
   - Expected scores & spreads  
   - Distribution of outcomes for further analysis (props, tails, etc)

The long-term goal is simple and ambitious:

> **Build the most accurate real-world sports prediction model ever created, without losing the “quantum computer” identity.**

---

## 1. Project Layout

At a high level:

```text
qepc_project/
├─ qepc/                      # Core QEPC Python package
│  ├─ autoload/               # Path helpers and project bootstrap
│  ├─ core/                   # Lambda engine, simulator, model config
│  ├─ sports/
│  │  └─ nba/                 # NBA-specific strengths, data loaders, etc.
│  ├─ utils/                  # Diagnostics, backups, helpers
│  └─ notebook_header.py      # Shared setup for all notebooks
│
├─ data/                      # Local data files (NOT all tracked in Git)
│  ├─ raw/                    # Heavy raw CSVs (TeamStatistics, PlayerStatistics, etc.)
│  ├─ props/                  # Player props summaries & splits
│  ├─ injuries/               # Injury override tables
│  ├─ results/                # Backtest outputs & logs
│  ├─ metadata/               # Arenas, misc metadata
│  └─ ...                     # Schedule, team form, etc.
│
├─ notebooks/
│  ├─ 00_setup/               # Environment + diagnostics
│  ├─ 01_core/                # Main QEPC workflows (smoketest, backtest, dashboard)
│  ├─ 02_utilities/           # Data ingest / injury ETL / API utilities
│  ├─ 03_dev/                 # Experiments, templates, sandbox
│  └─ Data Upgrades/          # Heavy rebuilds (historical pulls, props pipelines)
│
├─ .gitignore
├─ (optionally) requirements.txt / environment.yml
└─ README.md                  # This file


---

## 2. Core Python Modules

Some key modules you’ll see referenced in notebooks:

* `qepc_autoload.py`
  Project bootstrap. Resolves the project root, sets up import paths, and wires `data/` locations.

* `qepc/notebook_header.py`
  **Single source of truth for notebook setup.** Every Jupyter notebook should start with:

  ```python
from qepc.notebook_header import qepc_notebook_setup

env = qepc_notebook_setup(run_diagnostics=False)  # or True in 00_setup
data_dir = env.data_dir
raw_dir = env.raw_dir

  ```

* `qepc/core/model_config.py`
  Global configuration for the NBA engine:

  * `LEAGUE_AVG_POINTS`
  * Home court advantage knobs
  * Rest / fatigue parameters
  * Simulation settings (`DEFAULT_NUM_TRIALS`, `QUANTUM_NOISE_STD`, etc.)
  * City distance table for travel penalties

* `qepc/sports/nba/strengths_v2.py`
  Builds **recency-weighted team strengths** from game-level team logs (`TeamStatistics.csv` / `Team_Stats.csv`):

  * Computes ORtg, DRtg, Pace, Volatility, SOS
  * Uses **exponential recency weighting** (recent games matter more)

* `qepc/sports/nba/team_form.py`
  Uses `TeamForm.csv` to apply a **“form boost”** to each team’s ORtg based on recent scoring performance (hot/cold).

* `qepc/core/lambda_engine.py`
  Converts team strengths + schedule into **per-game lambdas**:

  * Combines ORtg/DRtg/Pace
  * Adds home court, rest, back-to-back, travel, and form adjustments
  * Produces `lambda_home`, `lambda_away`, `vol_home`, `vol_away` per game

* `qepc/core/simulator.py`
  Runs a **Monte Carlo simulation** of each game:

  * Uses Poisson-ish / Gaussian-ish sampling around λ
  * Adds correlated noise (“entangled scores”)
  * Outputs win probabilities and expected score distributions

* `qepc/utils/diagnostics.py`
  System health checks:

  * Confirms key data files exist
  * Validates CSV schemas
  * Verifies core modules import correctly

---

## 3. Required Local Data Files

These are the **main CSVs** the engine expects to find under `data/` (or `data/raw/`). Many of them are too large for Git and must be maintained locally.

### 3.1 Team & Player Game Logs

* **`data/raw/TeamStatistics.csv`**
  Team game logs (long history). Key columns (not exhaustive):

  * `gameDate` (ISO timestamp, e.g. `2025-11-17T21:00:00Z`)
  * `teamCity`, `teamName`
  * `opponentTeamCity`, `opponentTeamName`
  * `teamScore`, `opponentScore`
  * `reboundsTotal`, `assists`, `threePointersMade`, ...
  * `seasonWins`, `seasonLosses`, `season`

* **`data/raw/PlayerStatistics.csv`**
  Player-level game logs. Key columns:

  * `firstName`, `lastName`, `personId`
  * `gameId`, `gameDate`
  * `playerteamCity`, `playerteamName`
  * `points`, `reboundsTotal`, `assists`, `threePointersMade`, etc.

* **`data/raw/Team_Stats.csv`**
  May contain similar info with additional historical metadata (`teamCity_hist`, `teamAbbrev_hist`, `league`).

### 3.2 Schedule & Results

* **`data/raw/Games.csv`**
  Canonical schedule. Columns:

  * `Date`, `Time`
  * `Away Team`, `Home Team`
  * `Venue`, `Notes`

* **`data/Games.csv`**
  Processed schedule copy, often with an added:

  * `gameDate` (datetime)
  * (optionally) `gameId`

* **`data/GameResults_2025.csv`**
  Game-level results for the 2025 season:

  * `Date`
  * `Home_Team`, `Away_Team`
  * `Home_Score`, `Away_Score`
  * `Home_Win`, `Total_Score`, plus a few efficiency stats

* **`data/Schedule_with_Rest.csv`**
  Rest features per team per game:

  * `gameDate`
  * `Team`
  * `days_since_last_game`
  * `is_back_to_back`
  * `is_rested`

* **`data/TeamForm.csv`**
  Aggregated recent form:

  * `Team`
  * `Last_Game_Date`
  * `Last_N_PPG`, `Last_N_OPPG`
  * `Last_N_Wins`, `Last_N_Win_Pct`
  * Other rolling stats (`Last_N_Rebounds`, `Last_N_Assists`, etc.)

### 3.3 Player Props & Splits

Under `data/props/`:

* `Player_Season_Averages.csv`
* `Player_Averages_With_CI.csv`
* `Player_Recent_Form_L5.csv`
* `Player_Recent_Form_L10.csv`
* `Player_Recent_Form_L15.csv`
* `Player_Home_Away_Splits.csv`
* (and possibly others like `Player_Props_Full_Logs.csv`, etc.)

These are currently more for **props and player-level analysis** and may be partially integrated / WIP.

### 3.4 Injuries

* **`data/Injury_Overrides.csv`**
  Simple impact table (e.g., `Impact` values used as multipliers).

* **`data/injuries/Injury_Overrides_MASTER.csv`**

* **`data/Injury_Overrides_live_espn.csv`**

These support a **future injury module**, mapping player status → team λ adjustments.

---

## 4. Installation & Environment

### 4.1 Clone the repo

```bash
git clone https://github.com/wjdorsey/qepc_project.git
cd qepc_project
```

### 4.2 Create a Python environment

Using `conda` (example):

```bash
conda create -n qepc python=3.11 -y
conda activate qepc
```

Install core dependencies (exact list may vary):

```bash
pip install pandas numpy scipy matplotlib jupyter nba_api requests
```

If a `requirements.txt` or `environment.yml` exists, prefer:

```bash
pip install -r requirements.txt
# or
conda env create -f environment.yml
conda activate qepc
```

### 4.3 Set up local data

Populate the local `data/` and `data/raw/` directories with your CSVs:

* `data/raw/TeamStatistics.csv`
* `data/raw/PlayerStatistics.csv`
* `data/raw/Team_Stats.csv`
* `data/raw/Games.csv`
* `data/Games.csv`
* `data/GameResults_2025.csv`
* `data/Schedule_with_Rest.csv`
* `data/TeamForm.csv`
* `data/props/*.csv`
* `data/injuries/*.csv`

These files are not all versioned in Git; they are expected to be maintained locally.

---

## 5. Notebook Workflows (Order of Use)

This is the **recommended workflow** for working with QEPC in JupyterLab.

### 5.1 Step 0 – Start JupyterLab

From the project root:

```bash
cd C:\Users\wdors\qepc_project
conda activate qepc   # or your environment
jupyter lab
```

Open notebooks from the **`notebooks/`** folder.

---

### 5.2 Step 1 – Environment & Diagnostics

**Notebook:** `notebooks/00_setup/00_qepc_setup_environment.ipynb`

This notebook:

* Verifies the project root and Python imports.
* Ensures `qepc_autoload.py` and the `qepc` package are wired correctly.
* Runs the system diagnostic check via `qepc.utils.diagnostics.run_system_check()`:

  * Confirms key data files exist.
  * Validates basic schemas.
  * Confirms core modules import (simulator, strengths, backtest engine).

Run this notebook **first** after any major change to:

* Paths
* Data folder layout
* Environment / dependencies

---

### 5.3 Step 2 – Pipeline Smoketest

**Notebook:** `notebooks/01_core/qepc_pipeline_smoketest.ipynb`

Purpose:

* Run an **end-to-end smoke test** of the QEPC engine on a small slice of data.

Typical flow inside:

1. Import the shared header:

   ```python
   from qepc.notebook_header import qepc_notebook_setup

   env = qepc_notebook_setup(run_diagnostics=False)
   data_dir = env.data_dir
   raw_dir = env.raw_dir
   ```

2. Load a subset of recent games from `TeamStatistics.csv` (via helper code).

3. Use `get_team_strengths(...)` from `qepc.sports.nba.strengths_v2`.

4. Apply recent form with `apply_team_form_boost(...)` from `qepc.sports.nba.team_form`.

5. Build lambdas with `compute_lambda(...)` from `qepc.core.lambda_engine`.

6. Run a mini backtest / summary using `run_qepc_simulation(...)` from `qepc.core.simulator`.

The smoketest should print metrics like:

* Number of games evaluated
* Mean / median total error
* Mean spread error
* Sample rows of actual vs predicted totals

**If the smoketest passes with reasonable errors**, the pipeline is likely healthy.

---

### 5.4 Step 3 – Full Backtest

**Notebook(s):**

* `notebooks/01_core/qepc_backtest.ipynb`
  and/or
* `notebooks/01_core/qepc_enhanced_backtest_FIXED.ipynb`

These notebooks:

* Build a **full evaluation dataset** over a specified date range.
* Compute team strengths, lambdas, and run simulations.
* Compare predictions vs actuals from `GameResults_2025.csv` (and/or similar).
* Save backtest results under `data/results/backtests/`.

Typical outputs:

* **Win/loss prediction accuracy**
* **Mean absolute error (total + spread)**
* **Best / worst games by error**
* Possibly per-team error breakdowns

Use these notebooks to:

* Measure progress when you change λ logic, rest handling, form boosts, or any model config.
* Tune hyperparameters in `model_config.py` (e.g., noise, rest advantage).

---

### 5.5 Step 4 – Dashboards / Visualization (Optional)

**Notebook:** `notebooks/01_core/qepc_dashboard.ipynb`

Intended for:

* Visualizing distributions of predicted scores.
* Looking at team strength tables (ORtg / DRtg / Pace / Volatility).
* Exploring backtest results interactively.

This is optional but helps **debug intuitions** about the model.

---

### 5.6 Step 5 – Utility Notebooks

Located in `notebooks/02_utilities/`. Examples:

* `injury_data_fetch.ipynb`, `injuries_merge.ipynb`
  Fetch and merge injury data into `Injury_Overrides` / `Injury_Overrides_MASTER`.

* `balldontlie_sync.ipynb`
  Sync data using the balldontlie API (player/game logs).

* `01_nba_api_fetch_historical_team_data.ipynb`,
  `02_nba_api_comprehensive_player_fetcher.ipynb`,
  `03_qepc_player_props_processing.ipynb` (under **Data Upgrades**).
  Used for heavy data pulls and ETL when refreshing historical datasets.

You generally run these **only when updating data**, not in day-to-day usage.

---

### 5.7 Step 6 – Dev / Sandbox

Under `notebooks/03_dev/`, you’ll find:

* `qepc_sandbox.ipynb` – Safe place to experiment
* `qepc_props_backtest_TEMPLATE.ipynb` – For future props backtests
* `qepc_quantum_testing.ipynb` – Quantum-flavored experiments
* `tools.ipynb` – Miscellaneous tests

These notebooks do **not** define the canonical pipeline; they’re for iteration and experimentation.

---

## 6. Daily Usage (Future Pattern)

Once the model is stable, a typical daily workflow will look like:

1. **Update injuries / status** using a utility notebook (ESPN API, etc).
2. **Update schedule & rest** if needed (e.g., via `Schedule_with_Rest` builder).
3. Run a **“today’s games” notebook** that:

   * Pulls today’s schedule (from CSV and/or NBA API).
   * Joins with the latest team strengths + form + rest.
   * Builds λ and runs simulations.
   * Outputs a table of:

     * `Home Team`, `Away Team`
     * `Home_Win_Prob`, `Expected_Total`, `Expected_Spread`
     * Any additional tags (e.g. Chaos vs Grind scripts, etc.)

That notebook can eventually become your “QEPC control panel” for live slates.

---

## 7. Design Philosophy (Quantum + Accuracy)

Two core principles drive QEPC:

1. **Quantum-Inspired Engine, Not a Black Box**

   * Everything flows from λ (expected points) and variance.
   * Simulations explore a **multiverse of possible games** using controlled randomness and correlation.
   * Features like injuries, rest, travel, and form are implemented as **transparent constraints and warps** on λ, not as opaque black-box outputs.

2. **Accuracy is the Final Judge**

   * Every change should be testable via backtest notebooks.
   * QEPC favors **verifiable statistics over narrative**:

     * No “revenge games,” “birthday games,” or similar stories unless statistically validated.
   * New modules (e.g., props, advanced scripts) should be:

     * Toggleable (on/off).
     * Evaluated with before/after metrics.

---

## 8. Known Work-In-Progress Areas

* **Injury integration**
  Injury impact currently uses override tables and default impact values; a full, rigorously calibrated injury module is still in progress.

* **Player prop modeling**
  Prop CSVs (season averages, recent form, splits) are available and partially used, but the full QEPC prop engine is still under construction.

* **Multi-league support (NFL / NHL)**
  The core framework is designed to be multi-sport, but this repository is currently focused primarily on the NBA.

* **Live API integration (NBA API, ESPN, balldontlie)**
  Basic NBA API hooks exist (e.g., scoreboard, team stats). Deeper integration (e.g., automated nightly refreshes, robust ID mapping) is ongoing.

---

## 9. Contributing / Modifying

This is an evolving personal project. If you’re modifying:

* **Model parameters:**
  Look in `qepc/core/model_config.py`.

* **Team strength logic:**
  Look in `qepc/sports/nba/strengths_v2.py` and `qepc/sports/nba/team_form.py`.

* **Lambda and simulation logic:**
  Look in `qepc/core/lambda_engine.py` and `qepc/core/simulator.py`.

* **Notebook bootstrapping:**
  Keep all shared setup logic in `qepc/notebook_header.py` so notebooks stay synchronized.

Whenever you change core logic, re-run:

1. `notebooks/00_setup/00_qepc_setup_environment.ipynb` (sanity + diagnostics)
2. `notebooks/01_core/qepc_pipeline_smoketest.ipynb`
3. `notebooks/01_core/qepc_backtest.ipynb` (or enhanced backtest)

…and track whether **win accuracy** and **error metrics** are moving in the right direction.

---

*Last updated: November 2025*

```

```
