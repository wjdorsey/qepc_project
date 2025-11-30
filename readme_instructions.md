---

# 📚 QEPC Notebooks – How To Use Everything (And In What Order)

This document explains:

* What each notebook folder is for
* A **recommended workflow** (setup → daily use → backtesting → data upgrades)
* Which notebooks are “core” vs “utility” vs “experimental”

Paths below are relative to the project root, e.g. `notebooks/01_core/...`.

---

## 0. Notebook Folder Map

### `notebooks/00_setup/`

* **`00_adjusted_qepc_project_hub.ipynb`**
  “Home screen” for QEPC notebooks.

  * Loads the QEPC environment (`notebook_context`, `qepc_autoload`, etc.)
  * Can run quick predictions / quantum-style simulations
  * Shows today’s games via `nba_api`
  * Quick sanity checks that paths and data are wired correctly

> **Start here for interactive use.**

---

### `notebooks/01_core/` – Main QEPC workflows

These are the **core** notebooks for running QEPC:

* **`qepc_main.ipynb`**
  High-level project overview + quick actions. Good “dashboard” style notebook.

* **`qepc_pipeline_smoketest.ipynb`**
  End-to-end “pipe check” using a **tiny test game**:

  * Imports `strengths_v2`, `lambda_engine`, `simulator`
  * Builds team strengths from `Team_Stats`
  * Creates a 1-game schedule
  * Computes λ and simulates the game

  > Use this whenever you change code or data to confirm the engine still runs.

* **`qepc_schedules.ipynb`**
  Schedule viewer:

  * Browse upcoming games (using `get_today_games`, `get_upcoming_games`)
  * Filter by team and/or date range

  > Use this to inspect schedules or check that schedule data is loading correctly.

* **`qepc_simple_backtest.ipynb`**
  Lightweight backtest:

  * Uses your current QEPC engine if available
  * Falls back to a simple baseline if not
  * Focuses on recent games (e.g., last 30 days)

  > Good for a **quick check** that predictions roughly make sense.

* **`qepc_enhanced_backtest_FIXED.ipynb`**
  Full-featured backtest (main one to use):

  * Loads **actual game results**
  * Runs QEPC predictions over a configurable date range
  * Computes detailed accuracy metrics (win%, MAE, etc.)
  * Generates plots (if plotting libraries are available)

  > Use this when you want serious evaluation of changes.

* **`qepc_project_backup.ipynb`**
  Project backup tool:

  * Detects project root
  * Creates a ZIP backup of your code + selected data folders

  > Run this before major refactors or experiments.

---

### `notebooks/02_utilities/` – Data & utilities

These notebooks are for **maintaining / refreshing data**, not everyday use:

* **`injury_data_fetch.ipynb` / `adjusted_injury_data_fetch.ipynb`**

  * Fetch and normalize injury data from external sources (e.g., ESPN, APIs).
  * Output to CSVs like `data/Injury_Overrides_live_espn.csv`.

* **`injuries_merge.ipynb`**

  * Merge multiple injury data sources into a canonical `Injury_Overrides.csv`.

* **`Injury_impact_analysis.ipynb`**

  * Explore / validate how injuries map into QEPC’s impact model.
  * Works with `qepc.sports.nba.injury_impact_analysis`.

* **`balldontlie_sync.ipynb`**

  * Uses the balldontlie API to fetch/refresh certain NBA data.
  * Not part of the daily pipeline—more of a data maintenance tool.

* `data/` subfolder (inside `02_utilities`)

  * Mirror of some data files used for experimentation & utilities.
  * Not the “canonical” engine data; the engine uses `project_root/data/...`.

---

### `notebooks/03_dev/` – Experiments & dev

These are **sandbox / prototype** notebooks. They’re not required for normal use:

* **`qepc_sandbox.ipynb`** – freeform experiments
* **`qepc_quantum_testing.ipynb`** – playing with quantum-style logic
* **`qepc_props_backtest_TEMPLATE.ipynb`** – template for player prop backtesting
* **`tools.ipynb`** – scratchpad for tools and helper code

> Use these only when you’re deliberately experimenting. They’re not part of the “happy path” workflow.

---

### `notebooks/Data Upgrades/`

These are **heavy ETL / upgrade** notebooks used to build or refresh your big CSVs from APIs:

* **`01_nba_api_fetch_historical_team_data.ipynb`**

  * Pull team-level stats via `nba_api` (multiple seasons).
  * Saves outputs into `data/raw/...` for use by QEPC.

* **`02_nba_api_comprehensive_player_fetcher.ipynb`**

  * Pull comprehensive player game logs.
  * Populates `data/raw/PlayerStatistics.csv` and related player tables.

* **`03_qepc_player_props_processing.ipynb`**

  * Process raw player logs into props tables:

    * `props/Player_Season_Averages.csv`
    * `props/Player_Recent_Form_L5/L10/L15.csv`
    * `props/Player_Averages_With_CI.csv`, etc.

* **`04_how_to_use_quantum_core_FIXED.ipynb`**

  * Beginner-friendly “how to use the QEPC quantum core” guide:

    * Shows how to search for players (including special characters like Dončić, Jokić).
    * Examples of predicting a game and combining multiple predictions.

> Run these **rarely** (e.g., when upgrading to a new season or doing a full data refresh).

---

### Root-level: `notebooks/how_to_use_quantum_core.ipynb`

* Early “how to use QEPC” guide.
* Superseded by `Data Upgrades/04_how_to_use_quantum_core_FIXED.ipynb`

  > Prefer the `FIXED` version for current use.

---

## 1. Recommended Workflow – Fresh Session / Daily Use

When you sit down to *use* QEPC (not develop it), this is the streamlined flow:

1. **Open the Project Hub**

   * Notebook: `notebooks/00_setup/00_adjusted_qepc_project_hub.ipynb`
   * Run cells from top to bottom:

     * Loads environment (`notebook_context`, `qepc_autoload`).
     * Confirms core paths and key data files exist.
     * Optionally pulls today’s games via `nba_api`.

2. **Check Today’s Schedule / Upcoming Games (optional but useful)**

   * Notebook: `notebooks/01_core/qepc_schedules.ipynb`
   * Use this to:

     * View today’s games and the next 7 days.
     * Filter by team to see their upcoming schedule.
   * Confirms that your schedule data (or `nba_api`) is working.

3. **Explore / Predict with QEPC Quantum Tools**

   For interactive experiments, two good options:

   * `notebooks/Data Upgrades/04_how_to_use_quantum_core_FIXED.ipynb`

     * Beginner–friendly; shows how to:

       * Find players by name (including special characters).
       * Run example predictions & combine them quantum-style.

   * `notebooks/01_core/qepc_main.ipynb`

     * Overview + quick access to:

       * Today’s schedule
       * Core module checks
       * Simple example predictions

4. **(Optional) Quick Backtest Check While You Work**

   * Notebook: `notebooks/01_core/qepc_simple_backtest.ipynb`
   * Run this to:

     * Backtest QEPC (or a fallback baseline) on recent games.
     * Get a fast sanity check on performance **without** too much setup.

---

## 2. Recommended Workflow – After Code/Data Changes (Pre-Backtest)

Whenever you change:

* Core engine code (`lambda_engine`, `simulator`, `strengths_v2`, etc.), or
* Key data files (`Team_Stats.csv`, `Games.csv`, etc.),

run this sequence before doing serious backtests:

1. **Project Hub sanity check**

   * `00_setup/00_adjusted_qepc_project_hub.ipynb`
   * Make sure it runs start-to-finish without errors.

2. **Pipeline Smoketest**

   * `01_core/qepc_pipeline_smoketest.ipynb`
   * Confirms:

     * Team strengths can be computed from `Team_Stats`.
     * λ values compute without error.
     * Simulator can run a game successfully.

3. **(Optional) Schedule Viewer**

   * `01_core/qepc_schedules.ipynb`
   * Ensures schedule loading still works (either from `nba_api` or `data/Games.csv`).

Only after these pass is it worth moving on to full backtests.

---

## 3. Recommended Workflow – Full Backtesting

When you want to **evaluate QEPC** over a real set of games:

1. **Make sure the engine is healthy**

   * Run:

     * `00_adjusted_qepc_project_hub.ipynb`
     * `qepc_pipeline_smoketest.ipynb`

2. **Run Simple Backtest (quick sanity)**

   * `01_core/qepc_simple_backtest.ipynb`
   * Uses your current engine on a recent subset of games.

3. **Run Enhanced Backtest (main evaluation)**

   * `01_core/qepc_enhanced_backtest_FIXED.ipynb`
   * Steps inside:

     * Load actual game results (from your results / schedule data).
     * Run QEPC predictions over a date range you choose.
     * Compute metrics:

       * Win/loss prediction accuracy
       * Error metrics (MAE, etc.)
     * Plot prediction vs actual (if plotting libs are available).
     * Save outputs to `data/results/backtests/...`.

4. **Backup Before Big Experiments (optional but recommended)**

   * `01_core/qepc_project_backup.ipynb`
   * Creates a ZIP of your project (with configurable include/exclude lists).

---

## 4. When To Use the Utilities & Data Upgrade Notebooks

You **do not** need to run these daily.

Use them in these situations:

### Injury / Overrides Data Refresh

* `02_utilities/injury_data_fetch.ipynb` or `02_utilities/adjusted_injury_data_fetch.ipynb`
  When you want to pull fresh injuries from external APIs / sites.

* `02_utilities/injuries_merge.ipynb`
  When you need to merge different injury sources into the main `Injury_Overrides.csv`.

* `02_utilities/Injury_impact_analysis.ipynb`
  When tuning how injuries map into QEPC’s impact metrics (testing assumptions).

### Big Data Refresh / New Season Setup

* `Data Upgrades/01_nba_api_fetch_historical_team_data.ipynb`
* `Data Upgrades/02_nba_api_comprehensive_player_fetcher.ipynb`
* `Data Upgrades/03_qepc_player_props_processing.ipynb`

Run these:

* At the **start of a new season**, or
* If you want to refresh all underlying team/player/props data.

They populate:

* `data/raw/Team_Stats.csv`
* `data/raw/PlayerStatistics.csv`
* `props/Player_Season_Averages.csv`
* `props/Player_Recent_Form_L5/L10/L15.csv`, etc.

---

## 5. TL;DR – Minimal “Happy Path”

If you just want a **short checklist** for using QEPC + running backtests:

1. `00_setup/00_adjusted_qepc_project_hub.ipynb`
   → Confirm environment & data.

2. `01_core/qepc_pipeline_smoketest.ipynb`
   → Confirm engine (strengths → λ → sim) works.

3. `01_core/qepc_schedules.ipynb` (optional)
   → Inspect schedule / upcoming games.

4. `01_core/qepc_simple_backtest.ipynb`
   → Quick sanity backtest on recent games.

5. `01_core/qepc_enhanced_backtest_FIXED.ipynb`
   → Full backtest with metrics + plots + saved results.

6. `01_core/qepc_project_backup.ipynb` (optional)
   → Create a safety backup before deeper changes.

---


