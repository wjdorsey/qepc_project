# QEPC Restore Guide

This document explains how to restore and run the QEPC NBA engine on a fresh machine
(or after things get messy).

The **high-level steps** are:

1. Restore the project code (Git or ZIP)
2. Recreate the Python environment
3. Restore the required `data/` CSVs
4. Run the setup + diagnostics notebook
5. Run the pipeline smoketest and then backtests

---

## 1. Restore the Project Code

### 1.1 From GitHub

```bash
git clone https://github.com/wjdorsey/qepc_project.git
cd qepc_project
````

### 1.2 From a ZIP backup

1. Copy the ZIP file to your machine (e.g. `qepc_project_backup_YYYYMMDD.zip`).
2. Extract it so the folder structure looks like:

```text
qepc_project/
  ├─ qepc/
  ├─ data/
  ├─ notebooks/
  ├─ qepc_autoload.py
  └─ README.md
```

Make sure the **folder you open in JupyterLab** is the top-level `qepc_project/`.

---

## 2. Recreate the Python Environment

These steps assume Conda, but you can adapt to venv/Pyenv.

```bash
conda create -n qepc python=3.11 -y
conda activate qepc
cd path/to/qepc_project
```

Install required packages (approximate list):

```bash
pip install pandas numpy scipy matplotlib jupyter nba_api requests
```

If `requirements.txt` or `environment.yml` exists, prefer:

```bash
pip install -r requirements.txt
# OR
conda env create -f environment.yml
conda activate qepc
```

---

## 3. Restore Local Data Files

Most large CSVs are **not tracked in Git**. They must be restored from your
local backups or re-built using the Data Upgrade / utility notebooks.

The engine expects these (or equivalent) files to exist:

### 3.1 Raw game logs

Under `data/raw/`:

* `TeamStatistics.csv`
* `PlayerStatistics.csv`
* `Team_Stats.csv`
* `Games.csv` (raw schedule)

### 3.2 Processed schedule & results

Under `data/`:

* `Games.csv` (processed schedule with `gameDate`)
* `GameResults_2025.csv`
* `Schedule_with_Rest.csv`
* `TeamForm.csv`

### 3.3 Props & splits

Under `data/props/`:

* `Player_Season_Averages.csv`
* `Player_Averages_With_CI.csv`
* `Player_Recent_Form_L5.csv`
* `Player_Recent_Form_L10.csv`
* `Player_Recent_Form_L15.csv`
* `Player_Home_Away_Splits.csv`
* (Optional) `Player_Props_Full_Logs.csv`, etc.

### 3.4 Injuries

* `data/Injury_Overrides.csv`
* `data/injuries/Injury_Overrides_MASTER.csv`
* `data/Injury_Overrides_live_espn.csv`

If any of these are missing, you can often rebuild them using notebooks
under `notebooks/02_utilities/` and `notebooks/Data Upgrades/`.

---

## 4. Launch JupyterLab

From the project root:

```bash
cd path/to/qepc_project
conda activate qepc
jupyter lab
```

Open the `notebooks/` folder in the Jupyter file browser.

---

## 5. Run Setup & Diagnostics

### 5.1 Notebook header (shared setup)

Every notebook should start with:

```python
from qepc.notebook_header import qepc_notebook_setup

env = qepc_notebook_setup(run_diagnostics=False)
data_dir = env.data_dir
raw_dir = env.raw_dir
```

The `notebook_header.py` module:

* Finds the project root
* Ensures `qepc` is importable
* Imports `qepc_autoload` (path wiring + banner)
* Provides `data_dir` and `raw_dir`
* Optionally runs diagnostics

### 5.2 Run the setup notebook

Open and run:

> `notebooks/00_setup/00_qepc_setup_environment.ipynb`

Use:

```python
from qepc.notebook_header import qepc_notebook_setup
env = qepc_notebook_setup(run_diagnostics=True)
```

This will:

* Print the resolved project root
* Show `data_dir` / `raw_dir`
* Run `qepc.utils.diagnostics.run_system_check()` and report:

  * Required files: found / missing
  * Schema checks
  * Module import status

Make sure all **critical files** are `OK` before continuing.

---

## 6. Run the Pipeline Smoketest

Next, open:

> `notebooks/01_core/qepc_pipeline_smoketest.ipynb`

This notebook:

1. Loads a subset of recent games from `TeamStatistics.csv`.
2. Builds team strengths with `calculate_advanced_strengths(...)`.
3. Applies team form boosts from `TeamForm.csv`.
4. Computes game-level lambdas via `compute_lambda(...)`.
5. Runs a small backtest using `run_qepc_simulation(...)`.

You should see a **mini backtest summary**, including:

* Games evaluated
* Home-win “accuracy” (for the sample)
* Mean/median total error
* Mean spread error

If these metrics look wildly off or the notebook errors out, go back to:

* Section **3 (data files)**
* Section **5 (diagnostics)**

and check for missing/mis-shaped data.

---

## 7. Run Full Backtests

For a deeper evaluation, run:

* `notebooks/01_core/qepc_backtest.ipynb`
* or `notebooks/01_core/qepc_enhanced_backtest_FIXED.ipynb`

These will:

* Build a full evaluation dataset
* Compare predicted vs actual scores from `GameResults_2025.csv`
* Save results to `data/results/backtests/`

Typical outputs:

* Win/Loss prediction accuracy
* Mean Absolute Error on totals and spreads
* Best/worst games by error

These notebooks should be used to validate changes to:

* `qepc/core/model_config.py`
* `qepc/sports/nba/strengths_v2.py`
* `qepc/core/lambda_engine.py`
* Any form / injury / rest logic

---

## 8. Rebuilding Data (If Needed)

If some CSVs are missing or corrupted, use the **utility** and **Data Upgrades**
notebooks under:

* `notebooks/02_utilities/`
* `notebooks/Data Upgrades/`

Examples:

* `01_nba_api_fetch_historical_team_data.ipynb`
* `02_nba_api_comprehensive_player_fetcher.ipynb`
* `03_qepc_player_props_processing.ipynb`
* `injury_data_fetch.ipynb`, `injuries_merge.ipynb`
* `balldontlie_sync.ipynb`

These notebooks are heavier and not required for **daily use**, only when
you’re rebuilding datasets.

---

## 9. If Things Break

If QEPC stops running:

1. Confirm you are at the project root in a terminal:

   ```bash
   cd path/to/qepc_project
   ```

2. Re-activate the environment:

   ```bash
   conda activate qepc
   ```

3. Re-run:

   * `notebooks/00_setup/00_qepc_setup_environment.ipynb`
   * `notebooks/01_core/qepc_pipeline_smoketest.ipynb`

4. Read the diagnostics output:

   * Missing files?
   * Schema mismatch?
   * Import error on a specific module?

Fix the root cause (file path, missing CSV, etc.), then re-run.

---

*Last updated: December 2025

