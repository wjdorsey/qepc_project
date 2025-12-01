# Contributing to QEPC

This is an evolving, experimental project aimed at building a
**quantum-inspired, hyper-accurate NBA prediction engine**.

Because the system is complex and highly interconnected, changes should be
made carefully and always backed by diagnostics + backtests.

---

## 1. Project Philosophy

1. **Quantum-Inspired, Not Random**  
   QEPC uses probabilistic cascades, correlated noise, and λ-based modeling
   to mimic a “quantum computer” exploring many universes.  
   Changes should preserve **transparent, interpretable logic**, not hide
   everything inside a black box.

2. **Verifiable Stats > Vibes**  
   No hand-wavy narrative (“revenge game”, “birthday game”) unless you can
   show consistent, measurable impact. Data first, stories second.

3. **Accuracy is the Judge**  
   Every meaningful change should be validated via backtests and recorded
   metrics (win accuracy, MAE, etc.).

---

## 2. Before You Change Anything

1. Make sure the project runs cleanly:

   - Run `notebooks/00_setup/00_qepc_setup_environment.ipynb`
   - Run `notebooks/01_core/qepc_pipeline_smoketest.ipynb`

2. Confirm diagnostics say:

   - Key data files: `OK`
   - Schemas: `OK`
   - Core imports: `OK`

If the base state is broken, fix that first.

---

## 3. Where to Change Things

### 3.1 Model configuration

**File:** `qepc/core/model_config.py`

Controls:

- `LEAGUE_AVG_POINTS`
- Home court advantage (`BASE_HCA`, `TEAM_HCA_BOOST`)
- Rest & fatigue (`REST_ADVANTAGE_PER_DAY`, `B2B_PENALTY`, etc.)
- Simulation parameters (`DEFAULT_NUM_TRIALS`, `QUANTUM_NOISE_STD`, `SCORE_CORRELATION`)
- City distances / travel penalties

**Guideline:**  
When you change these values, re-run:

1. Pipeline smoketest (`qepc_pipeline_smoketest.ipynb`)
2. Main backtest (`qepc_backtest.ipynb`)

Record how metrics move.

---

### 3.2 Team strengths

**File:** `qepc/sports/nba/strengths_v2.py`

Responsible for:

- Reading team game logs (TeamStatistics / Team_Stats)
- Parsing dates and applying recency weighting
- Computing ORtg, DRtg, Pace, Volatility, SOS per team

**Guideline:**  
If you change how strengths are calculated:

- Validate that output table looks reasonable (no insane ORtg/DRtg)
- Make sure all 30 teams exist
- Re-run smoketest + backtest

---

### 3.3 Team form / recent performance

**File:** `qepc/sports/nba/team_form.py` (or similar)

Applies “form boosts” to team ORtg based on `TeamForm.csv`.

Changes here should be subtle and tested: too much boost can explode totals.

---

### 3.4 Lambda engine

**File:** `qepc/core/lambda_engine.py`

Takes:

- Team strengths
- Rest / B2B / travel
- Form adjustments
- Home court advantage

and outputs:

- `lambda_home`, `lambda_away`
- `vol_home`, `vol_away`

**Guideline:**

- Keep λ in a **realistic NBA range** (~95–130).
- Use clamping or scaling to prevent runaway values.
- Changes here strongly affect totals; always re-calibrate with backtests.

---

### 3.5 Simulator

**File:** `qepc/core/simulator.py`

Runs Monte Carlo simulations given λ and variance.

You can experiment with:

- Different distributions (Poisson, Normal, hybrid)
- Correlation structures between teams
- Number of trials

Again, always check error metrics before and after.

---

## 4. Notebook Header / Setup

**File:** `qepc/notebook_header.py`

This is the **single source of truth** for notebook setup. All notebooks should
start with:

```python
from qepc.notebook_header import qepc_notebook_setup
env = qepc_notebook_setup(run_diagnostics=False)
data_dir = env.data_dir
raw_dir = env.raw_dir
````

If you need new common behavior (e.g., seeding RNG, loading a common config),
add it **here**, not individually in every notebook.

---

## 5. Testing Changes

After any non-trivial change:

1. Run:

   * `notebooks/00_setup/00_qepc_setup_environment.ipynb`
   * `notebooks/01_core/qepc_pipeline_smoketest.ipynb`

2. Then run:

   * `notebooks/01_core/qepc_backtest.ipynb`
   * Optionally `qepc_enhanced_backtest_FIXED.ipynb`

3. Compare:

   * Win accuracy (%)
   * Mean absolute error on totals
   * Mean absolute error on spreads

Keep notes on which changes helped or hurt. QEPC should move toward:

* Lower errors
* More stable behavior
* No crazy outliers (e.g., 160–150 scorelines unless historically justified)

---

## 6. Data Changes

If you modify the structure of any CSV in `data/` or `data/raw/`:

* Update any loader code that relies on specific columns.
* Update diagnostics in `qepc/utils/diagnostics.py` to reflect the new schema.
* Consider writing a **small utility notebook** in `notebooks/02_utilities/`
  to regenerate or validate the new format.

---

## 7. Versioning & Experiments

To keep things organized, try to:

* Add comments like `# QEPC vXX.X – change description` near major tweaks.
* Use separate branches or at least separate commits for:

  * Model config changes
  * Data ETL changes
  * Notebook / visualization changes

This makes it easier to revert specific ideas that didn’t work out.

---

## 8. Golden Rule

> **Every interesting idea is allowed.
> Every change must face the backtest.**

If a quantum-flavored twist, injury model, or form boost improves repeated,
out-of-sample performance, it belongs in QEPC.
If it’s just vibes, it stays in the sandbox.

---

```
