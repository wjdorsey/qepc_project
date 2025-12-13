# QEPC NBA Player Points Quantum Runbook

This is a lightweight checklist for experimenting locally with the new player points projection stack. Everything auto-detects `PROJECT_ROOT` via `qepc.utils.paths.get_project_root` (no hardcoded paths).

## 1) Prereqs
- Ensure the QEPC Eoin parquet caches exist under `cache/imports/` (games, player boxes, team boxes).
- Python dependencies: `pandas` and a parquet backend (e.g., `pyarrow`); install if missing.

## 2) Quick backtest
```bash
python -m qepc.nba.backtest_player_points --start 2022-10-01 --end 2024-06-22 --min-minutes 10
```
Outputs overall MAE/bias plus calibration by minutes buckets. Use `--config path/to/config.json` to override τ/weights.

## 3) Tune decoherence + entanglement
```bash
python -m qepc.nba.tune_player_points_qpa --start 2022-10-01 --end 2023-12-31 --min-minutes 12
```
Writes the best config to `cache/tuning/player_points_best.json` for reuse.

## 4) Build expectations programmatically
```python
from qepc.nba.player_points_model import build_player_points_expectations, PlayerPointsConfig
cfg = PlayerPointsConfig()  # or PlayerPointsConfig.from_dict(json.load(open(...)))
preds = build_player_points_expectations(start_date="2023-01-01", end_date="2023-12-31", min_minutes=10, config=cfg)
```
Each row contains `game_id`, `player_id`, `actual_points`, `predicted_points`, `predicted_variance`, and diagnostics (season_mean, recency_mean, decoherence features, entanglement_corr).

## 5) Extend to rebounds/assists
The config dataclasses and per-game schema are stat-agnostic; add rebound/assist analogues by mirroring the decoherence + blending helpers in `qepc.nba.player_points_model`.
