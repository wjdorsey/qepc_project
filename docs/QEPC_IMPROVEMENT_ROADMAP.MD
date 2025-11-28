qepc_project/
  .gitignore
  .gitattributes
  README.md
  requirements.txt
  main.py

  qepc/                     # Core Python package
    __init__.py
    autoload/
    core/
    sports/
      nba/
      nfl/
      nhl/
    backtest/
    ui/
    utils/
    # (optional later) features/, props/, experimental/

  data/
    Games.csv
    Players.csv
    Injury_Overrides.csv
    cache/
      lineup_cache.json
      props_cache.json
    metadata/
      teams.json
      arenas.csv
      league_settings.json
    results/
      backtests/
        Backtest_Results_*.csv
      # (later) reports/, daily_predictions/

  notebooks/
    00_setup/
    01_core/
    02_utilities/
    03_dev/
    # (no "data" subfolder if we can move it into data/)

  scripts/
    bootstrap.py
    rebuild_from_backup.py
    # (later) fetch_all_data.py, update_schedules.py, validate_data.py

  tests/
    test_autoload.py
    test_lambda_engine.py
    test_schedule_integrity.py
    # (later) test_injury_processing.py, test_data_loading.py, etc.

  docs/
    RESTORE_GUIDE.md
    COMPLETE_PROJECT_SUMMARY.md
    scripts_README.md
    info_header_cells.txt
    QEPC_IMPROVEMENT_ROADMAP.md (copy here)
    # (later) API.md, subfolder READMEs

  experimental/
    nfl_quantum/
      ...
    props/
      props_backtest_notebook.py
      qepc_props_backtest.ipynb
      qepc_props_daily.ipynb
    legacy_bootstrap/
      bootstrap_qepc.py

  backups/
    qepc_backup_*.zip
