# QEPC NBA Pipeline Updates

- Added shared project-root detection helper (`qepc.utils.paths.get_project_root`) and wired all NBA modules/notebooks to use it.
- Introduced schema validation, leakage-free team state builders, and odds/games merge diagnostics (including smoke test entrypoint).
- Refreshed notebooks for portable execution and reproducible, leakage-free backtests.
