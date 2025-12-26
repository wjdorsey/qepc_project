# QEPC Codex Agent Rules (NBA)

## Non-negotiables
- No hardcoded machine paths (no C:\Users\...). Use project-root auto-detect + relative paths only.
- All rolling/time-window features MUST be leakage-safe: apply shift(1) (or equivalent past-only) before rolling.
- Keep changes minimal, testable, and reversible. Avoid refactors unrelated to the task.

## Audit checklist
1) For every feature used in predictions, locate exact source code and verify it does not use same-game or future data.
2) Verify split logic: date filters, min-minutes filters, and NaN handling happen consistently.
3) Look for `game_date` vs `game_datetime` mismatches and timezone conversions.
4) Confirm merges/sorts do not misalign predictions and actual targets.
5) Verify affine calibration is applied exactly once to the intended column.

## Required outputs
- AUDIT_REPORT.md with issues + tests + minimal fix.
- If code changes are made, provide PowerShell commands to reproduce before/after.
