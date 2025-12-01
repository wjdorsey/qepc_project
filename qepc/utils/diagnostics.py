"""
QEPC Module: diagnostics.py
System health check and pre-flight validation tool.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd
from IPython.display import display, HTML


# ---------------------------------------------------------------------------
# 0. Required files & expected schemas
# ---------------------------------------------------------------------------

# Files we consider "core" to a healthy QEPC install.
# Paths are relative to the project root.
REQUIRED_FILES: Dict[str, str] = {
    # Core schedule + data
    "Canonical Schedule": "data/Games.csv",
    "Raw Player Stats": "data/raw/PlayerStatistics.csv",
    # UPDATED: raw team stats now live in data/raw/Team_Stats.csv
    "Raw Team Stats": "data/raw/Team_Stats.csv",

    # Autoload / bootstrapping
    "Autoload Context": "qepc_autoload.py",

    # Restore docs (either is fine for humans, but we check both)
    "Restore Guide (Notebook)": "RESTORE_GUIDE.ipynb",
    "Restore Guide (Markdown)": "notebooks/RESTORE_GUIDE.md",
}

# Some required files are *local-only* large CSVs that may not exist
# in a lightweight GitHub clone. When these are missing, we treat the
# status as WARN (not FAIL) and include guidance on how to generate them.
LOCAL_OPTIONAL_FILES: Dict[str, str] = {
    "Raw Player Stats": (
        "Local-only large CSV missing. "
        "Generate via Data Upgrades notebooks or place PlayerStatistics.csv "
        "under data/raw/ if you want full offline backtesting."
    ),
    "Raw Team Stats": (
        "Local-only large CSV missing. "
        "Generate via Data Upgrades notebooks or place Team_Stats.csv "
        "under data/raw/ if you want full offline backtesting."
    ),
}

# Optional schema expectations. We only validate schema if the file exists.
EXPECTED_SCHEMAS: Dict[str, List[str]] = {
    # Canonical schedule
    "data/Games.csv": ["Date", "Time", "Away Team", "Home Team"],

    # Raw stats sanity checks (just a few key columns)
    "data/raw/PlayerStatistics.csv": ["gameId", "firstName"],

    # UPDATED: raw team stats file + minimal required columns
    "data/raw/Team_Stats.csv": ["gameId", "teamName"],
}


# ---------------------------------------------------------------------------
# 1. Path helper
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """
    Resolve the QEPC project root.

    Prefer qepc.autoload.paths.get_project_root(), but fall back to
    inferring it from this file's location if that import fails.
    """
    try:
        from qepc.autoload import paths
        return paths.get_project_root()
    except Exception:
        # Fallback: .../qepc/utils/diagnostics.py -> project_root
        return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# 2. Simple print + HTML helpers
# ---------------------------------------------------------------------------

def print_status(label: str, ok: bool, detail: str = "", status_override: str = None) -> None:
    """Print a single status line with emoji and text.

    ok:
        - True  -> treated as OK
        - False -> treated as FAIL, unless status_override is provided

    status_override:
        - If "WARN", we render a warning state (⚠️ WARN),
          which is useful for local-only optional files.
    """
    if status_override is not None:
        status = status_override
        if status_override.upper() == "WARN":
            emoji = "⚠️"
        else:
            emoji = "✅" if ok else "❌"
    else:
        status = "OK" if ok else "FAIL"
        emoji = "✅" if ok else "❌"

    msg = f"{emoji} {label}: {status}"
    if detail:
        msg += f" – {detail}"
    print(msg)


def _html_table(title: str, rows: List[Tuple[str, str, str]]) -> None:
    """Render a simple HTML table in Jupyter, if available."""
    if not rows:
        return

    html_rows = "".join(
        f"<tr>"
        f"<td><b>{label}</b></td>"
        f"<td style='color:{('#2e7d32' if status == 'OK' else ('#f9a825' if status == 'WARN' else '#c62828'))}'>{status}</td>"
        f"<td>{detail}</td>"
        f"</tr>"
        for (label, status, detail) in rows
    )

    html = f"""
    <h3>{title}</h3>
    <table border="1" cellspacing="0" cellpadding="4">
        <tr>
            <th>Check</th>
            <th>Status</th>
            <th>Details</th>
        </tr>
        {html_rows}
    </table>
    """
    try:
        display(HTML(html))
    except Exception:
        # Not in a notebook? Just skip HTML.
        pass


# ---------------------------------------------------------------------------
# 3. File existence checks
# ---------------------------------------------------------------------------

def check_required_files(root: Path) -> List[Tuple[str, str, str]]:
    """
    Check that all REQUIRED_FILES exist relative to the project root.

    Returns a list of (label, status, detail) tuples for display.
    """
    results: List[Tuple[str, str, str]] = []

    print("\n🔍 Checking required files...")
    for label, rel_path in REQUIRED_FILES.items():
        abs_path = root / rel_path
        exists = abs_path.exists()

        # Local-only optional big files (e.g., raw stats) get WARN, not FAIL,
        # when missing. This makes a fresh GitHub clone friendlier while still
        # nudging the user toward generating them for full offline use.
        if label in LOCAL_OPTIONAL_FILES and not exists:
            detail = f"Missing at {abs_path}. {LOCAL_OPTIONAL_FILES[label]}"
            print_status(label, False, detail, status_override="WARN")
            results.append((label, "WARN", detail))
            continue

        detail = f"{abs_path}" if exists else f"Missing at {abs_path}"
        print_status(label, exists, detail)
        results.append((label, "OK" if exists else "FAIL", detail))

    _html_table("Required Files", results)
    return results


# ---------------------------------------------------------------------------
# 4. Schema checks (optional)
# ---------------------------------------------------------------------------

def check_data_schemas(root: Path) -> List[Tuple[str, str, str]]:
    """
    For known data files that exist, validate that expected columns are present.
    If a file is missing, we skip schema checks (existence is handled elsewhere).
    """
    results: List[Tuple[str, str, str]] = []

    print("\n📊 Checking data schemas (where files exist)...")
    for rel_path, expected_cols in EXPECTED_SCHEMAS.items():
        abs_path = root / rel_path
        if not abs_path.exists():
            # Existence already handled in check_required_files; skip here
            continue

        try:
            df = pd.read_csv(abs_path, nrows=5)
        except Exception as e:
            detail = f"Failed to load: {e}"
            print_status(f"Schema: {rel_path}", False, detail)
            results.append((rel_path, "FAIL", detail))
            continue

        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            detail = f"Missing columns: {missing}"
            ok = False
        else:
            detail = "All expected columns present."
            ok = True

        print_status(f"Schema: {rel_path}", ok, detail)
        results.append((rel_path, "OK" if ok else "FAIL", detail))

    _html_table("Schema Checks", results)
    return results


# ---------------------------------------------------------------------------
# 5. Module import checks
# ---------------------------------------------------------------------------

def check_module_imports() -> List[Tuple[str, str, str]]:
    """
    Try importing key QEPC modules to ensure they are discoverable
    and don't crash immediately on import.
    """
    modules_to_test = [
        "qepc.autoload.paths",
        "qepc.core.lambda_engine",
        "qepc.core.simulator",
        "qepc.sports.nba.sim",
        "qepc.sports.nba.strengths_v2",
        "qepc.sports.nba.player_data",
        "qepc.sports.nba.opponent_data",
        "qepc.utils.backup",
        "qepc.backtest.backtest_engine",
    ]

    results: List[Tuple[str, str, str]] = []

    print("\n🧪 Checking key module imports...")
    for mod in modules_to_test:
        try:
            __import__(mod)
            print_status(f"Module: {mod}", True, "Loaded")
            results.append((mod, "OK", "Loaded successfully"))
        except Exception as e:
            detail = f"Import Failed: {e}"
            print_status(f"Module: {mod}", False, detail)
            results.append((mod, "FAIL", detail))

    _html_table("Module Imports", results)
    return results


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def run_system_check() -> Dict[str, Any]:
    """
    Run the full QEPC system check and return a summary dictionary.
    Intended to be called from notebooks as:

        from qepc.utils.diagnostics import run_system_check
        run_system_check()
    """
    root = get_project_root()
    print_status("Project Root", True, f"Resolved to {root}")

    files_result = check_required_files(root)
    schema_result = check_data_schemas(root)
    modules_result = check_module_imports()

    print("\n✨ DIAGNOSTICS COMPLETE.")

    return {
        "project_root": str(root),
        "files": files_result,
        "schemas": schema_result,
        "modules": modules_result,
    }
