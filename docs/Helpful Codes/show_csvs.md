import pandas as pd
from pathlib import Path

# Try to get the true project root from QEPC's autoload paths module
try:
    from qepc.autoload.paths import get_project_root
    project_root = get_project_root()
except Exception:
    # Fallback if that import fails for some reason
    project_root = Path.cwd()
    print("⚠️ Falling back to cwd as project root")

print("Project root:", project_root)

# Helper: pick the "best" match for a file name among many
def pick_best_match(matches):
    if not matches:
        return None
    # Prefer paths that live under a 'data' folder and NOT under 'notebooks'
    scored = []
    for p in matches:
        score = 0
        parts = [str(part).lower() for part in p.parts]
        if "data" in parts:
            score += 2
        if "raw" in parts:
            score += 1
        if "props" in parts:
            score += 1
        if "results" in parts:
            score += 1
        if "notebooks" in parts:
            score -= 2
        if ".ipynb_checkpoints" in str(p):
            score -= 5
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]

# (label, filename)
targets = [
    # Core game/team data
    ("TeamStatistics (team game logs)",      "TeamStatistics.csv"),
    ("Team_Stats (team season stats)",       "Team_Stats.csv"),
    ("PlayerStatistics (player logs)",       "PlayerStatistics.csv"),
    ("Canonical Games (schedule)",           "Games.csv"),
    ("GameResults_2025 (results)",           "GameResults_2025.csv"),
    ("Schedule_with_Rest",                   "Schedule_with_Rest.csv"),
    ("TeamForm",                             "TeamForm.csv"),

    # Roster / players
    ("Players",                              "Players.csv"),
    ("Players_Processed",                    "Players_Processed.csv"),

    # Injuries
    ("Injury_Overrides",                     "Injury_Overrides.csv"),
    ("Injury_Overrides_MASTER",              "Injury_Overrides_MASTER.csv"),
    ("Injury_Overrides_live_espn",           "Injury_Overrides_live_espn.csv"),

    # Props / aggregates
    ("Player_Season_Averages",               "Player_Season_Averages.csv"),
    ("Player_Averages_With_CI",              "Player_Averages_With_CI.csv"),
    ("Player_Recent_Form_L5",                "Player_Recent_Form_L5.csv"),
    ("Player_Recent_Form_L10",               "Player_Recent_Form_L10.csv"),
    ("Player_Recent_Form_L15",               "Player_Recent_Form_L15.csv"),
    ("Player_Home_Away_Splits",              "Player_Home_Away_Splits.csv"),
]

def preview_by_filename(label: str, filename: str, n: int = 3):
    print("\n" + "=" * 80)
    print(f"📄 {label}")
    print(f"Looking for filename: {filename}")

    # Find all matches anywhere under project_root
    matches = [p for p in project_root.rglob(filename)]
    if not matches:
        print("⚠️ No matches found in project.")
        return

    print("Found matches:")
    for m in matches:
        try:
            rel = m.relative_to(project_root)
        except ValueError:
            rel = m
        print("   •", rel)

    best = pick_best_match(matches)
    if best is None:
        print("⚠️ Could not choose a best match.")
        return

    try:
        rel_best = best.relative_to(project_root)
    except ValueError:
        rel_best = best

    print(f"\n✅ Using best match: {rel_best}")

    # Load a small sample (nrows=3) to avoid pulling full 300MB files
    try:
        df_sample = pd.read_csv(best, nrows=n)
        print(f"Sample shape: {df_sample.shape}")
        print("Columns:", list(df_sample.columns))
        print("\nSample rows:")
        display(df_sample)
    except Exception as e:
        print(f"❌ Error reading CSV sample: {e}")

for label, filename in targets:
    preview_by_filename(label, filename)
