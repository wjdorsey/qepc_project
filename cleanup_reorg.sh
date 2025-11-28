#!/bin/bash
# cleanup_reorg.sh: Reorganize qepc_project structure

set -e  # Exit on any error

# Ensure we're in the project root
cd "$(dirname "$0")" || exit 1

echo "Starting reorganization..."

# Create necessary directories
mkdir -p data/injuries
mkdir -p notebooks/02_utilities
mkdir -p archive
mkdir -p docs experimental tests backups

# Move and rename files (examples based on described fixes)
mv notebooks/data/data/Injury_Overrides_MASTER.csv data/injuries/Injury_Overrides_MASTER.csv 2>/dev/null || echo "File not found: Injury_Overrides_MASTER.csv"
rmdir notebooks/data/data 2>/dev/null || echo "Directory cleanup skipped"
rmdir notebooks/data 2>/dev/null || echo "Directory cleanup skipped"

mv 00_qepc_project_hub_FIXED.ipynb 00_qepc_project_hub.ipynb 2>/dev/null || echo "File not found: 00_qepc_project_hub_FIXED.ipynb"
mv qepc_sports_Untitled.ipynb archive/qepc_sports_Untitled.ipynb 2>/dev/null || echo "File not found: qepc_sports_Untitled.ipynb"

# Additional fixes (e.g., move notebooks to consistent subfolders)
# Adjust these mv commands based on your actual notebook locations
# mv some_notebook.ipynb notebooks/02_utilities/ 2>/dev/null || true

echo "Reorganization complete. Verify structure with 'ls -la'."