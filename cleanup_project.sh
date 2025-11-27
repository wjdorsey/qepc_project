#!/bin/bash
# ============================================================================
# QEPC Quick Cleanup - Simple Version
# ============================================================================
# Just the essential commands to organize your project
# Copy and paste these into your terminal
# ============================================================================

# Navigate to project root (adjust path if needed)
cd ~/qepc_project

# 1. Move notebook out of data folder
mv data/injuries_merge.ipynb notebooks/ 2>/dev/null || echo "injuries_merge.ipynb already moved"

# 2. Create organized notebook folders
mkdir -p notebooks/00_setup
mkdir -p notebooks/01_core
mkdir -p notebooks/02_utilities
mkdir -p notebooks/03_dev

# 3. Organize notebooks (adjust filenames to match your actual notebooks)
# Setup notebooks
mv notebooks/*setup*.ipynb notebooks/00_setup/ 2>/dev/null

# Core notebooks
mv notebooks/*dashboard*.ipynb notebooks/01_core/ 2>/dev/null
mv notebooks/*backtest*.ipynb notebooks/01_core/ 2>/dev/null
mv notebooks/*schedule*.ipynb notebooks/01_core/ 2>/dev/null
mv notebooks/*inspector*.ipynb notebooks/01_core/ 2>/dev/null

# Utility notebooks
mv notebooks/*backup*.ipynb notebooks/02_utilities/ 2>/dev/null
mv notebooks/injuries_merge.ipynb notebooks/02_utilities/ 2>/dev/null
mv notebooks/espn_data_fetch.ipynb notebooks/02_utilities/ 2>/dev/null

# Dev notebooks
mv notebooks/*sandbox*.ipynb notebooks/03_dev/ 2>/dev/null
mv notebooks/*dev*.ipynb notebooks/03_dev/ 2>/dev/null

# 4. Commit changes
git add .
git status
echo ""
echo "Review the changes above, then run:"
echo "git commit -m 'Organize notebooks into subfolders'"
echo "git push"