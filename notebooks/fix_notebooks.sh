#!/bin/bash
# Fix Notebooks Issues
# Run from notebooks/ directory: bash fix_notebooks.sh

set -e

echo "🔧 Fixing QEPC Notebook Issues..."
echo ""

# Check we're in notebooks folder
if [ ! -d "00_setup" ] || [ ! -d "01_core" ]; then
    echo "❌ Error: Run this script from the notebooks/ directory"
    echo "   Example: cd notebooks && bash fix_notebooks.sh"
    exit 1
fi

echo "✅ Running from notebooks/ directory"
echo ""

##############################
# Fix 1: Remove _FIXED suffixes
##############################

echo "📝 Fix #1: Removing _FIXED suffixes..."

# Hub notebook
if [ -f "00_setup/00_qepc_project_hub_FIXED.ipynb" ]; then
    # Backup old version if it exists
    if [ -f "00_setup/00_qepc_project_hub.ipynb" ]; then
        echo "   📦 Backing up old 00_qepc_project_hub.ipynb"
        mv "00_setup/00_qepc_project_hub.ipynb" \
           "00_setup/00_qepc_project_hub.OLD.ipynb"
    fi
    mv "00_setup/00_qepc_project_hub_FIXED.ipynb" \
       "00_setup/00_qepc_project_hub.ipynb"
    echo "   ✅ Renamed: 00_qepc_project_hub.ipynb"
else
    echo "   ℹ️  00_qepc_project_hub.ipynb already fixed"
fi

# Dashboard notebook
if [ -f "01_core/qepc_dashboard_FIXED.ipynb" ]; then
    # Backup old version if it exists
    if [ -f "01_core/qepc_dashboard.ipynb" ]; then
        echo "   📦 Backing up old qepc_dashboard.ipynb"
        mv "01_core/qepc_dashboard.ipynb" \
           "01_core/qepc_dashboard.OLD.ipynb"
    fi
    mv "01_core/qepc_dashboard_FIXED.ipynb" \
       "01_core/qepc_dashboard.ipynb"
    echo "   ✅ Renamed: qepc_dashboard.ipynb"
else
    echo "   ℹ️  qepc_dashboard.ipynb already fixed"
fi

echo ""

##############################
# Fix 2: Archive empty notebooks
##############################

echo "🗑️  Fix #2: Archiving empty/untitled notebooks..."

# Create archive directory
mkdir -p 03_dev/archive

# Archive nearly empty notebook
if [ -f "03_dev/qepc_sports_Untitled.ipynb" ]; then
    mv "03_dev/qepc_sports_Untitled.ipynb" \
       "03_dev/archive/"
    echo "   ✅ Archived: qepc_sports_Untitled.ipynb"
else
    echo "   ℹ️  qepc_sports_Untitled.ipynb already archived"
fi

# Archive any other Untitled notebooks
untitled_count=$(find . -name "*Untitled*.ipynb" -not -path "*/archive/*" 2>/dev/null | wc -l)
if [ "$untitled_count" -gt 0 ]; then
    find . -name "*Untitled*.ipynb" -not -path "*/archive/*" -exec mv {} 03_dev/archive/ \;
    echo "   ✅ Archived $untitled_count additional untitled notebook(s)"
fi

# Archive .OLD versions if they exist
if [ -f "00_setup/00_qepc_project_hub.OLD.ipynb" ]; then
    mkdir -p 00_setup/archive
    mv "00_setup/00_qepc_project_hub.OLD.ipynb" 00_setup/archive/
    echo "   ✅ Archived old hub notebook version"
fi

if [ -f "01_core/qepc_dashboard.OLD.ipynb" ]; then
    mkdir -p 01_core/archive
    mv "01_core/qepc_dashboard.OLD.ipynb" 01_core/archive/
    echo "   ✅ Archived old dashboard version"
fi

echo ""

##############################
# Verification
##############################

echo "🔍 Verification:"
echo ""

# Check for _FIXED files
fixed_count=$(find . -name "*_FIXED*" 2>/dev/null | wc -l)
if [ "$fixed_count" -eq 0 ]; then
    echo "   ✅ No _FIXED suffix files remaining"
else
    echo "   ⚠️  $fixed_count _FIXED files still present:"
    find . -name "*_FIXED*"
fi

# Check key notebooks exist
if [ -f "00_setup/00_qepc_project_hub.ipynb" ]; then
    echo "   ✅ Hub notebook in place"
else
    echo "   ❌ Hub notebook missing!"
fi

if [ -f "01_core/qepc_dashboard.ipynb" ]; then
    echo "   ✅ Dashboard notebook in place"
else
    echo "   ❌ Dashboard notebook missing!"
fi

# Check folder structure
if [ -d "02_utilities" ]; then
    echo "   ✅ 02_utilities folder exists"
else
    echo "   ⚠️  02_utilities folder not found"
fi

# Check archive created
if [ -d "03_dev/archive" ]; then
    archived=$(ls -1 03_dev/archive/*.ipynb 2>/dev/null | wc -l)
    echo "   ✅ Archive folder created ($archived files archived)"
else
    echo "   ℹ️  No archive needed"
fi

echo ""
echo "="*60
echo "✨ Automated fixes complete!"
echo "="*60
echo ""
echo "📝 Manual step still needed:"
echo "   Update first cell of 02_utilities/injury_data_fetch.ipynb"
echo "   See: injury_data_fetch_FIXED_CELL.py for the corrected code"
echo ""
echo "🧪 Testing recommended:"
echo "   1. Open 00_setup/00_qepc_project_hub.ipynb"
echo "   2. Run first cell - should show correct project root"
echo "   3. Test other notebooks similarly"
echo ""
echo "🎉 After manual fix, your notebooks will be production-ready!"
