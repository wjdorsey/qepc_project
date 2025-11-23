"""
QEPC Project CLI Entry Point.
Use: python main.py --backup
"""
import argparse
import sys
from pathlib import Path

# --- Path Setup to ensure qepc modules can be imported ---
# Add project root to sys.path (needed if main.py is run directly)
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from qepc.utils.backup import create_project_backup

def main():
    parser = argparse.ArgumentParser(description="QEPC Project Command Line Utilities.")
    parser.add_argument('--backup', action='store_true', help='Creates a timestamped ZIP backup of the core project.')
    
    args = parser.parse_args()

    if args.backup:
        try:
            create_project_backup(verbose=True)
        except Exception as e:
            print(f"\nFATAL: Backup operation failed. Error: {e}")
            sys.exit(1)
        
    else:
        # Default behavior or print usage if no arguments are provided
        parser.print_help()
        
if __name__ == "__main__":
    main()