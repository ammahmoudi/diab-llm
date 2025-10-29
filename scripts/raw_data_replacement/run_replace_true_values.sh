#!/bin/bash

# Wrapper script to run the true values replacement
# Usage: ./run_replace_true_values.sh [options]
# Options:
#   --dry-run           Run in dry run mode (no files modified)
#   --auto_confirm      Skip interactive confirmation
#   --experiments_dir   Specify experiments directory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/venv"

# Parse command line arguments
DRY_RUN=false
AUTO_CONFIRM=false
EXPERIMENTS_DIR=""
PYTHON_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            PYTHON_ARGS="$PYTHON_ARGS --dry-run"
            shift
            ;;
        --auto_confirm)
            AUTO_CONFIRM=true
            shift
            ;;
        --experiments-dir|--experiments_dir)
            EXPERIMENTS_DIR="$2"
            PYTHON_ARGS="$PYTHON_ARGS --experiments-root $2"
            shift 2
            ;;
        --experiments-root)
            EXPERIMENTS_DIR="$2"
            PYTHON_ARGS="$PYTHON_ARGS --experiments-root $2"
            shift 2
            ;;
        dry-run)
            # Support legacy usage
            DRY_RUN=true
            PYTHON_ARGS="$PYTHON_ARGS --dry-run"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Running true values replacement script..."
echo "Root directory: $ROOT_DIR"
if [ -n "$EXPERIMENTS_DIR" ]; then
    echo "Experiments directory: $EXPERIMENTS_DIR"
fi
echo ""

# Check if virtual environment exists, create if not
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    cd "$ROOT_DIR"
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Install required packages if not already installed
echo "Checking/installing required packages..."
pip install pandas numpy > /dev/null 2>&1 || {
    echo "Installing required packages..."
    pip install pandas numpy
}

cd "$ROOT_DIR"

if [ "$DRY_RUN" = true ]; then
    echo "Running in DRY RUN mode - no files will be modified"
    python "$SCRIPT_DIR/replace_true_values_with_raw.py" $PYTHON_ARGS
elif [ "$AUTO_CONFIRM" = true ]; then
    echo "Running in LIVE mode with auto-confirmation - files will be modified"
    python "$SCRIPT_DIR/replace_true_values_with_raw.py" $PYTHON_ARGS
else
    echo "Running in LIVE mode - files will be modified"
    read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/replace_true_values_with_raw.py" $PYTHON_ARGS
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

echo "Done!"
echo "Deactivating virtual environment..."
deactivate