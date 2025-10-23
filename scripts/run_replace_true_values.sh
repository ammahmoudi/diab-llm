#!/bin/bash

# Simple wrapper script to run the true values replacement
# Usage: ./run_replace_true_values.sh [dry-run]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/venv"

echo "Running true values replacement script..."
echo "Root directory: $ROOT_DIR"
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

if [ "$1" = "dry-run" ]; then
    echo "Running in DRY RUN mode - no files will be modified"
    python "$SCRIPT_DIR/replace_true_values_with_raw.py" --dry-run
else
    echo "Running in LIVE mode - files will be modified"
    read -p "Are you sure you want to proceed? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python "$SCRIPT_DIR/replace_true_values_with_raw.py"
    else
        echo "Operation cancelled."
        exit 1
    fi
fi

echo "Done!"
echo "Deactivating virtual environment..."
deactivate