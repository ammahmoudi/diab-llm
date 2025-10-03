#!/usr/bin/env python3
"""
Convenience wrapper for running all Chronos experiments.
Calls the main script from the correct location.
"""

import os
import sys
import subprocess

def main():
    # Get the script path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    chronos_runner = os.path.join(script_dir, "scripts", "chronos", "run_all_chronos_experiments.py")
    
    # Change to the correct directory and run
    os.chdir(script_dir)
    
    # Pass all arguments through
    cmd = [sys.executable, chronos_runner] + sys.argv[1:]
    return subprocess.call(cmd)

if __name__ == "__main__":
    exit(main())