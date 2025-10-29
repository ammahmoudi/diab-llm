#!/usr/bin/env python3
"""
Script to run all fairness analyzers with flexible experiment type selection

Supports both per-patient and all-patients experiments.
"""

import subprocess
import sys
import argparse
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

# Parse arguments
parser = argparse.ArgumentParser(description='Run all fairness analyzers')
parser.add_argument('--experiment-type', type=str, default='per_patient',
                   choices=['per_patient', 'all_patients'],
                   help='Type of experiment to analyze (default: per_patient)')
args = parser.parse_args()

analyzers = {
    "Gender": "fairness/analyzers/gender_fairness_analyzer.py",
    "Age": "fairness/analyzers/age_fairness_analyzer.py",
    "Pump Model": "fairness/analyzers/pump_model_fairness_analyzer.py",
    "Sensor Band": "fairness/analyzers/sensor_fairness_analyzer.py",
    "Cohort": "fairness/analyzers/cohort_fairness_analyzer.py",
    "Legendary": "fairness/analyzers/legendary_distillation_analyzer.py"
}

# Dynamically get python executable from venv
venv_python = project_root / "venv" / "bin" / "python"
python_cmd = str(venv_python) if venv_python.exists() else sys.executable

exp_type_label = "ALL-PATIENTS" if args.experiment_type == "all_patients" else "PER-PATIENT"

print("=" * 80)
print(f"RUNNING ALL FAIRNESS ANALYZERS ({exp_type_label} MODE)")
print("=" * 80)

if args.experiment_type == "all_patients":
    print("\nAnalyzing all-patients experiments:")
    print("  - Single model trained on all patients")
    print("  - Per-patient inference for each phase (teacher, student, distilled)")
else:
    print("\nAnalyzing per-patient experiments:")
    print("  - Each patient has separate training/student/distillation phases")

print()

results = {}

for name, script in analyzers.items():
    print(f"\n🔍 Running {name} Analyzer ({exp_type_label})...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [python_cmd, script, "--experiment-type", args.experiment_type],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print(f"✅ {name} Analyzer: PASSED")
            results[name] = "PASSED"
        else:
            print(f"❌ {name} Analyzer: FAILED")
            print(f"Error: {result.stderr[:200]}")
            results[name] = "FAILED"
    except subprocess.TimeoutExpired:
        print(f"⏱️ {name} Analyzer: TIMEOUT")
        results[name] = "TIMEOUT"
    except Exception as e:
        print(f"💥 {name} Analyzer: ERROR - {e}")
        results[name] = "ERROR"

print("\n" + "=" * 80)
print("EXECUTION SUMMARY")
print("=" * 80)

for name, status in results.items():
    icon = "✅" if status == "PASSED" else "❌"
    print(f"{icon} {name:20s}: {status}")

passed = sum(1 for s in results.values() if s == "PASSED")
total = len(results)

print("\n" + "=" * 80)
print(f"FINAL RESULT: {passed}/{total} analyzers passed ({exp_type_label} mode)")
print("=" * 80)

sys.exit(0 if passed == total else 1)
