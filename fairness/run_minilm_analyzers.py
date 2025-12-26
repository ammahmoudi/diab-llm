#!/usr/bin/env python3
"""
Script to run DISTILLATION fairness analyzers for MiniLM experiments

Analyzes fairness across the distillation process (teacher → student → distilled)
specifically for the minilm_distil_experiments folder.

Results are saved in:
- fairness/analysis_results/minilm_distillation_per_patient/
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
parser = argparse.ArgumentParser(
    description='Run all DISTILLATION fairness analyzers for MiniLM experiments',
    epilog='Results saved in fairness/analysis_results/minilm_distillation_per_patient/'
)
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

print("=" * 80)
print("RUNNING DISTILLATION FAIRNESS ANALYZERS (MiniLM EXPERIMENTS)")
print("=" * 80)

print("\n📊 Analyzing minilm_distil_experiments:")
print("  - Per-patient distillation experiments")
print("  - Results saved in: fairness/analysis_results/minilm_distillation_per_patient/")

print()

results = {}

for name, script in analyzers.items():
    print(f"\n🔍 Running {name} Analyzer (MiniLM)...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [python_cmd, script, "--experiment-type", "per_patient", "--experiments-folder", "minilm_distil_experiments"],
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
print(f"FINAL RESULT: {passed}/{total} analyzers passed (MiniLM experiments)")
print("=" * 80)

sys.exit(0 if passed == total else 1)
