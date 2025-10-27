#!/usr/bin/env python3
"""
Test script to validate all fairness analyzers
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).resolve().parent
fairness_dir = current_dir.parent
project_root = fairness_dir.parent
sys.path.insert(0, str(project_root))

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
print("TESTING ALL FAIRNESS ANALYZERS")
print("=" * 80)

results = {}

for name, script in analyzers.items():
    print(f"\n🧪 Testing {name} Analyzer...")
    print("-" * 80)
    
    try:
        result = subprocess.run(
            [python_cmd, script],
            capture_output=True,
            text=True,
            timeout=30
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
print("TEST SUMMARY")
print("=" * 80)

for name, status in results.items():
    icon = "✅" if status == "PASSED" else "❌"
    print(f"{icon} {name:20s}: {status}")

passed = sum(1 for s in results.values() if s == "PASSED")
total = len(results)

print("\n" + "=" * 80)
print(f"FINAL RESULT: {passed}/{total} analyzers passed")
print("=" * 80)

sys.exit(0 if passed == total else 1)
