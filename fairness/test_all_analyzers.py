#!/usr/bin/env python3
"""
Test script to validate all fairness analyzers
"""

import subprocess
import sys

analyzers = [
    ("Gender", "gender_fairness_analyzer.py"),
    ("Age", "age_fairness_analyzer.py"),
    ("Pump Model", "pump_model_fairness_analyzer.py"),
    ("Sensor Band", "sensor_fairness_analyzer.py"),
    ("Cohort", "cohort_fairness_analyzer.py"),
    ("Legendary", "legendary_fairness_analyzer.py"),
]

python_cmd = "/workspace/LLM-TIME/venv/bin/python"

print("=" * 80)
print("TESTING ALL FAIRNESS ANALYZERS")
print("=" * 80)

results = {}

for name, script in analyzers:
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
