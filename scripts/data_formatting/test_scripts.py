#!/usr/bin/env python3
"""
Test script to verify data formatting scripts work correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_script_help(script_path, script_name):
    """Test that a script shows help correctly."""
    try:
        result = subprocess.run([sys.executable, script_path, '--help'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"✅ {script_name} help works")
            return True
        else:
            print(f"❌ {script_name} help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ {script_name} test failed: {e}")
        return False

def main():
    print("🧪 Testing Data Formatting Scripts")
    print("=" * 40)
    
    scripts_dir = Path(__file__).parent
    
    # Test scripts
    scripts = [
        (scripts_dir / "complete_data_pipeline.py", "Complete Pipeline"),
        (scripts_dir / "quick_process.py", "Quick Process")
    ]
    
    success_count = 0
    total_count = len(scripts)
    
    for script_path, script_name in scripts:
        if script_path.exists():
            if test_script_help(script_path, script_name):
                success_count += 1
        else:
            print(f"❌ {script_name} not found: {script_path}")
    
    print("=" * 40)
    print(f"📊 Results: {success_count}/{total_count} scripts working")
    
    if success_count == total_count:
        print("🎉 All data formatting scripts are working correctly!")
        return 0
    else:
        print("⚠️  Some scripts have issues. Check output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())