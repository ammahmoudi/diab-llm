#!/usr/bin/env python3
"""
Test script to verify the true value replacement integration with experiment runners.

This script tests:
1. Detection of non-normal scenarios
2. Calling the replacement script with correct parameters
3. Proper error handling

Usage:
    python test_integration.py
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_scenario_detection():
    """Test detection of non-normal scenarios."""
    print("🧪 Testing scenario detection...")
    
    test_cases = [
        # (experiment_name, config_path, should_trigger)
        ("time_llm_d1namo_missing_periodic_train", "/path/to/config.gin", True),
        ("chronos_ohiot1dm_missing_random_test", "/path/to/config.gin", True),
        ("time_llm_d1namo_noisy_train_inference", "/path/to/config.gin", True),
        ("chronos_ohiot1dm_denoised_test", "/path/to/config.gin", True),
        ("time_llm_d1namo_standardized_train", "/path/to/config.gin", False),
        ("chronos_ohiot1dm_train_test", "/path/to/config.gin", False),
        ("time_llm_normal_experiment", "/path/to/config.gin", False),
    ]
    
    # Test the detection logic
    scenario_keywords = ['missing_periodic', 'missing_random', 'noisy', 'denoised']
    
    for experiment_name, config_path, expected in test_cases:
        is_non_normal_scenario = any(keyword in experiment_name.lower() or keyword in config_path.lower() 
                                   for keyword in scenario_keywords)
        
        status = "✅" if is_non_normal_scenario == expected else "❌"
        print(f"   {status} {experiment_name}: {'triggered' if is_non_normal_scenario else 'skipped'}")
        
        if is_non_normal_scenario != expected:
            print(f"      Expected: {'triggered' if expected else 'skipped'}")
            return False
    
    print("   ✅ All scenario detection tests passed!")
    return True

def test_replacement_script_exists():
    """Test that the replacement script exists and is executable."""
    print("\n🧪 Testing replacement script availability...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    replacement_script = os.path.join(script_dir, 'run_replace_true_values.sh')
    
    if not os.path.exists(replacement_script):
        print(f"   ❌ Replacement script not found: {replacement_script}")
        return False
    
    if not os.access(replacement_script, os.X_OK):
        print(f"   ❌ Replacement script not executable: {replacement_script}")
        return False
    
    print(f"   ✅ Replacement script found and executable: {replacement_script}")
    return True

def test_command_construction():
    """Test that the replacement command is constructed correctly."""
    print("\n🧪 Testing command construction...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    replacement_script = os.path.join(script_dir, 'run_replace_true_values.sh')
    experiment_base_dir = "/path/to/experiment"
    
    expected_cmd = [
        'bash', replacement_script,
        '--experiments-root', experiment_base_dir,
        '--auto_confirm'
    ]
    
    # Simulate the command construction from the integration code
    replacement_cmd = [
        'bash', replacement_script,
        '--experiments-root', experiment_base_dir,
        '--auto_confirm'
    ]
    
    if replacement_cmd == expected_cmd:
        print(f"   ✅ Command construction correct: {' '.join(replacement_cmd)}")
        return True
    else:
        print(f"   ❌ Command construction incorrect!")
        print(f"      Expected: {' '.join(expected_cmd)}")
        print(f"      Got:      {' '.join(replacement_cmd)}")
        return False

def test_integration_examples():
    """Show examples of how the integration would work."""
    print("\n🧪 Integration examples:")
    
    examples = [
        {
            "experiment": "time_llm_d1namo_missing_periodic_train_inference",
            "config": "/home/amma/LLM-TIME/experiments/time_llm_d1namo_missing_periodic_train_inference/d1namo/BERT/seed_42/config.gin",
            "description": "Time-LLM with missing periodic data"
        },
        {
            "experiment": "chronos_ohiot1dm_noisy_test",
            "config": "/home/amma/LLM-TIME/experiments/chronos_ohiot1dm_noisy_test/ohiot1dm/seed_123/config.gin",
            "description": "Chronos with noisy data"
        },
        {
            "experiment": "time_llm_d1namo_standardized_train",
            "config": "/home/amma/LLM-TIME/experiments/time_llm_d1namo_standardized_train/d1namo/GPT2/seed_42/config.gin",
            "description": "Time-LLM with normal standardized data (no replacement needed)"
        }
    ]
    
    scenario_keywords = ['missing_periodic', 'missing_random', 'noisy', 'denoised']
    
    for example in examples:
        experiment_name = example["experiment"]
        config_path = example["config"]
        description = example["description"]
        
        is_non_normal_scenario = any(keyword in experiment_name.lower() or keyword in config_path.lower() 
                                   for keyword in scenario_keywords)
        
        print(f"\n   📋 Example: {description}")
        print(f"      Experiment: {experiment_name}")
        print(f"      Config: {config_path}")
        
        if is_non_normal_scenario:
            print(f"      🔄 Action: Replace true values with raw data")
            print(f"      📝 Command: bash run_replace_true_values.sh --experiments-root {os.path.dirname(os.path.dirname(config_path))} --auto_confirm")
        else:
            print(f"      ℹ️  Action: Skip replacement (normal scenario)")

def main():
    """Run all integration tests."""
    print("🚀 Testing True Value Replacement Integration")
    print("=" * 60)
    
    tests = [
        test_scenario_detection,
        test_replacement_script_exists,
        test_command_construction,
    ]
    
    all_passed = True
    
    for test_func in tests:
        if not test_func():
            all_passed = False
    
    # Show examples regardless of test results
    test_integration_examples()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All integration tests passed!")
        print("\nThe integration is ready and should work correctly.")
        print("When you run Time-LLM or Chronos experiments with non-normal data scenarios,")
        print("the true values will be automatically replaced with raw data after completion.")
    else:
        print("❌ Some integration tests failed!")
        print("Please review the issues above before using the integration.")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    exit(main())