#!/usr/bin/env python3

import pandas as pd
import sys
import os

# Add the scripts directory to the path
sys.path.append('/home/amma/LLM-TIME/scripts')
from replace_true_values_with_raw import TrueValueReplacer

def verify_replacement():
    data_root = "/home/amma/LLM-TIME/data/ohiot1dm"
    experiments_root = "/home/amma/LLM-TIME/experiments"
    replacer = TrueValueReplacer(data_root, experiments_root)
    
    # Test file
    test_file = "/home/amma/LLM-TIME/experiments/time_llm_training_inference_ohiot1dm_train_standardized_test_denoised/time_llm_training_inference_ohiot1dm_train_standardized_test_denoised/seed_247659_model_BERT_dim_768_seq_6_context_6_pred_6_patch_6_epochs_10/patient_540/logs/logs_2025-10-13_21-21-55/inference_results_reformatted.csv"
    
    # Create a copy for testing
    test_copy = "/tmp/test_replacement.csv"
    os.system(f"cp '{test_file}' '{test_copy}'")
    
    print("=== BEFORE REPLACEMENT ===")
    df_before = pd.read_csv(test_copy)
    true_cols = [col for col in df_before.columns if 'true' in col]
    print("First 3 rows of true columns:")
    print(df_before[true_cols].head(3))
    
    # Perform the replacement
    patient_id = "540"
    config = "6_6" 
    replacer.replace_true_values_in_csv(test_copy, patient_id, config)
    
    print("\n=== AFTER REPLACEMENT ===")
    df_after = pd.read_csv(test_copy)
    print("First 3 rows of true columns:")
    print(df_after[true_cols].head(3))
    
    # Load raw data to verify exact match
    raw_file = f"/home/amma/LLM-TIME/data/ohiot1dm/raw_formatted/{config}/{patient_id}-ws-testing.csv"
    raw_df = pd.read_csv(raw_file)
    
    print("\n=== VERIFICATION ===")
    mappings = [
        ('BG_{t+1}', 't_6_true'),
        ('BG_{t+2}', 't_7_true'), 
        ('BG_{t+3}', 't_8_true'),
        ('BG_{t+4}', 't_9_true'),
        ('BG_{t+5}', 't_10_true'),
        ('BG_{t+6}', 't_11_true')
    ]
    
    for raw_col, exp_col in mappings:
        raw_vals = raw_df[raw_col].values[:3]  # First 3 values
        new_vals = df_after[exp_col].values[:3]  # First 3 values after replacement
        matches = all(raw_vals == new_vals)
        print(f"{raw_col} -> {exp_col}: {raw_vals} == {new_vals} -> {matches}")
    
    # Clean up
    os.remove(test_copy)
    
    return True

if __name__ == "__main__":
    verify_replacement()
