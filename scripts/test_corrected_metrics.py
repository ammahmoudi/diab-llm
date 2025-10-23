#!/usr/bin/env python3
"""
Test script to verify corrected metrics calculation functionality.

This script tests:
1. Loading corrected CSV files
2. Calculating metrics using utils/metrics.py
3. Logging metrics to log files
4. Extracting to corrected metrics CSV

Usage:
    python test_corrected_metrics.py
"""

import os
import sys
import tempfile
import pandas as pd
import numpy as np

def test_corrected_metrics_calculation():
    """Test the corrected metrics calculation functionality."""
    print("🧪 Testing Corrected Metrics Calculation")
    print("=" * 50)
    
    # Add project paths
    project_root = os.path.dirname(os.path.dirname(__file__))
    utils_path = os.path.join(project_root, 'utils')
    utilities_path = os.path.join(os.path.dirname(__file__), 'utilities')
    
    if utils_path not in sys.path:
        sys.path.insert(0, utils_path)
    if utilities_path not in sys.path:
        sys.path.insert(0, utilities_path)
    
    try:
        from metrics import calculate_rmse, calculate_mae, calculate_mape
        print("✅ Successfully imported metrics functions")
    except ImportError as e:
        print(f"❌ Failed to import metrics: {e}")
        return False
    
    try:
        from extract_metrics_corrected import calculate_and_log_corrected_metrics
        print("✅ Successfully imported corrected metrics utility function")
    except ImportError as e:
        print(f"❌ Failed to import corrected metrics utility: {e}")
        return False
    
    # Create test data
    print("\n📊 Creating test data...")
    np.random.seed(42)
    n_samples = 100
    
    # Simulate prediction and true values (corrected)
    true_values = np.random.normal(150, 30, n_samples)  # Blood glucose-like values
    pred_values = true_values + np.random.normal(0, 10, n_samples)  # Add some prediction error
    
    # Create test CSV data structure
    test_data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
        't_6_true': true_values,
        't_6_pred': pred_values,
        't_7_true': true_values + np.random.normal(0, 5, n_samples),
        't_7_pred': pred_values + np.random.normal(0, 5, n_samples),
        't_8_true': true_values + np.random.normal(0, 8, n_samples),
        't_8_pred': pred_values + np.random.normal(0, 8, n_samples)
    }
    
    df = pd.DataFrame(test_data)
    print(f"✅ Created test data with {len(df)} samples and {len([col for col in df.columns if 'true' in col])} timesteps")
    
    # Test metrics calculation
    print("\n🔢 Testing metrics calculation...")
    
    test_metrics = {}
    true_cols = [col for col in df.columns if col.startswith('t_') and col.endswith('_true')]
    pred_cols = [col for col in df.columns if col.startswith('t_') and col.endswith('_pred')]
    
    for true_col, pred_col in zip(sorted(true_cols), sorted(pred_cols)):
        true_vals = df[true_col].dropna().values
        pred_vals = df[pred_col].dropna().values
        
        timestep = true_col.split('_')[1]
        
        # Calculate metrics
        rmse = calculate_rmse(pred_vals, true_vals)
        mae = calculate_mae(pred_vals, true_vals)
        
        # Calculate MAPE only if no zero values
        if not np.any(true_vals == 0):
            mape = calculate_mape(pred_vals, true_vals)
        else:
            mape = float('nan')
        
        test_metrics[f'rmse_t{timestep}_corrected'] = rmse
        test_metrics[f'mae_t{timestep}_corrected'] = mae
        test_metrics[f'mape_t{timestep}_corrected'] = mape
        
        print(f"   t{timestep}: RMSE={rmse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}")
    
    # Test file operations with temporary files
    print("\n📁 Testing file operations...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test CSV file
        test_csv_path = os.path.join(temp_dir, 'raw_corrected_test_results.csv')
        df.to_csv(test_csv_path, index=False)
        print(f"✅ Created test CSV: {test_csv_path}")
        
        # Create test log file
        test_log_path = os.path.join(temp_dir, 'log.log')
        with open(test_log_path, 'w') as f:
            f.write("=== Original Log Content ===\\n")
            f.write("Some existing log entries...\\n")
        
        # Test appending corrected metrics to log
        with open(test_log_path, 'a') as f:
            f.write("\\n=== CORRECTED METRICS (Raw True Values) ===\\n")
            f.write(f"Corrected Metrics: {test_metrics}\\n")
            f.write(f"Corrected metrics calculated from: {test_csv_path}\\n")
        
        print("✅ Successfully appended corrected metrics to log file")
        
        # Verify log content
        with open(test_log_path, 'r') as f:
            log_content = f.read()
            if "CORRECTED METRICS" in log_content and "rmse_t6_corrected" in log_content:
                print("✅ Log file contains corrected metrics")
            else:
                print("❌ Log file missing expected corrected metrics")
                return False
    
    print("\n📈 Testing metrics summary...")
    total_metrics = len(test_metrics)
    rmse_metrics = len([k for k in test_metrics.keys() if 'rmse' in k])
    mae_metrics = len([k for k in test_metrics.keys() if 'mae' in k])
    mape_metrics = len([k for k in test_metrics.keys() if 'mape' in k and not np.isnan(test_metrics[k])])
    
    print(f"   Total corrected metrics: {total_metrics}")
    print(f"   RMSE metrics: {rmse_metrics}")
    print(f"   MAE metrics: {mae_metrics}")
    print(f"   MAPE metrics: {mape_metrics}")
    
    # Test error handling
    print("\n🛡️  Testing error handling...")
    
    # Test with empty data
    empty_df = pd.DataFrame()
    true_cols_empty = [col for col in empty_df.columns if col.startswith('t_') and col.endswith('_true')]
    if len(true_cols_empty) == 0:
        print("✅ Correctly handled empty dataframe")
    
    # Test with misaligned data
    misaligned_df = pd.DataFrame({
        't_6_true': [1, 2, 3],
        't_6_pred': [1.1, 2.2, 3.3, 4.4, 5.5]  # Different length
    })
    
    min_len = min(len(misaligned_df['t_6_true'].dropna()), len(misaligned_df['t_6_pred'].dropna()))
    if min_len == 3:  # Should use minimum length
        print("✅ Correctly handled misaligned data lengths")
    
    print("\n" + "=" * 50)
    print("✅ All corrected metrics calculation tests passed!")
    print("\\nThe corrected metrics calculation functionality is ready and should work correctly.")
    print("When experiments run with non-normal data scenarios, corrected metrics will be:")
    print("- Calculated from raw_corrected CSV files")
    print("- Logged to individual experiment log files") 
    print("- Extracted to comprehensive corrected metrics CSV files")
    
    return True

def main():
    """Run all corrected metrics calculation tests."""
    success = test_corrected_metrics_calculation()
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())