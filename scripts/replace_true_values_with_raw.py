#!/usr/bin/env python3
"""
Script to replace true values in experiment CSV files with raw formatted data.

This script processes both Chronos and Time-LLM experiment results and replaces
the true values (which are currently from noisy/denoised data) with the corresponding
raw formatted data values.

Setup:
    # Create and activate virtual environment
    python3 -m venv venv
    source venv/bin/activate  # On Linux/Mac
    # or
    venv\\Scripts\\activate     # On Windows
    
    # Install required packages
    pip install pandas numpy

Usage:
    python replace_true_values_with_raw.py [--dry-run] [--data-dir DATA_DIR] [--experiments-dir EXPERIMENTS_DIR]

Example:
    python replace_true_values_with_raw.py --dry-run
    python replace_true_values_with_raw.py --data-dir ./data --experiments-dir ./experiments
"""

import os
import pandas as pd
import numpy as np
import glob
from pathlib import Path
import shutil
import argparse
from typing import Dict, List, Tuple, Optional
import re


class TrueValueReplacer:
    """Class to handle replacement of true values in experiment CSV files."""
    
    def __init__(self, data_root: Path, experiments_root: Path, dry_run: bool = False):
        self.data_root = Path(data_root)
        self.experiments_root = Path(experiments_root)
        self.dry_run = dry_run
        
        # Raw data directory structure
        self.raw_data_dir = self.data_root / "ohiot1dm" / "raw_formatted"
        
        # Cache for raw data to avoid repeated loading
        self.raw_data_cache = {}
        
    def load_raw_data(self, patient_id: str, context_pred_config: str = "6_6") -> Optional[pd.DataFrame]:
        """
        Load raw formatted data for a specific patient.
        
        Args:
            patient_id: Patient ID (e.g., "540")
            context_pred_config: Configuration like "6_6", "6_9", or "9_9"
            
        Returns:
            DataFrame with raw data or None if not found
        """
        cache_key = f"{patient_id}_{context_pred_config}"
        
        if cache_key in self.raw_data_cache:
            return self.raw_data_cache[cache_key]
            
        raw_file_path = self.data_root / "ohiot1dm" / "raw_formatted" / context_pred_config / f"{patient_id}-ws-testing.csv"
        
        if not raw_file_path.exists():
            print(f"Warning: Raw data file not found: {raw_file_path}")
            return None
            
        try:
            df = pd.read_csv(raw_file_path)
            self.raw_data_cache[cache_key] = df
            print(f"Loaded raw data for patient {patient_id} from {raw_file_path}")
            return df
        except Exception as e:
            print(f"Error loading raw data from {raw_file_path}: {e}")
            return None
    
    def extract_true_values_from_raw(self, raw_df: pd.DataFrame, prediction_horizon: int) -> Dict:
        """
        Extract true values from raw data for the given prediction horizon.
        
        Args:
            raw_df: Raw data DataFrame
            prediction_horizon: Number of prediction steps
            
        Returns:
            Dict mapping column names to their values for replacement
        """
        # For 6_6 config, we want the prediction columns (t+1 to t+6)
        # These are columns 6-11 in the raw data (0-indexed)
        prediction_cols = []
        for i in range(prediction_horizon):
            col_name = f"BG_{{t+{i+1}}}"
            if col_name in raw_df.columns:
                prediction_cols.append(col_name)
        
        if not prediction_cols:
            print(f"Warning: No prediction columns found for horizon {prediction_horizon}")
            return {}
        
        # Return each prediction column as a separate array
        result = {}
        for i, col in enumerate(prediction_cols):
            # Map to experiment column names: t_6_true, t_7_true, etc.
            exp_col_name = f"t_{6+i}_true"
            result[exp_col_name] = raw_df[col].tolist()
        
        return result
    
    def replace_true_values_in_csv(self, csv_path: Path, raw_true_values: Dict, 
                                 prediction_horizon: int, backup: bool = True) -> bool:
        """
        Replace true values in experiment CSV file with raw values.
        
        Args:
            csv_path: Path to the experiment CSV file
            raw_true_values: Dict mapping column names to their raw values
            prediction_horizon: Number of prediction steps
            backup: Whether to create a backup of the original file
            
        Returns:
            True if successful, False otherwise
        """
        if not csv_path.exists():
            print(f"Warning: CSV file not found: {csv_path}")
            return False
            
        try:
            # Create backup if requested
            if backup and not self.dry_run:
                backup_path = csv_path.with_suffix('.csv.backup')
                shutil.copy2(csv_path, backup_path)
                print(f"Created backup: {backup_path}")
            elif backup and self.dry_run:
                backup_path = csv_path.with_suffix('.csv.backup')
                print(f"[DRY RUN] Would create backup: {backup_path}")
            
            # Load the experiment results
            df = pd.read_csv(csv_path)
            
            # Identify true value columns (e.g., t_6_true, t_7_true, ...)
            true_columns = [col for col in df.columns if col.endswith('_true')]
            
            if not true_columns:
                print(f"Warning: No true value columns found in {csv_path}")
                return False
            
            total_rows = len(df)
            print(f"Processing {csv_path}")
            print(f"  Experiment rows: {total_rows}, True columns: {len(true_columns)}")
            
            replaced_count = 0
            
            # Replace each true column with corresponding raw values
            for col in true_columns:
                if col in raw_true_values:
                    raw_values = raw_true_values[col]
                    print(f"  Replacing {col}: experiment has {total_rows} rows, raw data has {len(raw_values)} values")
                    
                    # Handle length mismatch
                    if len(raw_values) >= total_rows:
                        # Use the first total_rows values from raw data
                        df[col] = raw_values[:total_rows]
                        replaced_count += 1
                    else:
                        # Pad with NaN if raw data is shorter
                        padded_values = raw_values + [np.nan] * (total_rows - len(raw_values))
                        df[col] = padded_values
                        replaced_count += 1
                        print(f"    Warning: Padded {total_rows - len(raw_values)} values with NaN")
                else:
                    print(f"    Warning: No raw data found for column {col}")
            
            print(f"  Replaced {replaced_count}/{len(true_columns)} true value columns")
            
            # Save the modified CSV
            new_csv_path = csv_path.with_name(f"raw_corrected_{csv_path.name}")
            
            if self.dry_run:
                print(f"[DRY RUN] Would create corrected file: {new_csv_path}")
            else:
                df.to_csv(new_csv_path, index=False)
                print(f"Created corrected file: {new_csv_path}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            return False
    
    def extract_patient_and_config_from_path(self, path: Path) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract patient ID and context/prediction configuration from experiment path.
        
        Args:
            path: Path to experiment directory or file
            
        Returns:
            Tuple of (patient_id, config) or (None, None) if not found
        """
        # Look for patient_XXX in the path
        patient_match = None
        for part in path.parts:
            if part.startswith('patient_'):
                patient_match = part.replace('patient_', '')
                break
                
        if not patient_match:
            return None, None
            
        # Try to extract context and prediction horizon from path
        # Look for patterns like "context_6_pred_6" or "pred_6"
        config = "6_6"  # Default
        path_str = str(path)
        
        # Try to find context and pred values
        context_match = re.search(r'context_(\d+)', path_str)
        pred_match = re.search(r'pred_(\d+)', path_str)
        
        if context_match and pred_match:
            context = context_match.group(1)
            pred = pred_match.group(1)
            config = f"{context}_{pred}"
        elif pred_match:
            pred = pred_match.group(1)
            config = f"6_{pred}"  # Assume context is 6 if not specified
            
        return patient_match, config
    
    def process_chronos_experiments(self, experiment_name: str) -> int:
        """
        Process all Chronos experiment results.
        
        Args:
            experiment_name: Name of the Chronos experiment folder
            
        Returns:
            Number of files processed successfully
        """
        print(f"\n=== Processing Chronos Experiment: {experiment_name} ===")
        
        experiment_path = self.experiments_root / experiment_name / experiment_name
        if not experiment_path.exists():
            print(f"Experiment path not found: {experiment_path}")
            return 0
            
        processed_count = 0
        
        # Find all final_results.csv files
        final_results_files = list(experiment_path.glob("**/final_results.csv"))
        
        print(f"Found {len(final_results_files)} final_results.csv files")
        
        for csv_file in final_results_files:
            patient_id, config = self.extract_patient_and_config_from_path(csv_file)
            
            if not patient_id:
                print(f"Could not extract patient ID from {csv_file}")
                continue
                
            if not config:
                print(f"Could not extract config from {csv_file}")
                continue
                
            # At this point both patient_id and config are guaranteed to be not None
            assert patient_id is not None
            assert config is not None
            
            print(f"\nProcessing patient {patient_id} with config {config}")
            
            # Load raw data
            raw_df = self.load_raw_data(patient_id, config)
            if raw_df is None:
                continue
                
            # Extract prediction horizon from config
            try:
                pred_horizon = int(config.split('_')[1])
            except (IndexError, ValueError) as e:
                print(f"Error parsing prediction horizon from config {config}: {e}")
                continue
            
            # Extract true values from raw data
            raw_true_values = self.extract_true_values_from_raw(raw_df, pred_horizon)
            
            # Replace true values in CSV
            if self.replace_true_values_in_csv(csv_file, raw_true_values, pred_horizon):
                processed_count += 1
                
        return processed_count
    
    def process_time_llm_experiments(self, experiment_name: str) -> int:
        """
        Process all Time-LLM experiment results.
        
        Args:
            experiment_name: Name of the Time-LLM experiment folder
            
        Returns:
            Number of files processed successfully
        """
        print(f"\n=== Processing Time-LLM Experiment: {experiment_name} ===")
        
        experiment_path = self.experiments_root / experiment_name / experiment_name
        if not experiment_path.exists():
            print(f"Experiment path not found: {experiment_path}")
            return 0
            
        processed_count = 0
        
        # Find all inference_results_reformatted.csv files
        inference_files = list(experiment_path.glob("**/inference_results_reformatted.csv"))
        
        print(f"Found {len(inference_files)} inference_results_reformatted.csv files")
        
        for csv_file in inference_files:
            patient_id, config = self.extract_patient_and_config_from_path(csv_file)
            
            if not patient_id:
                print(f"Could not extract patient ID from {csv_file}")
                continue
                
            if not config:
                print(f"Could not extract config from {csv_file}")
                continue
                
            # At this point both patient_id and config are guaranteed to be not None
            assert patient_id is not None
            assert config is not None
            
            print(f"\nProcessing patient {patient_id} with config {config}")
            
            # Load raw data
            raw_df = self.load_raw_data(patient_id, config)
            if raw_df is None:
                continue
                
            # Extract prediction horizon from config
            try:
                pred_horizon = int(config.split('_')[1])
            except (IndexError, ValueError) as e:
                print(f"Error parsing prediction horizon from config {config}: {e}")
                continue
            
            # Extract true values from raw data
            raw_true_values = self.extract_true_values_from_raw(raw_df, pred_horizon)
            
            # Replace true values in CSV
            if self.replace_true_values_in_csv(csv_file, raw_true_values, pred_horizon):
                processed_count += 1
                
        return processed_count
    
    def process_all_experiments(self, experiment_patterns: Optional[List[str]] = None):
        """
        Process all experiments matching the given patterns.
        
        Args:
            experiment_patterns: List of experiment name patterns to match.
                                If None, processes common experiment types.
        """
        if experiment_patterns is None:
            experiment_patterns = [
                "chronos_trained_inference_ohiot1dm_*",
                "time_llm_training_inference_ohiot1dm_*"
            ]
        
        total_processed = 0
        
        for pattern in experiment_patterns:
            matching_experiments = list(self.experiments_root.glob(pattern))
            
            for exp_path in matching_experiments:
                if not exp_path.is_dir():
                    continue
                    
                exp_name = exp_path.name
                
                if "chronos" in exp_name.lower():
                    count = self.process_chronos_experiments(exp_name)
                elif "time_llm" in exp_name.lower():
                    count = self.process_time_llm_experiments(exp_name)
                else:
                    print(f"Unknown experiment type: {exp_name}")
                    continue
                    
                total_processed += count
                
        print(f"\n=== Summary ===")
        print(f"Total files processed successfully: {total_processed}")


def main():
    parser = argparse.ArgumentParser(
        description="Replace true values in experiment CSV files with raw formatted data"
    )
    parser.add_argument(
        "--data-root", 
        type=str, 
        default="/home/amma/LLM-TIME/data",
        help="Root directory containing the data folder"
    )
    parser.add_argument(
        "--experiments-root", 
        type=str, 
        default="/home/amma/LLM-TIME/experiments",
        help="Root directory containing the experiments folder"
    )
    parser.add_argument(
        "--experiments", 
        nargs="*",
        help="Specific experiment names to process (default: all)"
    )
    parser.add_argument(
        "--no-backup", 
        action="store_true",
        help="Don't create backup files"
    )
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Show what would be changed without modifying files"
    )
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.data_root):
        print(f"Error: Data root directory not found: {args.data_root}")
        return 1
        
    if not os.path.exists(args.experiments_root):
        print(f"Error: Experiments root directory not found: {args.experiments_root}")
        return 1
    
    # Create replacer instance
    replacer = TrueValueReplacer(args.data_root, args.experiments_root, dry_run=args.dry_run)
    
    # Process experiments
    try:
        if args.experiments:
            # Process specific experiments
            total_processed = 0
            for exp_name in args.experiments:
                if "chronos" in exp_name.lower():
                    count = replacer.process_chronos_experiments(exp_name)
                elif "time_llm" in exp_name.lower():
                    count = replacer.process_time_llm_experiments(exp_name)
                else:
                    print(f"Unknown experiment type: {exp_name}")
                    continue
                total_processed += count
            
            print(f"\nTotal files processed: {total_processed}")
        else:
            # Process all experiments
            replacer.process_all_experiments()
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        print(f"Error during processing: {e}")
        return 1
        
    return 0


if __name__ == "__main__":
    exit(main())