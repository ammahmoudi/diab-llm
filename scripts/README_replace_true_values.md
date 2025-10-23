# True Values Replacement Script

## Overview
This script replaces the true values in experiment CSV files (both Chronos and Time-LLM) with the actual raw formatted data values. The current experiment results contain true values from denoised or noisy data, but they should reference the original raw formatted data.

## Files
- `replace_true_values_with_raw.py` - Main Python script
- `run_replace_true_values.sh` - Simple bash wrapper script
- `README_replace_true_values.md` - This documentation

## What the Script Does

1. **Identifies Experiment Types**: Automatically detects Chronos and Time-LLM experiments
2. **Locates CSV Files**: 
   - Chronos: `final_results.csv`
   - Time-LLM: `inference_results_reformatted.csv`
3. **Extracts Patient and Configuration**: Gets patient ID and prediction horizon from file paths
4. **Loads Raw Data**: Retrieves corresponding raw formatted data from `/data/ohiot1dm/raw_formatted/`
5. **Replaces True Values**: Updates all `t_X_true` columns with raw data values
6. **Creates New Files**: Saves corrected data as `raw_corrected_*.csv` files
7. **Creates Backups**: Original files are backed up as `*.csv.backup`

## Usage

### Basic Usage (Process All Experiments)
```bash
# From the scripts directory
./run_replace_true_values.sh

# Or run Python script directly
python3 replace_true_values_with_raw.py
```

### Advanced Usage
```bash
# Process specific experiments
python3 replace_true_values_with_raw.py --experiments chronos_trained_inference_ohiot1dm_noisy time_llm_training_inference_ohiot1dm_train_standardized_test_noisy

# Use different data/experiment paths
python3 replace_true_values_with_raw.py --data-root /path/to/data --experiments-root /path/to/experiments

# Don't create backup files
python3 replace_true_values_with_raw.py --no-backup

# Get help
python3 replace_true_values_with_raw.py --help
```

## Data Structure Assumptions

### Raw Data Format
- Located in `/data/ohiot1dm/raw_formatted/6_6/` (or 6_9, 9_9 for other configurations)
- Files named as `{patient_id}-ws-testing.csv`
- Columns: `BG_{t-5}`, `BG_{t-4}`, ..., `BG_{t}`, `BG_{t+1}`, ..., `BG_{t+6}`

### Experiment Results Format
- **Chronos**: `final_results.csv` with columns like `t_6_true`, `t_6_pred`, `t_7_true`, `t_7_pred`, etc.
- **Time-LLM**: `inference_results_reformatted.csv` with same column format

### Directory Structure Expected
```
experiments/
├── chronos_trained_inference_ohiot1dm_*/
│   └── chronos_trained_inference_ohiot1dm_*/
│       └── seed_*/
│           └── patient_*/
│               └── logs/
│                   └── logs_*/
│                       └── final_results.csv
└── time_llm_training_inference_ohiot1dm_*/
    └── time_llm_training_inference_ohiot1dm_*/
        └── seed_*/
            └── patient_*/
                └── logs/
                    └── logs_*/
                        └── inference_results_reformatted.csv
```

## Output

### New Files Created
- `raw_corrected_final_results.csv` - Corrected Chronos results
- `raw_corrected_inference_results_reformatted.csv` - Corrected Time-LLM results

### Backup Files
- `final_results.csv.backup` - Original Chronos results
- `inference_results_reformatted.csv.backup` - Original Time-LLM results

## Features

### Automatic Detection
- Detects patient IDs from directory names (`patient_540`, etc.)
- Extracts prediction horizon from experiment paths (`pred_6`, `context_6_pred_6`, etc.)
- Handles different configurations (6_6, 6_9, 9_9)

### Error Handling
- Validates file existence before processing
- Handles missing raw data gracefully
- Reports progress and errors clearly
- Pads with NaN values if insufficient raw data

### Data Caching
- Caches loaded raw data to improve performance when processing multiple experiments for the same patient

## Troubleshooting

### Common Issues
1. **"Raw data file not found"** - Check that raw formatted data exists in expected location
2. **"No true value columns found"** - CSV file may not have expected format
3. **"Not enough raw values"** - Raw data may be shorter than expected; script will pad with NaN

### Verification
After running the script, verify that:
1. New `raw_corrected_*.csv` files are created
2. True value columns contain values from raw data
3. Prediction columns remain unchanged
4. Backup files exist

## Notes
- The script preserves all prediction values unchanged
- Only true value columns are modified
- Original files are backed up by default
- The script is designed to be idempotent (safe to run multiple times)