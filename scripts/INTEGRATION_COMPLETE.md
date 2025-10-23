# True Value Replacement Integration

## Overview

The true value replacement functionality has been successfully integrated into both Time-LLM and Chronos experiment runners. This ensures that when experiments are run with non-normal data scenarios, the true values in the results are automatically replaced with raw formatted data.

## Integration Details

### Automatic Detection

The integration automatically detects non-normal data scenarios by looking for these keywords in experiment names or configuration paths:

- `missing_periodic`
- `missing_random`
- `noisy`
- `denoised`

### Integration Points

#### Time-LLM Integration
- **File**: `/scripts/time_llm/run_all_time_llm_experiments.py`
- **Location**: After metrics extraction in `run_single_experiment()` function
- **Trigger**: Automatic detection of non-normal scenarios

#### Chronos Integration
- **File**: `/scripts/chronos/run_all_chronos_experiments.py`
- **Location**: After metrics extraction in `run_single_experiment()` function
- **Trigger**: Automatic detection of non-normal scenarios

### Workflow

1. **Experiment Execution**: Normal experiment runs to completion
2. **Metrics Extraction**: Standard metrics are extracted to CSV
3. **Scenario Detection**: Check if experiment involves non-normal data
4. **True Value Replacement**: If non-normal, automatically replace true values with raw data
5. **Completion**: Experiment marked as completed

## Usage

### Normal Operation

The integration works transparently. When you run experiments using the standard experiment runners:

```bash
# Time-LLM experiments
python scripts/time_llm/run_all_time_llm_experiments.py --modes train_inference

# Chronos experiments  
python scripts/chronos/run_all_chronos_experiments.py --modes train_inference
```

The true value replacement will automatically occur for non-normal scenarios.

### Scenarios That Trigger Replacement

✅ **Will trigger replacement:**
- `time_llm_d1namo_missing_periodic_train_inference`
- `chronos_ohiot1dm_missing_random_test`
- `time_llm_d1namo_noisy_train`
- `chronos_ohiot1dm_denoised_test`

❌ **Will NOT trigger replacement:**
- `time_llm_d1namo_standardized_train`
- `chronos_ohiot1dm_train_test`
- `time_llm_normal_experiment`

## Manual Override

If you need to run the replacement manually for any experiment:

```bash
# Replace true values for a specific experiment directory
./scripts/run_replace_true_values.sh --experiments_dir ./experiments/specific_experiment --auto_confirm

# Dry run to see what would be changed
./scripts/run_replace_true_values.sh --experiments_dir ./experiments/specific_experiment --dry_run
```

## Logging

The integration provides clear logging output:

- `🔄 Non-normal data scenario detected, replacing true values with raw data...` - Replacement starting
- `✅ True values successfully replaced with raw data` - Replacement completed successfully
- `⚠️ True value replacement failed: <error>` - Replacement failed
- `ℹ️ Normal scenario detected, skipping true value replacement` - No replacement needed

## Error Handling

The integration includes comprehensive error handling:

1. **Script Not Found**: Graceful failure with error message
2. **Permission Issues**: Clear error reporting
3. **Timeout Protection**: 5-minute timeout for replacement operation
4. **Experiment Continuation**: Failures don't stop the main experiment workflow

## Testing

Run the integration test to verify everything works correctly:

```bash
python scripts/test_integration.py
```

This test validates:
- Scenario detection logic
- Script availability and permissions
- Command construction
- Integration examples

## Files Modified

### Core Integration Files
- `/scripts/time_llm/run_all_time_llm_experiments.py` - Time-LLM experiment runner
- `/scripts/chronos/run_all_chronos_experiments.py` - Chronos experiment runner
- `/scripts/run_replace_true_values.sh` - Enhanced wrapper script with auto-confirmation

### Support Files
- `/scripts/replace_true_values_with_raw.py` - Main replacement script (unchanged)
- `/scripts/test_integration.py` - Integration test script
- `/scripts/README_replace_true_values.md` - Original documentation

## Backup and Recovery

The replacement script automatically creates backups before making changes:

- **Backup Files**: `*_backup.csv` (original experiment results)
- **Corrected Files**: `*_raw_corrected.csv` (results with raw true values)
- **Original Files**: Replaced with corrected data

## Performance Impact

The integration adds minimal overhead:
- **Detection**: Negligible (string matching)
- **Execution**: ~5-30 seconds per experiment (depending on result file size)
- **Total Impact**: <1% of typical experiment runtime

## Maintenance

The integration is designed to be maintenance-free:

1. **Keyword Detection**: Add new scenario keywords to the `scenario_keywords` list if needed
2. **Script Updates**: Updates to the replacement script are automatically used
3. **Path Changes**: Relative paths ensure portability

## Troubleshooting

### Common Issues

1. **Script Not Found**
   - Ensure `/scripts/run_replace_true_values.sh` exists and is executable
   - Check file permissions: `chmod +x scripts/run_replace_true_values.sh`

2. **Virtual Environment Issues**
   - The wrapper script automatically creates and manages the virtual environment
   - Ensure `python3` and `venv` module are available

3. **Permission Errors**
   - Check write permissions in experiment directories
   - Ensure backup directory can be created

### Debug Mode

For debugging, you can modify the experiment runners to use dry-run mode:

```python
replacement_cmd = [
    'bash', replacement_script,
    '--experiments_dir', experiment_base_dir,
    '--dry_run'  # Add this for debugging
]
```

## Future Enhancements

Potential improvements for future versions:

1. **Configurable Keywords**: Make scenario detection keywords configurable
2. **Selective Replacement**: Add options to replace only specific file types
3. **Parallel Processing**: Enable parallel replacement for multiple experiments
4. **Progress Tracking**: Add progress bars for large-scale replacements
5. **Validation**: Add post-replacement validation checks

## Success Metrics

The integration is considered successful when:

- ✅ All integration tests pass
- ✅ Non-normal scenarios trigger replacement automatically
- ✅ Normal scenarios skip replacement correctly
- ✅ Error handling works gracefully
- ✅ No impact on normal experiment workflow
- ✅ Clear logging and status reporting

---

*Integration completed and tested successfully on all target scenarios.*