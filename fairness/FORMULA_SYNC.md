# Formula Synchronization Between Scripts

This document ensures both `investigate_fairness_issues.py` and `analyze_training_impact.py` use identical formulas and thresholds.

## Constants (Synced)

Both scripts now use:
```python
self.POOR_RATIO = 1.5  # Fairness ratio above this is POOR
self.SIGNIFICANT_DEGRADATION = 0.1  # 10% degradation is significant
```

## Degradation Percentage Formula (Synced)

**Formula**: `(degradation / baseline_ratio) * 100`

### In investigate_fairness_issues.py:
- **Line 96** (Distillation): `(degradation / teacher_ratio) * 100`
- **Line 199** (Inference): `(degradation / inference_only_ratio) * 100`

### In analyze_training_impact.py:
- **Line 76**: `(fairness_change / inference_only['fairness_ratio']) * 100`

## Chart Averaging Logic (Synced)

Both scripts calculate the "Training Impact" chart by:
1. Only including features where `degradation >= SIGNIFICANT_DEGRADATION (0.1)`
2. Averaging the `degradation_pct` values for those features only

### Example for Trained Standard:
- Gender: 1.036 → 1.157 = +0.121 degradation (+11.7%)  ✓ Included (≥0.1)
- Pump Model: 1.034 → 1.266 = +0.232 degradation (+22.4%)  ✓ Included (≥0.1)
- Cohort: 1.006 → 1.073 = +0.067 degradation (+6.7%)  ✗ Excluded (<0.1)
- Sensor Band: 1.180 → 1.236 = +0.056 degradation (+4.8%)  ✗ Excluded (<0.1)
- Age Group: 1.089 → 1.093 = +0.004 degradation (+0.3%)  ✗ Excluded (<0.1)

**Average (Significant Only)**: (11.7 + 22.4) / 2 = **17.1%**

## Visual Improvements

### analyze_training_impact.py improvements:
1. **Compact y-axis labels** in Top 15 chart: `"Feature: Group (Scenario)"` instead of multi-line
2. **Reduced font size** for y-axis: `labelsize=8`
3. **Clear chart subtitle**: "(Features with degradation ≥0.1 only)"
4. **Feature markers**: ★ indicates features included in the average

## Verification

Both scripts now produce consistent results:
- Same degradation percentages for each feature
- Same "significant degradation" threshold (0.1)
- Same averaging logic for charts
- Same fairness ratio thresholds
