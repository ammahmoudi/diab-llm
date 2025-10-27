# Fairness Analysis Framework

**Comprehensive fairness analysis for knowledge distillation in diabetes prediction models**

## Overview

This framework provides tools to analyze fairness across multiple demographic dimensions in your distilled models. It compares Teacher (large model), Student Baseline (small model without distillation), and Distilled (small model with knowledge distillation) to understand how distillation affects fairness.

## Features

- ✅ **6 Demographic Analyzers**: Gender, Age Group, Pump Model, Sensor Band, Study Cohort, and Comprehensive (Legendary)
- ✅ **Multi-Phase Analysis**: Compare Teacher vs Student vs Distilled models
- ✅ **Per-Group Impact**: See which specific groups improved or worsened
- ✅ **JSON Reports**: Structured, machine-readable output
- ✅ **Visual Analysis**: 4-panel comprehensive charts
- ✅ **Embedded Tables**: Summary tables in visualizations (Legendary)
- ✅ **CSV Export**: Spreadsheet-ready per-group summaries

## Quick Start

### Run Individual Analyzer

```bash
# Analyze gender fairness
python fairness/analyzers/gender_fairness_analyzer.py

# Analyze age group fairness
python fairness/analyzers/age_fairness_analyzer.py

# Analyze pump model fairness
python fairness/analyzers/pump_model_fairness_analyzer.py

# Analyze sensor band fairness
python fairness/analyzers/sensor_fairness_analyzer.py

# Analyze cohort fairness
python fairness/analyzers/cohort_fairness_analyzer.py
```

### Run Comprehensive Analysis (Recommended)

```bash
# Analyze ALL features at once with embedded summary table
python fairness/analyzers/legendary_distillation_analyzer.py
```

### Test All Analyzers

```bash
python fairness/tests/test_all_analyzers.py
```

## Available Analyzers

### 1. Gender Fairness Analyzer
- **File**: `analyzers/gender_fairness_analyzer.py`
- **Groups**: Male, Female
- **Output**: JSON report + 4-panel PNG visualization

### 2. Age Group Fairness Analyzer
- **File**: `analyzers/age_fairness_analyzer.py`
- **Groups**: 20-40, 40-60, 60-80
- **Output**: JSON report + 4-panel PNG visualization

### 3. Pump Model Fairness Analyzer
- **File**: `analyzers/pump_model_fairness_analyzer.py`
- **Groups**: 630G, 530G
- **Output**: JSON report + 4-panel PNG visualization

### 4. Sensor Band Fairness Analyzer
- **File**: `analyzers/sensor_fairness_analyzer.py`
- **Groups**: Empatica, Basis
- **Output**: JSON report + 4-panel PNG visualization

### 5. Cohort Fairness Analyzer
- **File**: `analyzers/cohort_fairness_analyzer.py`
- **Groups**: 2018, 2020
- **Output**: JSON report + 4-panel PNG visualization

### 6. Legendary Distillation Analyzer (Comprehensive)
- **File**: `analyzers/legendary_distillation_analyzer.py`
- **Groups**: ALL (analyzes all 5 features simultaneously)
- **Output**: 
  - JSON report with comprehensive data
  - PNG visualization with embedded summary table
  - CSV file with per-group performance metrics

## Output Files

All results are saved in `fairness/analysis_results/`

### Individual Analyzers Generate:
1. `{feature}_fairness_report_{timestamp}.json` - Structured data report
2. `{feature}_fairness_analysis_{timestamp}.png` - 4-panel visualization

### Legendary Analyzer Generates:
1. `legendary_distillation_report_{timestamp}.json` - Comprehensive JSON report
2. `legendary_distillation_analysis_{timestamp}.png` - Multi-panel visualization with embedded table
3. `legendary_summary_table_{timestamp}.csv` - Per-group summary table

### JSON Report Structure

```json
{
  "report_type": "Gender Fairness Analysis - Distillation Impact",
  "generated": "20251027 150000",
  "analysis_type": "gender",
  "groups": {
    "Female": {
      "patient_count": 5,
      "phases": {
        "teacher": {"rmse_mean": 21.7002, "mae_mean": 13.478},
        "student_baseline": {"rmse_mean": 21.7894, "mae_mean": 13.5207},
        "distilled": {"rmse_mean": 21.7167, "mae_mean": 13.4534}
      }
    }
  },
  "distillation_impact": {
    "teacher_fairness_ratio": 1.0725,
    "distilled_fairness_ratio": 1.0718,
    "change": -0.0007,
    "conclusion": "DISTILLATION MAINTAINS OR IMPROVES FAIRNESS"
  },
  "per_group_impact": {
    "Female": {
      "teacher_rmse": 21.7002,
      "distilled_rmse": 21.7167,
      "rmse_change": 0.0164,
      "percent_change": 0.08,
      "status": "Slightly Worse"
    }
  }
}
```

## Fairness Metrics

### Fairness Ratio
The ratio between the worst-performing group and best-performing group:
```
Fairness Ratio = max(group_rmse) / min(group_rmse)
```

### Fairness Levels
- **1.00 - 1.10x**: EXCELLENT fairness
- **1.10 - 1.25x**: GOOD fairness  
- **1.25 - 1.50x**: ACCEPTABLE fairness
- **1.50+**: POOR fairness (needs attention)

### Per-Group Status
- **IMPROVED**: RMSE decreased by > 0.5
- **Slightly Improved**: RMSE decreased by < 0.5
- **Slightly Worse**: RMSE increased by < 0.5
- **WORSE**: RMSE increased by > 0.5

## Visualization Panels

Each individual analyzer generates a 4-panel visualization:

1. **Top Left**: RMSE Comparison (Teacher vs Student vs Distilled)
2. **Top Right**: Fairness Progression across phases
3. **Bottom Left**: Performance Ratios by phase
4. **Bottom Right**: Summary with key metrics

The Legendary analyzer adds:
- **Top**: Overall fairness change by feature (bar chart)
- **Middle**: Per-group impacts for each feature (5 charts)
- **Bottom**: Embedded summary table with all metrics
- **Legend**: Color coding explanation

## Integration with Training

### Use Fairness Metrics in Training

```python
from fairness.metrics.fairness_metrics import FairnessMetrics

# Calculate fairness during training
fairness_calculator = FairnessMetrics()
fairness_ratio = fairness_calculator.demographic_parity(predictions, demographics)
```

### Apply Fairness-Aware Loss

```python
from fairness.loss_functions.fairness_losses import FairnessAwareLoss

# Add fairness penalty to training loss
fairness_loss = FairnessAwareLoss(demographic_groups=groups)
total_loss = prediction_loss + 0.1 * fairness_loss(predictions, demographics)
```

## Project Structure

```
fairness/
├── README.md                          # This file
├── analyzers/                         # Fairness analyzers
│   ├── base_analyzer.py              # Base class for all analyzers
│   ├── gender_fairness_analyzer.py   # Gender analysis
│   ├── age_fairness_analyzer.py      # Age group analysis
│   ├── pump_model_fairness_analyzer.py
│   ├── sensor_fairness_analyzer.py
│   ├── cohort_fairness_analyzer.py
│   └── legendary_distillation_analyzer.py  # Comprehensive
├── utils/                             # Utility functions
│   └── analyzer_utils.py             # Common analysis utilities
├── metrics/                           # Fairness metrics
│   └── fairness_metrics.py           # Metric calculations
├── loss_functions/                    # Fairness-aware losses
│   └── fairness_losses.py            # Loss implementations
├── visualization/                     # Plotting utilities
│   └── fairness_plots.py             # Chart generation
├── analysis/                          # Patient-level analysis
│   └── patient_analyzer.py           # Individual patient analysis
├── tests/                             # Tests and examples
│   ├── test_all_analyzers.py         # Test suite
│   └── example_analyzer.py           # Example implementation
└── analysis_results/                  # Output directory
    ├── *_report_*.json               # JSON reports
    ├── *_analysis_*.png              # Visualizations
    └── *_summary_*.csv               # CSV tables
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn
```

## Advanced Usage

### Programmatic Access

```python
from fairness.analyzers.gender_fairness_analyzer import GenderFairnessAnalyzer

# Initialize analyzer
analyzer = GenderFairnessAnalyzer()

# Run analysis
analyzer.analyze_latest()

# Access results programmatically
# Results are saved as JSON files in analysis_results/
```

### Custom Fairness Thresholds

Edit the thresholds in `base_analyzer.py`:

```python
def interpret_fairness_ratio(self, ratio: float) -> str:
    if ratio < 1.10:
        return "EXCELLENT"
    elif ratio < 1.25:
        return "GOOD"
    elif ratio < 1.50:
        return "ACCEPTABLE"
    else:
        return "POOR"
```

## Tips for Interpretation

### Good Signs
- Fairness ratio < 1.25 (GOOD or EXCELLENT)
- Distillation maintains or improves fairness (negative change)
- Most groups show "Slightly Improved" or stable status

### Warning Signs
- Fairness ratio > 1.50 (POOR)
- Large positive fairness change (> 0.10)
- Multiple groups showing "WORSE" status

### Taking Action
1. **If fairness worsens**: Consider using fairness-aware training losses
2. **If specific groups worsen**: Examine data representation for those groups
3. **If overall fairness is poor**: May need to retrain with balanced sampling

## Troubleshooting

### No Data Found
- Ensure experiment results exist in `distillation_experiments/`
- Check that patient results contain all three phases (teacher, student_baseline, distilled)

### Import Errors
- Make sure you're running from the project root
- Verify all dependencies are installed

### Visualization Issues
- If charts look compressed, adjust figure size in the analyzer
- For overlapping text, increase spacing parameters

## Recent Updates

### October 27, 2025
- ✅ Removed all emojis for clean, professional output
- ✅ Converted all reports from TXT to JSON format
- ✅ Added embedded summary tables in Legendary visualizations
- ✅ Fixed title overlapping in multi-panel charts
- ✅ Added CSV export for spreadsheet analysis
- ✅ Organized tests and examples into dedicated folder

## Support

For issues or questions, please refer to:
- Test suite: `fairness/tests/test_all_analyzers.py`
- Example analyzer: `fairness/tests/example_analyzer.py`
- Base implementation: `fairness/analyzers/base_analyzer.py`

---

**All 6 analyzers tested and operational** ✅
