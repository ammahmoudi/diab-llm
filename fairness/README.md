# Fairness Analysis Framework# Fairness Analysis Framework# Fairness Analysis Framework



Comprehensive fairness analysis for distillation and inference scenarios in diabetes prediction models.



## Overview**Comprehensive fairness analysis for distillation and inference scenarios in diabetes prediction models****Comprehensive fairness analysis for knowledge distillation in diabetes prediction models**



This framework analyzes fairness across demographic dimensions in two contexts:



1. **Distillation Analysis**: Teacher → Student → Distilled models during knowledge distillation## Overview## Overview

2. **Inference Scenarios**: Different training conditions (inference-only, standard, noisy, denoised)



## Quick Start

This framework analyzes fairness across demographic dimensions in two contexts:This framework provides tools to analyze fairness across multiple demographic dimensions in your distilled models. It compares Teacher (large model), Student Baseline (small model without distillation), and Distilled (small model with knowledge distillation) to understand how distillation affects fairness.

### Distillation Analysis



```bash

# Per-patient experiments1. **Distillation Analysis**: Teacher → Student → Distilled models during knowledge distillation## Features

python fairness/run_distillation_analyzers.py --experiment-type per_patient

2. **Inference Scenarios**: Different training conditions (inference-only, standard, noisy, denoised)

# All-patients experiments

python fairness/run_distillation_analyzers.py --experiment-type all_patients- ✅ **6 Demographic Analyzers**: Gender, Age Group, Pump Model, Sensor Band, Study Cohort, and Comprehensive (Legendary)

```

## Quick Start- ✅ **Multi-Phase Analysis**: Compare Teacher vs Student vs Distilled models

### Inference Scenarios Analysis

- ✅ **Per-Group Impact**: See which specific groups improved or worsened

```bash

# All features### Distillation Analysis- ✅ **JSON Reports**: Structured, machine-readable output

python fairness/run_inference_analyzers.py --feature all

- ✅ **Visual Analysis**: 4-panel comprehensive charts

# Specific feature  

python fairness/run_inference_analyzers.py --feature gender```bash- ✅ **Embedded Tables**: Summary tables in visualizations (Legendary)

```

# Per-patient experiments- ✅ **CSV Export**: Spreadsheet-ready per-group summaries

## Analyzers

python fairness/run_distillation_analyzers.py --experiment-type per_patient

### Distillation Analyzers (6)

## Experiment Types

Analyze fairness during knowledge distillation across 5 demographic features plus comprehensive analysis.

# All-patients experiments

**Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)

python fairness/run_distillation_analyzers.py --experiment-type all_patientsThe analyzers support two types of experiments:

**Phases Analyzed**: Teacher → Student Baseline → Distilled

```

**Results Location**: 

- `fairness/analysis_results/distillation_per_patient/`### 1. Per-Patient Experiments (default)

- `fairness/analysis_results/distillation_all_patients/`

### Inference Scenarios Analysis- **Structure**: Each patient has separate training, student, and distillation phases

### Inference Scenario Analyzers (6)

- **Location**: `distillation_experiments/pipeline_*/`

Analyze fairness across different training/inference conditions.

```bash- **Phases analyzed**: Teacher, Student Baseline, Distilled

**Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)

# All features- **Usage**: Default mode

**Scenarios Analyzed**:

1. Inference Only (no training)python fairness/run_inference_analyzers.py --feature all

2. Trained on Standard Data

3. Trained on Noisy Data### 2. All-Patients Experiments

4. Trained on Denoised Data

# Specific feature- **Structure**: Single model trained on all patients, with per-patient inference

**Results Location**: `fairness/analysis_results/inference_scenarios/`

python fairness/run_inference_analyzers.py --feature gender- **Location**: `distillation_experiments/all_patients_pipeline/`

## Output Files

```- **Phases analyzed**: Teacher, Student Baseline, Distilled

Each analyzer generates:

- **JSON Report**: Structured fairness metrics and statistics- **Usage**: Add `--experiment-type all_patients` flag

- **PNG Visualization**: Multi-panel charts with RMSE comparisons and fairness analysis

- **CSV Table** (Legendary only): Per-group performance summary across all scenarios## Analyzers



### Legendary Analyzer Visualization## Quick Start



Single comprehensive PNG file containing:### Distillation Analyzers (6)

- **Top Panel**: Fairness heatmap across all features and scenarios/phases

- **Middle Panels**: RMSE comparison charts for each demographic feature### Run Individual Analyzer

- **Bottom Panel**: Summary table with per-group RMSE values and best/worst scenarios

Analyze fairness during knowledge distillation:

## Demographics

- Gender, Age, Pump Model, Sensor Band, Cohort```bash

- **Gender**: Male, Female

- **Age Group**: 20-40, 40-60, 60-80 years- Legendary (comprehensive analysis)# Per-patient experiments (default)

- **Pump Model**: 630G, 530G

- **Sensor Band**: Empatica, Basispython fairness/analyzers/gender_fairness_analyzer.py

- **Cohort**: 2018, 2020

**Phases**: Teacher → Student Baseline → Distilledpython fairness/analyzers/age_fairness_analyzer.py

## Fairness Metrics



### Fairness Ratio

**Results**: `fairness/analysis_results/distillation_{per_patient|all_patients}/`# All-patients experiments

```

Fairness Ratio = max(group_rmse) / min(group_rmse)python fairness/analyzers/gender_fairness_analyzer.py --experiment-type all_patients

```

### Inference Scenario Analyzers (6)python fairness/analyzers/age_fairness_analyzer.py --experiment-type all_patients

Lower ratios indicate more fair performance across groups.

```

### Fairness Levels

Analyze fairness across training conditions:

- **1.00-1.10**: EXCELLENT

- **1.10-1.25**: GOOD- Gender, Age, Pump Model, Sensor Band, Cohort### Run Comprehensive Analysis (Recommended)

- **1.25-1.50**: ACCEPTABLE

- **1.50+**: POOR (requires attention)- Legendary (comprehensive analysis)



## Project Structure```bash



```**Scenarios**:# Per-patient mode (default)

fairness/

├── README.md1. Inference Only (no training)python fairness/analyzers/legendary_distillation_analyzer.py

├── run_distillation_analyzers.py     # Run distillation analyzers

├── run_inference_analyzers.py        # Run inference analyzers2. Trained on Standard Data

├── analyzers/

│   ├── base_analyzer.py              # Base class for distillation3. Trained on Noisy Data# All-patients mode

│   ├── base_inference_analyzer.py    # Base class for inference

│   ├── *_fairness_analyzer.py        # 5 distillation analyzers4. Trained on Denoised Datapython fairness/analyzers/legendary_distillation_analyzer.py --experiment-type all_patients

│   ├── inference_*_analyzer.py       # 5 inference analyzers

│   ├── legendary_distillation_analyzer.py```

│   └── inference_legendary_analyzer.py

└── analysis_results/**Results**: `fairness/analysis_results/inference_scenarios/`

    ├── distillation_per_patient/

    ├── distillation_all_patients/### Run All Analyzers

    └── inference_scenarios/

```## Output Files



## Requirements```bash



Install from main project requirements:Each analyzer generates:# Per-patient mode (default)



```bash- **JSON Report**: Structured fairness metricspython fairness/run_all_analyzers.py

pip install -r requirements.txt

```- **PNG Visualization**: Charts with RMSE comparisons and fairness ratios



Key dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`- **CSV Table** (Legendary only): Per-group performance summary# All-patients mode



## Advanced Usagepython fairness/run_all_analyzers.py --experiment-type all_patients



### Run Individual Analyzer### Legendary Analyzer Output```



```bash

# Distillation

python fairness/analyzers/gender_fairness_analyzer.py --experiment-type per_patientSingle comprehensive visualization containing:## Available Analyzers



# Inference- Top: Fairness heatmap across all features and scenarios

python fairness/analyzers/inference_gender_analyzer.py

```- Middle: RMSE comparison charts for each feature### 1. Gender Fairness Analyzer



### Programmatic Access- Bottom: Summary table with per-group RMSE values- **File**: `analyzers/gender_fairness_analyzer.py`



```python- **Groups**: Male, Female

from fairness.analyzers.legendary_distillation_analyzer import LegendaryDistillationAnalyzer

## Demographics- **Output**: JSON report + 4-panel PNG visualization

analyzer = LegendaryDistillationAnalyzer(experiment_type='per_patient')

analyzer.analyze()

```

- **Gender**: Male, Female### 2. Age Group Fairness Analyzer

## Troubleshooting

- **Age Group**: 20-40, 40-60, 60-80- **File**: `analyzers/age_fairness_analyzer.py`

- **No data found**: Ensure experiment results exist in `distillation_experiments/` directory

- **Import errors**: Run commands from project root and verify all dependencies installed- **Pump Model**: 630G, 530G- **Groups**: 20-40, 40-60, 60-80

- **Visualization issues**: Adjust figure size parameters in analyzer code if needed

- **Sensor Band**: Empatica, Basis- **Output**: JSON report + 4-panel PNG visualization

---

- **Cohort**: 2018, 2020

**All 12 analyzers tested and operational** ✅

### 3. Pump Model Fairness Analyzer

## Fairness Metrics- **File**: `analyzers/pump_model_fairness_analyzer.py`

- **Groups**: 630G, 530G

### Fairness Ratio- **Output**: JSON report + 4-panel PNG visualization

```

Fairness Ratio = max(group_rmse) / min(group_rmse)### 4. Sensor Band Fairness Analyzer

```- **File**: `analyzers/sensor_fairness_analyzer.py`

- **Groups**: Empatica, Basis

### Levels- **Output**: JSON report + 4-panel PNG visualization

- **1.00-1.10**: EXCELLENT

- **1.10-1.25**: GOOD  ### 5. Cohort Fairness Analyzer

- **1.25-1.50**: ACCEPTABLE- **File**: `analyzers/cohort_fairness_analyzer.py`

- **1.50+**: POOR- **Groups**: 2018, 2020

- **Output**: JSON report + 4-panel PNG visualization

## Project Structure

### 6. Legendary Distillation Analyzer (Comprehensive)

```- **File**: `analyzers/legendary_distillation_analyzer.py`

fairness/- **Groups**: ALL (analyzes all 5 features simultaneously)

├── README.md- **Output**: 

├── run_distillation_analyzers.py     # Run distillation analyzers  - JSON report with comprehensive data

├── run_inference_analyzers.py        # Run inference analyzers  - PNG visualization with embedded summary table

├── analyzers/  - CSV file with per-group performance metrics

│   ├── base_analyzer.py              # Distillation base class

│   ├── base_inference_analyzer.py    # Inference base class## Output Files

│   ├── *_fairness_analyzer.py        # 5 distillation analyzers

│   ├── inference_*_analyzer.py       # 5 inference analyzersAll results are saved in `fairness/analysis_results/`

│   ├── legendary_distillation_analyzer.py

│   └── inference_legendary_analyzer.pyOutput filenames include the experiment mode (`per_patient` or `all_patients`) to distinguish results.

└── analysis_results/

    ├── distillation_per_patient/### Individual Analyzers Generate

    ├── distillation_all_patients/

    └── inference_scenarios/1. `{feature}_fairness_report_{mode}_{timestamp}.json` - Structured data report

```2. `{feature}_fairness_analysis_{mode}_{timestamp}.png` - 4-panel visualization



## Requirements### Legendary Analyzer Generates



Install from main project requirements:1. `legendary_distillation_report_{mode}_{timestamp}.json` - Comprehensive JSON report

```bash2. `legendary_distillation_analysis_{mode}_{timestamp}.png` - Multi-panel visualization with embedded table

pip install -r requirements.txt3. `legendary_summary_table_{mode}_{timestamp}.csv` - Per-group summary table

```

## Directory Structure

Key dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`

### Per-Patient Experiments

## Advanced Usage

```

### Run Individual Analyzerdistillation_experiments/

  pipeline_YYYY-MM-DD_HH-MM-SS/

```bash    patient_XXX/

# Distillation      phase_1_teacher/

python fairness/analyzers/gender_fairness_analyzer.py --experiment-type per_patient        teacher_training_summary.json

      phase_2_student/

# Inference        student_baseline_summary.json

python fairness/analyzers/inference_gender_analyzer.py      phase_3_distillation/

```        distillation_summary.json

```

### Programmatic Access

### All-Patients Experiments

```python

from fairness.analyzers.legendary_distillation_analyzer import LegendaryDistillationAnalyzer```

distillation_experiments/

analyzer = LegendaryDistillationAnalyzer(experiment_type='per_patient')  all_patients_pipeline/

analyzer.analyze()    pipeline_YYYY-MM-DD_HH-MM-SS/

```      phase_1_teacher/

        per_patient_inference/

## Troubleshooting          time_llm_per_patient_inference_ohiot1dm/

            experiment_results.csv  (teacher results for each patient)

- **No data found**: Check experiment results exist in `distillation_experiments/`      phase_2_student/

- **Import errors**: Run from project root and verify dependencies        per_patient_inference/

- **Visualization issues**: Adjust figure size in analyzer code          time_llm_per_patient_inference_ohiot1dm/

            experiment_results.csv  (student results for each patient)

---      phase_3_distillation/

        per_patient_inference/

**All 12 analyzers tested and operational** ✅          time_llm_per_patient_inference_ohiot1dm/

            experiment_results.csv  (distilled results for each patient)
```

The `experiment_results.csv` files should contain:

```csv
seed,model,dtype,mode,context_length,pred_length,patient_id,log_datetime,rmse,mae,mape
42,BERT-tiny,,,,,540,2025-10-28_17-33-39,27.301605,18.921614,0.13372737
42,BERT-tiny,,,,,544,2025-10-28_17-33-52,21.467813,14.709017,0.09639849
...
```

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
├── run_all_analyzers.py              # Script to run all analyzers
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
├── tests/                             # Examples
│   └── example_analyzer.py           # Example implementation
└── analysis_results/                  # Output directory
    ├── *_report_*.json               # JSON reports
    ├── *_analysis_*.png              # Visualizations
    └── *_summary_*.csv               # CSV tables
```

## Requirements

All required packages are included in the main project requirements file:

```bash
pip install -r requirements.txt
```

Key dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

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

### October 29, 2025
- ✅ Moved `test_all_analyzers.py` to `run_all_analyzers.py` in main fairness directory
- ✅ Consolidated requirements into main project requirements.txt
- ✅ Cleaned up folder structure (tests folder now only contains examples)

### October 27, 2025
- ✅ Removed all emojis for clean, professional output
- ✅ Converted all reports from TXT to JSON format
- ✅ Added embedded summary tables in Legendary visualizations
- ✅ Fixed title overlapping in multi-panel charts
- ✅ Added CSV export for spreadsheet analysis
- ✅ Organized tests and examples into dedicated folder

## Support

For issues or questions, please refer to:
- Run all analyzers: `fairness/run_all_analyzers.py`
- Example analyzer: `fairness/tests/example_analyzer.py`
- Base implementation: `fairness/analyzers/base_analyzer.py`

---

**All 6 analyzers tested and operational** ✅
