# Fairness Analysis Framework

Comprehensive fairness analysis for distillation and inference scenarios in diabetes prediction models.

## Overview

This framework analyzes fairness across demographic dimensions in two contexts:

1. **Distillation Analysis**: Teacher → Student → Distilled models during knowledge distillation
2. **Inference Scenarios**: Different training conditions (inference-only, standard, noisy, denoised)

## Quick Start

**Note**: Run all commands from the project root directory (`/home/amma/LLM-TIME/`)

### Distillation Analysis

```bash
# Per-patient experiments
python fairness/run_distillation_analyzers.py --experiment-type per_patient

# All-patients experiments
python fairness/run_distillation_analyzers.py --experiment-type all_patients
```

### Inference Scenarios Analysis

```bash
# All features
python fairness/run_inference_analyzers.py --feature all

# Specific feature
python fairness/run_inference_analyzers.py --feature gender
```

## Analyzers

### Distillation Analyzers (6)

Analyze fairness during knowledge distillation across 5 demographic features plus comprehensive analysis.

- **Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)
- **Phases Analyzed**: Teacher → Student Baseline → Distilled
- **Results Location**:
  - fairness/analysis_results/distillation_per_patient/
  - fairness/analysis_results/distillation_all_patients/

### Inference Scenario Analyzers (6)

Analyze fairness across different training/inference conditions.

- **Features**: Gender, Age, Pump Model, Sensor Band, Cohort, Legendary (all features)
- **Scenarios Analyzed**:
  1. Inference Only (no training)
  2. Trained on Standard Data
  3. Trained on Noisy Data
  4. Trained on Denoised Data
- **Results Location**: fairness/analysis_results/inference_scenarios/

## Output Files

Each analyzer generates:

- **JSON Report**: Structured fairness metrics and statistics
- **PNG Visualization**: Multi-panel charts with RMSE comparisons and fairness analysis
- **CSV Table** (Legendary only): Per-group performance summary across all scenarios

### Legendary Analyzer Visualization

Single comprehensive PNG file containing:

- **Top Panel**: Fairness heatmap across all features and scenarios/phases
- **Middle Panels**: RMSE comparison charts for each demographic feature
- **Bottom Panel**: Summary table with per-group RMSE values and best/worst scenarios

## Demographics

- **Gender**: Male, Female
- **Age Group**: 20-40, 40-60, 60-80 years
- **Pump Model**: 630G, 530G
- **Sensor Band**: Empatica, Basis
- **Cohort**: 2018, 2020

## Fairness Metrics

### Fairness Ratio

Fairness Ratio = max(group_rmse) / min(group_rmse)

Lower ratios indicate more fair performance across groups.

### Fairness Levels

- **1.00-1.10**: EXCELLENT
- **1.10-1.25**: GOOD
- **1.25-1.50**: ACCEPTABLE
- **1.50+**: POOR (requires attention)

## Project Structure

```
fairness/
├── README.md
├── run_distillation_analyzers.py
├── run_inference_analyzers.py
├── analyzers/
│   ├── base_analyzer.py
│   ├── base_inference_analyzer.py
│   ├── *_fairness_analyzer.py        (5 distillation analyzers)
│   ├── inference_*_analyzer.py       (5 inference analyzers)
│   ├── legendary_distillation_analyzer.py
│   └── inference_legendary_analyzer.py
└── analysis_results/
    ├── distillation_per_patient/
    ├── distillation_all_patients/
    └── inference_scenarios/
```

## Requirements

Install from main project requirements:

```bash
pip install -r requirements.txt
```

Key dependencies: pandas, numpy, matplotlib, seaborn

## Advanced Usage

### Run Individual Analyzer

```bash
# Distillation
python fairness/analyzers/gender_fairness_analyzer.py --experiment-type per_patient

# Inference
python fairness/analyzers/inference_gender_analyzer.py
```

### Programmatic Access

```python
from fairness.analyzers.legendary_distillation_analyzer import LegendaryDistillationAnalyzer

analyzer = LegendaryDistillationAnalyzer(experiment_type='per_patient')
analyzer.analyze()
```

## Troubleshooting

- **No data found**: Ensure experiment results exist in distillation_experiments/ directory
- **Import errors**: Run commands from project root and verify all dependencies installed
- **Visualization issues**: Adjust figure size parameters in analyzer code if needed

---

All 12 analyzers tested and operational
