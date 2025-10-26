# Fairness Analysis Framework

Comprehensive fairness analysis across multiple demographic dimensions for distillation models.

## Quick Start

### 1. Run All Fairness Analyzers
```bash
cd /workspace/LLM-TIME/fairness
python test_all_analyzers.py
```

### 2. Run Individual Analyzers
```bash
# Gender fairness
python gender_fairness_analyzer.py

# Age group fairness
python age_fairness_analyzer.py

# Pump model fairness
python pump_model_fairness_analyzer.py

# Sensor band fairness
python sensor_fairness_analyzer.py

# Study cohort fairness
python cohort_fairness_analyzer.py

# Comprehensive all-feature analysis
python legendary_fairness_analyzer.py
```

### 3. Interactive Analysis (Recommended)
```bash
cd /workspace/LLM-TIME/fairness/notebooks
jupyter notebook Gender_Fairness_Analysis.ipynb
```

## Main Analyzers Guide

### 📊 Individual Demographic Analyzers

#### 1. `gender_fairness_analyzer.py` - Gender Fairness
Analyzes fairness between male (7 patients) and female (5 patients) groups.
- **Visualization**: 2x2 dashboard with RMSE comparison, fairness progression, performance ratios, and summary
- **Output**: Detailed report with fairness assessment and distillation impact analysis
- **Current Result**: 1.20x ratio (GOOD fairness)

#### 2. `age_fairness_analyzer.py` - Age Group Fairness
Analyzes fairness across age groups: 20-40, 40-60, 60-80 years.
- **Visualization**: Age-specific performance comparison with confidence intervals
- **Output**: Multi-group fairness ratios and recommendations
- **Current Result**: 1.25x ratio (GOOD fairness)

#### 3. `pump_model_fairness_analyzer.py` - Pump Model Fairness
Analyzes fairness between 630G and 530G insulin pump models.
- **Visualization**: Device-specific performance metrics
- **Output**: Technology bias assessment
- **Current Result**: 1.11x ratio (EXCELLENT fairness)

#### 4. `sensor_fairness_analyzer.py` - Sensor Band Fairness
Analyzes fairness between Empatica and Basis sensor bands.
- **Visualization**: Sensor-specific accuracy comparison
- **Output**: Hardware bias detection
- **Current Result**: 1.09x ratio (EXCELLENT fairness)

#### 5. `cohort_fairness_analyzer.py` - Study Cohort Fairness
Analyzes fairness between 2018 and 2020 study cohorts.
- **Visualization**: Temporal consistency analysis
- **Output**: Data collection period impact assessment
- **Current Result**: 1.09x ratio (EXCELLENT fairness)

#### 6. `legendary_fairness_analyzer.py` - Comprehensive Analysis
Analyzes fairness across ALL demographic features simultaneously.
- **Visualization**: Multi-dimensional fairness overview with rankings
- **Output**: Complete fairness profile identifying most/least fair features
- **Results**: 4/5 features GOOD or better, Age showing highest disparity

## Framework Structure

```
fairness/
├── 📊 Individual Analyzers (All with consistent 2x2 dashboard format)
│   ├── gender_fairness_analyzer.py      # Gender fairness analysis
│   ├── age_fairness_analyzer.py         # Age group fairness
│   ├── pump_model_fairness_analyzer.py  # Insulin pump fairness
│   ├── sensor_fairness_analyzer.py      # Sensor band fairness
│   ├── cohort_fairness_analyzer.py      # Study cohort fairness
│   └── legendary_fairness_analyzer.py   # All-feature comprehensive
│
├── 🧪 Testing & Validation
│   └── test_all_analyzers.py            # Automated test suite
│
├── � Interactive Notebooks
│   └── notebooks/
│       └── Gender_Fairness_Analysis.ipynb
│
├── 🔧 Core Components
│   ├── analysis/patient_analyzer.py     # Patient demographics
│   ├── metrics/fairness_metrics.py      # Fairness calculations  
│   ├── loss_functions/fairness_losses.py # 4 fairness-aware loss types
│   └── visualization/fairness_plots.py  # Plotting functions
│
├── 📁 Results & Reports (auto-generated)
│   └── analysis_results/
│       ├── *_fairness_analysis_*.png    # Visualizations
│       ├── *_fairness_report_*.txt      # Text reports
│       └── *_fairness_analysis.json     # JSON results
│
└── 📚 Documentation
    ├── README.md                         # This file
    ├── FAIRNESS_THRESHOLDS_EXPLAINED.md
    ├── GENDER_FAIRNESS_RESULTS.md
    └── integration_guide.py
```

## What You Get

### Comprehensive Fairness Analysis Across Multiple Dimensions
All analyzers provide:
- **Consistent 2x2 Dashboard Format**: RMSE comparison, fairness progression, performance ratios, experiment summary
- **Detailed Text Reports**: Timestamped reports with fairness assessments and recommendations
- **JSON Results**: Machine-readable results for integration with other tools
- **Automated Fairness Assessment**: EXCELLENT/GOOD/ACCEPTABLE/POOR ratings

### Current Results Summary (Latest Experiment)

| Demographic Feature | Groups | Fairness Ratio | Level | Status |
|---------------------|--------|----------------|-------|--------|
| **Gender** | Male vs Female | 1.20x | GOOD | ✅ |
| **Pump Model** | 630G vs 530G | 1.11x | EXCELLENT | ✅ |
| **Sensor Band** | Empatica vs Basis | 1.09x | EXCELLENT | ✅ |
| **Cohort** | 2018 vs 2020 | 1.09x | EXCELLENT | ✅ |
| **Age Group** | 20-40, 40-60, 60-80 | 1.25x | GOOD | ✅ |

**Overall Assessment**: The model demonstrates good to excellent fairness across all demographic dimensions, with age showing the highest (but still acceptable) disparity.

### Fairness Thresholds
- **< 1.10x**: 🟢 EXCELLENT fairness
- **1.10x - 1.25x**: 🟡 GOOD fairness  
- **1.25x - 1.50x**: 🟠 ACCEPTABLE fairness
- **> 1.50x**: 🔴 POOR fairness - intervention needed

### Generated Files (auto-timestamped)
- Fairness analysis plots (PNG format, 300 DPI)
- Comprehensive text reports with recommendations
- JSON results for programmatic access
- All saved to `analysis_results/` directory