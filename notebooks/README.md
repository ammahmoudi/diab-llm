# DiabLLM Notebooks

Analysis and demonstration notebooks for the DiabLLM project.

## 📊 Analysis Notebooks

### 1. **efficiency_analysis.ipynb** (formerly clean_llm_analysis.ipynb)
**Purpose**: Comprehensive model efficiency analysis
- Loads inference and training data from logs
- Calculates performance metrics (latency, throughput, memory usage)
- **Generates LaTeX tables** for publication (uses `utils/latex_table_generator.py`)
- **Creates efficiency plots** (CPU/GPU latency, memory, power consumption)
- Saves outputs to `outputs/latex_tables/` and `outputs/plots/`

**Key Functions**:
- `generate_all_tables()` - Creates comprehensive LaTeX tables
- `create_inference_plots()` - Generates all efficiency charts
- `calculate_energy_metrics()` - Power consumption analysis

### 2. **distillation_analysis.ipynb** (formerly distillation_results_analysis.ipynb)
**Purpose**: Knowledge distillation experiments analysis
- Analyzes 60 distillation runs (12 patients × 5 seeds)
- Compares teacher vs student vs distilled model performance
- **Generates distillation LaTeX tables** (custom table generator included)
- **Creates comparison charts** (RMSE, improvement metrics, patient-wise analysis)
- Statistical significance testing

**Key Features**:
- Patient-wise performance analysis
- Improvement rate calculations
- Publication-ready LaTeX tables for distillation results
- Comprehensive visualizations (bar charts, heatmaps, distribution plots)

### 3. **grid_search_analysis.ipynb**
**Purpose**: Hyperparameter tuning visualization
- Parses grid search logs
- Visualizes parameter impact on model performance
- Identifies optimal hyperparameter combinations

### 4. **time_series_visualization.ipynb**
**Purpose**: Blood glucose time series plotting
- Visualizes glucose predictions vs ground truth
- Multi-step forecasting plots
- Patient-specific time series analysis

## 🧪 Demo Notebooks

### 5. **forecasting_chronos.ipynb**
**Purpose**: Chronos model demonstrations (from AutoGluon tutorial)
- Zero-shot forecasting examples
- Fine-tuning Chronos on custom data
- Handling covariates and static features

## 🎨 Where Are Your Chart/Table Generation Codes?

### LaTeX Table Generators:
1. **`utils/latex_table_generator.py`** - Main comprehensive table generator
   - `create_comprehensive_standardized_table()` - Professional publication tables
   - `generate_all_tables()` - Batch table generation
   - Used by `efficiency_analysis.ipynb`

2. **`utils/latex_tools.py`** - Additional LaTeX utilities
   - `generate_latex_table()` - Generic table formatter

3. **`distillation_analysis.ipynb`** - Contains inline distillation table generator
   - `generate_distillation_latex_table()` function (cell #29)

### Chart/Plot Generators:
1. **`utils/clean_plotting.py`** - Clean plotting functions
2. **`utils/analysis_utils.py`** - Analysis and plotting utilities
   - `create_inference_plots()` - Efficiency visualization suite
3. **`utils/training_analysis.py`** - Training metrics visualization
   - Uses matplotlib and seaborn for publication-quality plots
4. **`scripts/analysis/analyze_distillation_results.py`** - Standalone distillation analysis script
   - Can be run from command line for batch analysis

## 📁 Output Structure

```
notebooks/
├── outputs/
│   ├── latex_tables/          # Generated LaTeX tables
│   │   └── comprehensive_standardized_metrics.tex
│   ├── plots/                 # Generated charts
│   └── analysis_results/      # Summary reports
└── AutogluonModels/           # Trained Chronos models
```

## 🚀 Quick Start

**To generate tables and charts for your paper:**
```python
# Run efficiency_analysis.ipynb
# It will automatically:
# 1. Load all experimental data
# 2. Generate LaTeX tables → outputs/latex_tables/
# 3. Create all efficiency plots → outputs/plots/
```

**For distillation results:**
```python
# Run distillation_analysis.ipynb
# Generates distillation LaTeX tables and comparison charts
```

All notebooks use utility modules from `../utils/` and are fully functional!
