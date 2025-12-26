# Notebook Organization

This directory contains Jupyter notebooks for analysis and demonstration of the DiabLLM project.

## Active Notebooks (5 total)

### 1. Efficiency Analysis
**clean_llm_analysis.ipynb** - Main efficiency analysis notebook
- Comprehensive efficiency metrics (inference, training, distillation)
- Uses modules from `utils/` directory
- Generates plots, tables, and reports
- Status: ✅ Active and tested

### 2. Model Demonstrations
**forecasting_chronos.ipynb** - Chronos model forecasting examples
- Complete Chronos usage guide
- Data loading and preprocessing examples
- Forecasting demonstrations
- Status: ✅ Active

**distillation_results_analysis.ipynb** - Knowledge distillation analysis
- Teacher-student performance comparison
- Distillation effectiveness metrics
- Comprehensive visualization
- Status: ✅ Active

### 3. Utility Notebooks
**analyze_grid_logs.ipynb** - Hyperparameter grid search log analysis
- Simple log parsing and visualization
- Status: ✅ Active

**step_trend.ipynb** - Time series step trend visualization
- Simple plotting utility for time series data
- Status: ✅ Active

## Supporting Modules

All analysis modules are in the main `utils/` directory:
- `utils/enhanced_data_loader.py` - Data loading for Time-LLM and Chronos
- `utils/analysis_utils.py` - Analysis and visualization functions
- `utils/training_analysis.py` - Training and distillation analysis
- `utils/edge_analysis.py` - Edge deployment assessment
- `utils/latex_table_generator.py` - LaTeX table generation for papers

## Output Structure

Notebooks save outputs to:
```
outputs/
├── plots/          # Visualization files (.png)
├── data/           # Processed data (.csv)
└── reports/        # Analysis reports (.md)
```

## Usage

To run the main efficiency analysis:
```bash
jupyter notebook clean_llm_analysis.ipynb
```

All notebooks use the project's virtual environment and modules from `utils/`.
