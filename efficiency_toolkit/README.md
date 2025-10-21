# LLM-TIME Efficiency Analysis Toolkit

A comprehensive toolkit for analyzing the efficiency of Time-LLM and Chronos models, including training efficiency, inference performance, memory usage analysis, and knowledge distillation evaluation.

## 🚀 Quick Start

### From Project Root
```bash
# Run all efficiency experiments and analyze results
python efficiency_runner.py

# Analyze existing results only
python efficiency_runner.py --analyze-only

# Run experiments with analysis
python efficiency_runner.py --analyze
```

### Direct Usage
```bash
cd efficiency_toolkit/core
python comprehensive_efficiency_runner.py --help
```

## 📁 Toolkit Structure

```
efficiency_toolkit/
├── __init__.py                 # Main toolkit interface
├── core/                       # Core efficiency tools
│   ├── comprehensive_efficiency_runner.py  # Main runner (unified tool)
│   ├── efficiency_calculator.py           # Efficiency calculations
│   ├── real_time_profiler.py             # Real-time performance profiling
│   └── efficiency_reporting.py           # Report generation utilities
├── analysis/                   # Analysis tools and legacy scripts
│   ├── extract_efficiency_results.py     # Legacy efficiency extraction
│   ├── extract_all_metrics.py           # Comprehensive metrics extraction
│   ├── combine_reports.py               # Report combination utilities
│   ├── experiment_efficiency_analysis.ipynb    # Jupyter analysis notebook
│   └── comprehensive_efficiency_analysis.ipynb # Comprehensive notebook
├── scripts/                    # Automation scripts
│   └── run_efficiency_tests.sh          # Shell script for efficiency tests
└── results/                    # Generated analysis results
    └── efficiency_analysis_results/      # Timestamped analysis outputs
```

## 🛠️ Core Components

### ComprehensiveEfficiencyRunner
The main unified tool that consolidates all efficiency analysis functionality:

- **Experiment Execution**: Run efficiency experiments across all model types
- **Data Analysis**: Comprehensive analysis with pandas/numpy
- **Results Organization**: Structured folder hierarchy with timestamped runs
- **Report Generation**: Markdown reports with insights and rankings

### Key Features
- ✅ **Unified Analysis**: All efficiency tools consolidated into one system
- ✅ **Structured Output**: Organized results in timestamped folders
- ✅ **Multi-Model Support**: Time-LLM, Chronos, and Distillation experiments
- ✅ **Rich Reporting**: Markdown reports with performance rankings
- ✅ **Memory Profiling**: Real-time memory and GPU usage tracking
- ✅ **Edge Analysis**: Edge deployment feasibility assessment

## 📊 Analysis Results Structure

Each analysis run creates a timestamped folder with:

```
results/efficiency_analysis_results/analysis_YYYYMMDD_HHMMSS/
├── raw_data/           # Raw experiment data (CSV)
├── summaries/          # Model type summaries
├── rankings/           # Efficiency rankings
├── reports/            # Markdown reports
├── comparisons/        # Training vs inference comparisons
└── analysis_index.txt  # Summary of all generated files
```

## 🔧 Usage Examples

### Python API
```python
from efficiency_toolkit.core.comprehensive_efficiency_runner import ComprehensiveEfficiencyRunner

# Initialize runner
runner = ComprehensiveEfficiencyRunner()

# Run all experiments
runner.run_all_experiments()

# Analyze results
analysis_results = runner.analyze_all_experiments(save_results=True)

# Access specific model results
distillation_results = analysis_results['by_model_type']['distillation']
```

### Command Line Options
```bash
# Show help
python efficiency_runner.py --help

# Run with specific test patient
python efficiency_runner.py --patient 570

# Custom output directory
python efficiency_runner.py --output-dir /custom/path

# Dry run to see what would execute
python efficiency_runner.py --dry-run

# Verbose output
python efficiency_runner.py --verbose
```

## 📈 Analysis Capabilities

### Model Performance Comparison
- Training efficiency rankings
- Inference speed comparisons
- Memory usage analysis
- GPU utilization tracking

### Edge Deployment Analysis
- Edge device feasibility assessment
- Resource constraint evaluation
- Deployment recommendations

### Knowledge Distillation Analysis
- Teacher vs Student vs Distilled model comparison
- Distillation efficiency evaluation
- Performance trade-off analysis

## 🧪 Experiment Types Supported

1. **Time-LLM Experiments**
   - Training mode efficiency
   - Inference mode performance
   - Cross-scenario evaluation

2. **Chronos Experiments**
   - T5-base and T5-tiny models
   - Time series forecasting efficiency

3. **Knowledge Distillation**
   - Teacher model (BERT)
   - Student model (TinyBERT)
   - Distilled model inference

## 🚀 Getting Started

1. **Setup Environment**
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run Quick Analysis**
   ```bash
   python efficiency_runner.py --analyze-only
   ```

3. **View Results**
   ```bash
   # Check the latest analysis folder
   ls -la efficiency_toolkit/results/efficiency_analysis_results/
   ```

## 📝 Generated Reports

The toolkit generates comprehensive markdown reports including:

- **Executive Summary**: Total experiments and breakdown
- **Performance Analysis**: Model-specific summaries and rankings
- **Key Insights**: Performance rankings and recommendations
- **Efficiency Rankings**: Training, memory, and inference efficiency scores

## 🔄 Migration Notes

This toolkit consolidates and replaces several scattered scripts:
- ❌ `extract_efficiency_results.py` (moved to analysis/)
- ❌ `comprehensive_results_summary.py` (empty, removed)
- ✅ `comprehensive_efficiency_runner.py` (main unified tool)

## 🤝 Contributing

When adding new efficiency analysis features:
1. Add core functionality to `core/`
2. Add analysis scripts to `analysis/`
3. Update the main `ComprehensiveEfficiencyRunner` class
4. Add tests and documentation

## 📄 License

Part of the LLM-TIME project. See main project LICENSE for details.