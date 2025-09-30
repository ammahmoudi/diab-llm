# Efficiency Benchmarking System

## ✅ Current Working System

This directory contains the **real-time performance profiling system** that provides comprehensive efficiency metrics for LLM time series models, fully addressing reviewer requirements.

## 📁 Active Files

- ✅ **`real_time_profiler.py`** - Main profiling system with NVIDIA ML integration
- ✅ **`efficiency_calculator.py`** - Basic efficiency calculations
- ✅ **`combine_reports.py`** - Combines multiple performance reports into comprehensive analysis

## 🚀 How to Use

The system is **automatically integrated** into `main.py`. Simply run your model training/inference:

```bash
python main.py --config_path your_config.gin --log_level INFO
```

This automatically generates:
1. **Individual performance reports** (training, inference) 
2. **Comprehensive combined report** with all metrics
3. **Real-time monitoring** of CPU, GPU, memory, power, temperature

## 📊 Generated Metrics

**Reviewer Requirements Fully Met:**
- ✅ **CPU/GPU/Edge Latency**: Real measured timing during inference
- ✅ **Model Size on Disk**: Actual file sizes
- ✅ **RAM/VRAM Usage**: Process-specific + system-wide memory tracking
- ✅ **Throughput**: Calculated from real measurements
- ✅ **Edge Feasibility**: Quantitative device compatibility analysis

**Enhanced Monitoring:**
- GPU utilization, temperature, power consumption (NVIDIA ML)
- Process-specific vs system-wide memory usage
- Statistical analysis (P95/P99 latencies)
- Edge device compatibility assessment

## � Report Locations

Reports are automatically saved in your experiment log directory:
```
experiment_configs_*/logs/logs_*/
├── real_performance_report_*_training_*.json
├── real_performance_report_*_inference_*.json
└── comprehensive_performance_report_*_comprehensive_*.json
```

## 🔧 System Requirements

- Python packages: `psutil`, `torch`, `numpy`, `nvidia-ml-py`
- NVIDIA GPU with CUDA (for GPU monitoring)
- Linux/Windows compatible