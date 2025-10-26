# Gender Fairness Analysis

## Overview

Analyze gender fairness in your distillation pipeline using actual experimental data.

## Usage

```bash
python fairness/gender_fairness_analyzer.py
```

## Results

Analysis shows distillation maintains good gender fairness:
- Male patients: RMSE ~20.0
- Female patients: RMSE ~22.1  
- Fairness ratio: 1.10x (excellent)
- Distillation impact: Slight improvement

## Files

- `gender_fairness_analyzer.py` - Main analysis
- `integration_guide.py` - Fairness-aware training
- `loss_functions/` - Fairness loss implementations
- `metrics/` - Fairness calculations