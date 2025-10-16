# Time-LLM Knowledge Distillation System

A comprehensive knowledge distillation pipeline for Time-LLM models, enabling efficient model compression while maintaining temporal forecasting performance.

## 🎯 Overview

This system implements a 3-phase knowledge distillation pipeline:
1. **Teacher Training**: Train large, high-capacity models (BERT, DistilBERT)
2. **Student Baseline**: Train smaller student models independently
3. **Knowledge Distillation**: Transfer knowledge from teacher to student models

## 🚀 Quick Start

### Basic Usage
```bash
# Simple distillation with default parameters
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --seed 42

# Custom training configuration
./distill_pipeline.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570,584 \
    --dataset ohiot1dm \
    --seed 42 \
    --teacher-epochs 5 \
    --student-epochs 3 \
    --distill-epochs 3 \
    --lr 0.002 \
    --batch-size 16 \
    --alpha 0.6 \
    --beta 0.4 \
    --kl-weight 0.15 \
    --temperature 4.0

# Dry run (preview commands without execution)
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --dry-run
```

## 📋 Parameters

### Required Parameters
- `--teacher`: Teacher model name (see supported models below)
- `--student`: Student model name (see supported models below)
- `--patients`: Patient IDs (comma-separated or single, e.g., "570" or "570,584")
- `--dataset`: Dataset name ("ohiot1dm" or "d1namo")

### Optional Parameters

#### Training Configuration
- `--seed`: Random seed for reproducibility (default: 238822)
- `--teacher-epochs`: Teacher training epochs (default: 1)
- `--student-epochs`: Student baseline epochs (default: 1)
- `--distill-epochs`: Knowledge distillation epochs (default: 1)

#### Optimization Parameters
- `--lr`: Learning rate for all phases (default: 0.001)
- `--batch-size`: Batch size for all phases (default: 32)

#### Distillation Hyperparameters
- `--alpha`: Weight for ground truth loss (student learning from actual labels, default: 0.5)
- `--beta`: Weight for teacher output loss (student learning from teacher predictions, default: 0.5)
- `--kl-weight`: Weight for KL divergence loss (knowledge transfer strength, default: 0.1)
- `--temperature`: Temperature for knowledge distillation softmax (controls soft target smoothness, default: 3.0)

#### Utility
- `--dry-run`: Preview commands without execution

## 🤖 Supported Models

### Teacher Models (Large, High-Capacity)
- `bert` / `bert-base-uncased` / `bert-base-cased`
- `bert-large-uncased` / `bert-large-cased`
- `distilbert` / `distilbert-base-uncased` / `distilbert-base-cased`
- `tinybert`

### Student Models (Small, Efficient)
- `tinybert`
- `distilbert` / `distilbert-base-uncased` / `distilbert-base-cased`
- `prajjwal1/bert-tiny` (2 layers, 128 dim)
- `prajjwal1/bert-mini` (4 layers, 256 dim)
- `prajjwal1/bert-small` (4 layers, 512 dim)
- `prajjwal1/bert-medium` (8 layers, 512 dim)

## 📁 Output Structure

The pipeline creates organized output directories:
```
distillation_experiments/pipeline_runs/pipeline_YYYY-MM-DD_HH-MM-SS/
├── phase_1_teacher/          # Teacher model training
│   ├── config_teacher_*.gin   # Training configuration
│   ├── bert-base-uncased_570_1epochs/  # Model outputs
│   └── logs/                  # Training logs
├── phase_2_student/           # Student baseline training
│   ├── config_student_*.gin   # Training configuration  
│   ├── prajjwal1-bert-tiny_570_1epochs/  # Model outputs
│   └── logs/                  # Training logs
└── phase_3_distillation/      # Knowledge distillation
    ├── config_distill_*.gin   # Distillation configuration
    ├── distilled_models/      # Final distilled models
    └── logs/                  # Distillation logs
```

## 🎛️ Individual Script Usage

### 1. Teacher Training
```bash
python distillation/scripts/train_teachers.py \
    --model bert-base-uncased \
    --patients 570 \
    --dataset ohiot1dm \
    --epochs 5 \
    --lr 0.001 \
    --batch-size 32 \
    --seed 42
```

### 2. Student Baseline Training
```bash
python distillation/scripts/train_students.py \
    --model prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --epochs 3 \
    --lr 0.001 \
    --seed 42
```

### 3. Knowledge Distillation
```bash
python distillation/scripts/distill_students.py \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --distill-epochs 3 \
    --lr 0.001 \
    --batch-size 32 \
    --alpha 0.5 \
    --beta 0.5 \
    --kl-weight 0.1 \
    --temperature 3.0 \
    --seed 42
```

## 📊 Datasets

### Supported Datasets
- **ohiot1dm**: Ohio T1DM dataset for glucose forecasting
- **d1namo**: D1NAMO dataset for continuous glucose monitoring

### Data Path Auto-Detection
The system automatically detects data paths:
- `ohiot1dm` → `./data/ohiot1dm/raw_standardized/`
- `d1namo` → `./data/d1nemo/raw_standardized/`

Expected data structure:
```
data/
├── ohiot1dm/
│   └── raw_standardized/
│       ├── 570-ws-training.csv
│       ├── 570-ws-testing.csv
│       ├── 584-ws-training.csv
│       ├── 584-ws-testing.csv
│       └── t1dm_prompt.txt
└── d1namo/
    └── raw_standardized/
        ├── [patient files]
        └── t1dm_prompt.txt
```

## 🔧 Configuration

### Model Architecture Mapping
The system maps HuggingFace model names to internal configurations:

| Model Name | Internal Config | Layers | Hidden Size |
|------------|----------------|---------|-------------|
| bert-base-uncased | BERT | 12 | 768 |
| bert-large-uncased | BERT | 24 | 1024 |
| distilbert-base-uncased | DistilBERT | 6 | 768 |
| prajjwal1/bert-tiny | BERT | 2 | 128 |
| prajjwal1/bert-mini | BERT | 4 | 256 |
| prajjwal1/bert-small | BERT | 4 | 512 |
| prajjwal1/bert-medium | BERT | 8 | 512 |

### Distillation Loss Components

The knowledge distillation loss combines multiple components:

```
Total Loss = α × Ground_Truth_Loss + β × Teacher_Output_Loss + λ × KL_Divergence_Loss
```

- **Ground Truth Loss (α)**: Student learns from actual labels
- **Teacher Output Loss (β)**: Student mimics teacher predictions  
- **KL Divergence Loss (λ)**: Knowledge transfer via soft targets with temperature scaling

## 📈 Performance Tips

### Hyperparameter Tuning
- **Higher α**: More focus on task accuracy
- **Higher β**: More knowledge transfer from teacher
- **Higher temperature**: Softer probability distributions
- **Higher KL weight**: Stronger knowledge distillation effect

### Computational Efficiency
- Use smaller batch sizes for GPU memory constraints
- Start with fewer epochs and scale up based on convergence
- Use `--dry-run` to verify configurations before training

## 🐛 Troubleshooting

### Common Issues
1. **Model not found**: Ensure model names match supported choices exactly
2. **Data path errors**: Verify data files exist in expected structure
3. **Memory errors**: Reduce batch size or use smaller models
4. **Permission errors**: Ensure write access to output directories

### Debug Commands
```bash
# Check available models
python distillation/scripts/train_teachers.py --model all --list-checkpoints

# Verify data paths with dry run
./distill_pipeline.sh --teacher bert --student tinybert --patients 570 --dataset ohiot1dm --dry-run

# Test individual components
python distillation/scripts/train_teachers.py --model bert --patients 570 --dataset ohiot1dm --dry-run
```

## 📝 Examples

### Example 1: Quick Distillation
```bash
# Fast distillation for testing
./distill_pipeline.sh \
    --teacher bert-base-uncased \
    --student prajjwal1/bert-tiny \
    --patients 570 \
    --dataset ohiot1dm \
    --teacher-epochs 1 \
    --student-epochs 1 \
    --distill-epochs 1
```

### Example 2: High-Quality Distillation
```bash
# Comprehensive distillation with multiple patients
./distill_pipeline.sh \
    --teacher bert-large-uncased \
    --student prajjwal1/bert-small \
    --patients 570,584 \
    --dataset ohiot1dm \
    --teacher-epochs 10 \
    --student-epochs 5 \
    --distill-epochs 8 \
    --lr 0.0005 \
    --batch-size 64 \
    --alpha 0.3 \
    --beta 0.7 \
    --kl-weight 0.2 \
    --temperature 5.0
```

### Example 3: Multi-Dataset Experiments
```bash
# Train on ohiot1dm
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset ohiot1dm --seed 42

# Train on d1namo  
./distill_pipeline.sh --teacher bert-base-uncased --student prajjwal1/bert-tiny --patients 570 --dataset d1namo --seed 42
```

## 🏗️ Architecture

### Pipeline Flow
```
Input Data → Teacher Training → Student Baseline → Knowledge Distillation → Compressed Model
     ↓              ↓                ↓                     ↓                      ↓
   Raw CSV    Large Model      Small Model        Distilled Model         Efficient Model
   Patients     Training        Training           (Teacher→Student)      Ready for Deployment
```

### File Organization
```
distillation/
├── scripts/
│   ├── train_teachers.py      # Teacher model training
│   ├── train_students.py      # Student baseline training
│   └── distill_students.py    # Knowledge distillation
├── __init__.py
└── utils/                     # Utility functions
```

---

**Note**: This system is designed for temporal forecasting tasks with Time-LLM models. Ensure your data follows the expected CSV format with target columns and appropriate temporal structure.