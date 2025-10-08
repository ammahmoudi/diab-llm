# Knowledge Distillation for Time-LLM

Simple guide for training smaller, faster models from larger teacher models.

## What is Knowledge Distillation?

- **Teacher models**: Large, accurate models (BERT, DistilBERT)
- **Student models**: Small, fast models (TinyBERT) 
- **Distillation**: Train student to mimic teacher's predictions

## Quick Start

### 1. Complete Distillation Pipeline (Recommended)

```bash
# Test first (always do this):
./distill_and_run.sh d1namo raw_standardized '001' 'bert,tinybert' 2 0.001 --dry-run

# Run full 3-step distillation pipeline:
# Step 1: Train BERT (teacher)
# Step 2: Train TinyBERT (baseline student) 
# Step 3: Distill TinyBERT from BERT (distilled student)
./distill_and_run.sh d1namo raw_standardized '001' 'bert,tinybert' 5 0.001
```

### 2. What the Pipeline Does

The `distill_and_run.sh` script automatically performs a **3-step knowledge distillation pipeline**:

1. **Train Teacher Models** (e.g., BERT) - Large, accurate models
2. **Train Student Models** (e.g., TinyBERT) - Small models trained independently  
3. **Knowledge Distillation** - Train student to mimic teacher's predictions

**Result:** You get 3 models per patient:
- **BERT** (teacher) - Large, accurate
- **TinyBERT** (baseline) - Small, independent training
- **TinyBERT-Distilled** - Small, learned from BERT teacher

### 3. Manual Steps (Advanced)

```bash
# If you want to run steps manually:
# Step 1: Train teacher model  
python distillation/scripts/train_teachers.py --models bert --dataset 584

# Step 2: Train baseline student
python distillation/scripts/flexible_experiment_runner.py --models tinybert --dataset 584

# Step 3: Distill student from teacher
python distillation/scripts/distill_students.py --teacher bert --student tinybert --dataset 584
```

### 3. Distillation Tool

```bash
# Test system:
./run_distillation.sh test

# Full pipeline:
./run_distillation.sh full-pipeline --dataset 584
```

## � What Each Command Does

### **Understanding the Models:**
- **Teacher models**: Large, powerful models (BERT, DistilBERT)  
- **Student models**: Smaller, faster models (TinyBERT)
- **Distillation**: Teaching student to copy teacher's behavior

### **Available Models:**
- **Teachers**: `bert`, `distilbert` (large, accurate)
- **Students**: `tinybert` (small, fast)

### **Available Data:**
- **d1namo**: Patients 001-007 (smaller dataset, good for testing)
- **ohiot1dm**: Patients 540-596 (larger dataset, research use)

## 🎯 Common Use Cases

### **Just Testing:**
```bash
# Quick test to see if everything works:
./distill_and_run.sh d1namo raw_standardized '001' 'tinybert' 1 0.001 --dry-run
```

### **Compare Model Sizes:**
```bash
# Compare big vs small models:
./distill_and_run.sh d1namo raw_standardized '001' 'bert,tinybert' 3 0.001
```

### **Research Experiment:**
```bash  
# Multiple patients, more training:
./distill_and_run.sh ohiot1dm raw_standardized '584,570' 'distilbert,tinybert' 10 0.0005
```

## Available Options

### Models
- **Teachers**: `bert`, `distilbert` (large, accurate)
- **Students**: `tinybert` (small, fast)

### Datasets  
- **d1namo**: Patients 001-007 (smaller, for testing)
- **ohiot1dm**: Patients 540-596 (larger, for research)

### Data Types
- `raw_standardized` (most common)
- `noisy_standardized` (with noise)

## Common Commands

### Testing
```bash
# Quick test:
./distill_and_run.sh d1namo raw_standardized '001' 'tinybert' 1 0.001 --dry-run
```

### Compare Models
```bash
# Compare teacher vs student:
./distill_and_run.sh d1namo raw_standardized '001' 'bert,tinybert' 3 0.001
```

### Multiple Patients  
```bash
# Train on multiple patients:
./distill_and_run.sh d1namo raw_standardized '001,002,003' 'tinybert' 5 0.001
```

### Research Scale
```bash
# Larger experiment:
./distill_and_run.sh ohiot1dm raw_standardized '584,570' 'distilbert,tinybert' 10 0.0005
```

## Tips

- Always use `--dry-run` first to test
- Start with d1namo data (smaller/faster)
- Use 1-2 epochs for quick tests
- Check results in `results/flexible_experiments/`

## Folder Structure

```
distillation/
├── core/           # Core distillation engines (4 files)
├── scripts/        # Main automation scripts (5 files)  
├── configs/        # Distillation configurations (5 files)
└── old_scripts/    # Legacy shell scripts (archived)
```

### Main Entry Points

```bash
# Generate single config:
./distill_config.sh <dataset> <data_type> <patient> <model>

# Generate and run experiments:
./distill_and_run.sh <dataset> <data_type> <patients> <models> [epochs] [lr]
```

## Troubleshooting

- **"No teacher checkpoint found"**: Train teacher model first
- **CUDA errors**: Check GPU memory, use smaller batch size  
- **Path errors**: Make sure you're in project root directory

## Results

After training, check:
- `results/flexible_experiments/` - experiment summaries
- `results/experiments/` - detailed logs and model files
- Compare student vs teacher performance in the summary files
