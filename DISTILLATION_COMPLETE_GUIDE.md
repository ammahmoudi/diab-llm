# 🧠 Time-LLM Knowledge Distillation Complete Guide

This comprehensive guide covers everything you need to know about training teacher models, student models, and performing knowledge distillation in the Time-LLM project.

## 📖 What is Knowledge Distillation?

Knowledge distillation is a technique for training smaller, faster models by learning from larger, more accurate models:

- **Teacher Models**: Large, accurate models (BERT, DistilBERT) with high performance but slower inference
- **Student Models**: Small, fast models (TinyBERT) that learn to mimic teacher behavior
- **Distillation Process**: Training student to copy teacher's predictions while maintaining smaller size

## 🚀 Quick Start

### Option 1: Complete Pipeline (Recommended)

Run the entire 3-phase pipeline with a single command:

```bash
cd /home/amma/LLM-TIME
source venv/bin/activate

# Quick test (always start with this)
bash distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 1 --student-epochs 1 --distill-epochs 1 --dry-run

# Full pipeline run
bash distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 1 --student-epochs 1 --distill-epochs 1
```

### Option 2: Alternative Pipeline Script

```bash
# Another pipeline option (simplified interface)
bash distill_and_run.sh ohiot1dm 570 1 bert tinybert --dry-run
bash distill_and_run.sh ohiot1dm 570 1 bert tinybert
```

## 🔧 What Each Pipeline Does

### Phase 1: Teacher Training
- Trains a large, accurate model (BERT/DistilBERT)
- High performance but slow inference
- Serves as the "teacher" for knowledge transfer

### Phase 2: Student Baseline Training  
- Trains a small, fast model (TinyBERT) independently
- Lower performance but fast inference
- Establishes baseline before distillation

### Phase 3: Knowledge Distillation
- Student learns from teacher's predictions
- Combines teacher's knowledge with student's efficiency
- Results in improved small model performance

### Phase 4: Inference & Comparison
- Tests all 3 models (teacher, baseline student, distilled student)
- Compares performance metrics and efficiency
- Generates comprehensive reports

## 📁 Organized Output Structure

All results are organized in timestamped directories:

```
distillation_experiments/pipeline_runs/pipeline_YYYY-MM-DD_HH-MM-SS/
├── configs/                    # All configuration files
│   ├── config_teacher_bert_570_1epochs.gin
│   └── config_distill_bert_to_tinybert_570.gin
├── phase_1_teacher/           # Teacher model results
│   └── bert_570_1epochs/
│       ├── logs/              # Training logs, checkpoints
│       └── summary.json       # Performance summary
├── phase_2_student/           # Student baseline results
│   ├── configs/               # Student configurations
│   └── results/               # Training results
├── phase_3_distillation/      # Distillation results
│   ├── bert_to_tinybert_570/  # Distillation logs
│   ├── summary.json           # Distillation summary
│   └── student_distilled.pth  # Final distilled model
└── inference_results/         # Performance comparison
```

## 🎯 Available Models

### Teacher Models (Large & Accurate)
- **BERT**: 12 layers, 768 dimensions - Most accurate
- **DistilBERT**: 6 layers, 768 dimensions - Balanced option

### Student Models (Small & Fast)  
- **TinyBERT**: 4 layers, 312 dimensions - Fastest option

## 📊 Available Datasets

### D1NAMO (Small, Testing)
- Patients: 001, 002, 003, 004, 005, 006, 007
- Good for: Quick testing, development
- Size: Smaller datasets, faster training

### OhioT1DM (Large, Research)
- Patients: 540, 559, 563, 570, 575, 588, 591, 596
- Good for: Research experiments, production
- Size: Larger datasets, longer training

### Data Types
- `raw_standardized`: Standard preprocessed data (recommended)
- `noisy_standardized`: Data with added noise for robustness testing

## 💡 Common Usage Examples

### 🧪 Testing & Development

```bash
# Quick functionality test
bash distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 1 --student-epochs 1 --distill-epochs 1 --dry-run

# Small scale test
bash distill_and_run.sh d1namo 001 1 bert tinybert
```

### 🔬 Research Experiments

```bash
# Multi-epoch training
bash distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 5 --student-epochs 5 --distill-epochs 3

# Compare different teacher-student pairs
bash distill_pipeline.sh --teacher distilbert --student tinybert --dataset 570 \
  --teacher-epochs 3 --student-epochs 3 --distill-epochs 2
```

### 🏭 Production Training

```bash
# High-quality training with more epochs
bash distill_pipeline.sh --teacher bert --student tinybert --dataset 570 \
  --teacher-epochs 10 --student-epochs 8 --distill-epochs 5

# Multiple patient training (requires manual scripting)
for patient in 570 575 588; do
  bash distill_pipeline.sh --teacher bert --student tinybert --dataset $patient \
    --teacher-epochs 5 --student-epochs 5 --distill-epochs 3
done
```

## 🔧 Manual Step-by-Step Execution

For advanced users who want to run individual phases:

### Step 1: Train Teacher Model

```bash
python distillation/scripts/train_teachers.py \
  --model bert \
  --dataset 570 \
  --epochs 5 \
  --output-dir "my_experiment/teacher" \
  --config-dir "my_experiment/configs"
```

### Step 2: Train Student Baseline

```bash
python distillation/scripts/flexible_experiment_runner.py \
  --dataset ohiot1dm \
  --data-type raw_standardized \
  --patients 570 \
  --models tinybert \
  --epochs 5 \
  --lr 0.001 \
  --output-dir "my_experiment/student"
```

### Step 3: Knowledge Distillation

```bash
python distillation/scripts/distill_students.py \
  --teacher bert \
  --student tinybert \
  --dataset 570 \
  --distill-epochs 3 \
  --teacher-checkpoint-dir "my_experiment/teacher" \
  --student-config-dir "my_experiment/student" \
  --output-dir "my_experiment/distillation" \
  --config-output-dir "my_experiment/configs"
```

## 🛠️ Configuration Generation

Generate individual configuration files:

```bash
# Generate teacher config
bash distill_config.sh ohiot1dm raw_standardized 570 bert 5 0.001

# Generate student config  
bash distill_config.sh ohiot1dm raw_standardized 570 tinybert 5 0.001
```

## 📈 Understanding Results

### Performance Metrics
- **RMSE**: Root Mean Square Error (lower is better)
- **MAE**: Mean Absolute Error (lower is better)  
- **MAPE**: Mean Absolute Percentage Error (lower is better)

### Efficiency Metrics
- **Model Size**: Memory footprint
- **Inference Time**: Prediction speed
- **Training Time**: Learning duration

### Expected Improvements
- **Teacher (BERT)**: Highest accuracy, slowest inference
- **Student Baseline**: Lowest accuracy, fastest inference
- **Distilled Student**: Balanced accuracy/speed (goal: close to teacher accuracy with student speed)

## 🚨 Troubleshooting

### Common Issues

**"No teacher checkpoint found"**
```bash
# Solution: Train teacher first or check path
ls distillation_experiments/pipeline_runs/*/phase_1_teacher/*/logs/*/checkpoints/
```

**"CUDA out of memory"**
```bash
# Solution: Use smaller batch size or fewer epochs
# Check GPU memory: nvidia-smi
```

**"Config file not found"**
```bash
# Solution: Check if virtual environment is activated
source venv/bin/activate
```

**"Permission denied"**
```bash
# Solution: Make scripts executable
chmod +x distill_pipeline.sh distill_and_run.sh distill_config.sh
```

### Debugging Tips

1. **Always start with --dry-run** to see what will happen
2. **Use single epochs first** for quick testing
3. **Check logs** in the generated directories
4. **Verify data paths** in the configuration files
5. **Monitor GPU usage** during training

## 📋 Best Practices

### Development Workflow
1. Start with `d1namo` dataset (smaller, faster)
2. Use 1-2 epochs for initial testing
3. Always run `--dry-run` first
4. Check results before scaling up

### Production Workflow
1. Use `ohiot1dm` dataset for final experiments
2. Train with 5-10 epochs for quality results
3. Save configurations for reproducibility
4. Document experiment parameters

### Resource Management
- Monitor GPU memory usage
- Use appropriate batch sizes
- Clean up old experiment files periodically
- Back up important model checkpoints

## 🗂️ Project Structure

```
LLM-TIME/
├── distill_pipeline.sh          # Main pipeline script
├── distill_and_run.sh          # Alternative pipeline
├── distill_config.sh           # Config generator
├── distillation/
│   ├── scripts/                # Main automation scripts
│   │   ├── train_teachers.py   # Teacher training
│   │   ├── flexible_experiment_runner.py  # Student training
│   │   ├── distill_students.py # Knowledge distillation
│   │   └── flexible_config_generator.py   # Config generation
│   ├── core/                   # Core distillation engines  
│   └── old_scripts/           # Legacy scripts (archived)
├── configs/                    # Configuration files
└── distillation_experiments/   # All experiment results
```

## 🎓 Getting Started Checklist

- [ ] Clone repository with submodules
- [ ] Create and activate virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Test with dry run: `bash distill_pipeline.sh ... --dry-run`  
- [ ] Run small experiment with 1 epoch
- [ ] Check results in `distillation_experiments/`
- [ ] Scale up to production training
- [ ] Analyze performance improvements

## 📞 Support

For questions or issues:
1. Check this guide first
2. Review logs in experiment directories  
3. Try with `--dry-run` to test setup
4. Verify virtual environment is activated
5. Check GPU availability with `nvidia-smi`

---

*This guide consolidates all distillation functionality into a single reference. All previous scattered documentation has been unified here.*