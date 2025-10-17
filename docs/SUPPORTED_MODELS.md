# 🤖 Supported Models in Time-LLM Distillation Pipeline

This document lists all models supported across the Time-LLM implementation and distillation pipeline.

## 📋 Model Compatibility Matrix

| Model Name | HuggingFace ID | Parameters | Time-LLM | Teacher | Student | Distillation |
|------------|----------------|------------|----------|---------|---------|--------------|
| **LLAMA** | `huggyllama/llama-7b` | ~6.7B | ✅ | ✅ | ❌ | ❌ |
| **GPT2** | `openai-community/gpt2` | ~117M | ✅ | ✅ | ❌ | ✅ |
| **BERT** | `google-bert/bert-base-uncased` | ~110M | ✅ | ✅ | ✅ | ✅ |
| **BERT-Large** | `bert-large-uncased` | ~340M | ❌ | ✅ | ❌ | ❌ |
| **DistilBERT** | `distilbert/distilbert-base-uncased` | ~66M | ✅ | ✅ | ✅ | ✅ |
| **TinyBERT** | `huawei-noah/TinyBERT_General_4L_312D` | ~14M | ✅ | ✅ | ✅ | ✅ |
| **BERT-tiny** | `prajjwal1/bert-tiny` | ~4.4M | ✅ | ✅ | ✅ | ✅ |
| **BERT-mini** | `prajjwal1/bert-mini` | ~11M | ❌ | ✅ | ✅ | ✅ |
| **BERT-small** | `prajjwal1/bert-small` | ~29M | ❌ | ✅ | ✅ | ✅ |
| **BERT-medium** | `prajjwal1/bert-medium` | ~41M | ❌ | ✅ | ✅ | ✅ |
| **MiniLM** | `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large` | ~33M | ✅ | ✅ | ✅ | ✅ |
| **MobileBERT** | `google/mobilebert-uncased` | ~25M | ✅ | ✅ | ✅ | ✅ |
| **ALBERT** | `albert/albert-base-v2` | ~12-18M | ✅ | ✅ | ✅ | ✅ |
| **OPT-125M** | `facebook/opt-125m` | ~125M | ✅ | ✅ | ✅ | ✅ |

## 🎯 Recommended Teacher-Student Pairs

### High Performance Pairs
```bash
# BERT → TinyBERT (Most tested)
--teacher bert-base-uncased --student prajjwal1/bert-tiny

# BERT → DistilBERT (Balanced)
--teacher bert-base-uncased --student distilbert-base-uncased

# DistilBERT → BERT-tiny (Good compression)
--teacher distilbert-base-uncased --student prajjwal1/bert-tiny
```

### Experimental Pairs
```bash
# MobileBERT → BERT-tiny (Mobile optimized)
--teacher google/mobilebert-uncased --student prajjwal1/bert-tiny

# ALBERT → MiniLM (Efficient pair)
--teacher albert/albert-base-v2 --student nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large
```

## 🔧 Model Name Mappings

### Input Formats Accepted
The distillation pipeline accepts both short names and full HuggingFace model IDs:

**Short Names:**
- `bert`, `distilbert`, `tinybert`, `minilm`, `mobilebert`, `albert`

**Full HuggingFace IDs:**
- `google-bert/bert-base-uncased`
- `distilbert/distilbert-base-uncased`
- `prajjwal1/bert-tiny`
- `huawei-noah/TinyBERT_General_4L_312D`
- `nreimers/MiniLMv2-L6-H384-distilled-from-BERT-Large`
- `google/mobilebert-uncased`
- `albert/albert-base-v2`
- `facebook/opt-125m`

### Example Usage
```bash
# Using short names
bash distill_pipeline.sh --teacher bert --student tinybert --patients 570 --dataset ohiot1dm

# Using full HuggingFace IDs
bash distill_pipeline.sh \
  --teacher google-bert/bert-base-uncased \
  --student prajjwal1/bert-tiny \
  --patients 570 --dataset ohiot1dm
```

## 📊 Model Characteristics

### Teacher Models (Large, High Accuracy)
- **BERT** (110M): Best general performance
- **DistilBERT** (66M): Good balance of size/performance
- **GPT2** (117M): Decoder-only architecture
- **OPT-125M** (125M): Meta's efficient decoder

### Student Models (Small, Fast Inference)
- **BERT-tiny** (4.4M): Smallest, fastest
- **TinyBERT** (14M): Purpose-built for distillation
- **MiniLM** (33M): Good performance/size trade-off
- **MobileBERT** (25M): Mobile-optimized

## ⚠️ Important Notes

1. **LLAMA models** are only supported in the base Time-LLM model, not in distillation (too large for typical distillation scenarios)

2. **BERT variants** (`prajjwal1/bert-*`) are correctly mapped to their respective configurations in all scripts

3. **Model configurations** are automatically set based on the model name, including:
   - Layer count (`llm_layers`)
   - Hidden dimensions (`llm_dim`)
   - Model comments for tracking

4. **Filename sanitization** is applied automatically for model names with forward slashes

## 🔍 Verification

To verify model support, you can test the mappings:

```bash
cd /home/amma/LLM-TIME
python3 -c "
from distillation.scripts.train_teachers import TeacherTrainer
teacher = TeacherTrainer()
print('Supported teacher models:', list(teacher.teacher_models.keys()))
"
```

This ensures consistent model support across all components of the distillation pipeline.