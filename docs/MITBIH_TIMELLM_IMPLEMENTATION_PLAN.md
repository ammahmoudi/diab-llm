# MIT-BIH Time-LLM Classification Implementation Plan

## Goal

Turn the current repository into a working **Time-LLM AAMI 5-class ECG classification + distillation + fairness** pipeline.

## Non-breaking implementation policy

The current BG forecasting system must remain usable while this ECG work is added.

So implementation should follow these rules:

1. **Add new files instead of rewriting BG pipeline files when possible.**
2. **Keep ECG logic under separate MIT-BIH-specific modules, runners, and configs.**
3. **Make classification mode opt-in, not the default path.**
4. **Do not break existing Time-LLM BG forecasting commands, configs, or distillation flows.**

### Recommended separation strategy

Use new or clearly separated files such as:

- `data_processing/ecg_*`
- `experiments/mitbih/*`
- `distillation/scripts/distill_mitbih_*`
- `fairness/analyzers/ecg_*`

If an existing shared module must be changed, prefer:

- adding a new `task_type` or `mode` flag
- preserving old defaults exactly
- isolating ECG classification paths behind explicit configuration

### Practical rule

The BG system should still work exactly as before if the user never touches the new MIT-BIH files or config modes.

This plan is implementation-focused. It assumes the research decisions are already made:

- Main task = MIT-BIH beat-centered AAMI 5-class classification
- Main architecture = Time-LLM-inspired classifier
- Secondary architecture = encoder + classifier head as fallback/ablation
- External comparator = friend’s ECG CNN classifier

---

## 1. Implementation targets

We need four connected subsystems:

1. **ECG data layer**
2. **Time-LLM classification model variant**
3. **Classification distillation pipeline**
4. **Fairness analysis for multi-class ECG outputs**

---

## 2. Data layer tasks

### 2.1 Metadata parsing

Create:

- `data_processing/ecg_metadata_parser.py`

Responsibilities:

- Read `.hea` files
- Extract record ID, age, sex, leads, notes
- Create metadata table

Output:

- `data/mit-bih-arrhythmia/metadata_records.csv`

### 2.2 Label mapping

Create:

- `data_processing/ecg_label_map.py`

Responsibilities:

- Read raw MIT-BIH symbols
- Map to AAMI 5 classes:
  - `N`
  - `S`
  - `V`
  - `F`
  - `Q`
- Define exclusions and unknown-handling clearly

### 2.3 Beat-window dataset builder

Create:

- `data_processing/ecg_mitbih_dataset.py`

Responsibilities:

- Load ECG waveform
- Choose primary lead, prefer `MLII`
- Normalize per record
- Extract beat-centered windows
- Attach class label and subgroup metadata
- Support record-level split

Expected sample output:

```python
{
    "record_id": ...,
    "beat_index": ...,
    "x": signal_window,
    "y": class_id,
    "sex": ...,
    "age_group": ...,
    "paced_group": ...,
    "difficulty_group": ...
}
```

### 2.4 Split policy

Implement record-level split builder:

- Train / val / test
- Seed-controlled
- No beat-level leakage

Optional output:

- `beat_index.csv`

---

## 3. Time-LLM classification model tasks

### 3.1 Add classification mode

Wherever the current Time-LLM code defines forecasting head behavior, add a classification mode.

Core change:

```text
forecast head -> classification head
```

### 3.2 Add classification head

Needed behavior:

- Input = hidden sequence output from encoder/backbone
- Pooling = mean pooling or center-aware pooling
- Head = linear logits for 5 classes

Recommended first version:

- Mean pooling
- Linear layer to 5 logits

### 3.3 Support fallback ablation path

Allow optional mode:

- Backbone frozen or semi-frozen
- Simple encoder + classifier head variant

This is for comparison and debugging.

### 3.4 Model output contract

The model should return:

- Logits
- Probabilities
- Predicted class

Not forecast windows.

---

## 4. Training pipeline tasks

### 4.1 Standard classifier training runner

Create:

- `experiments/mitbih/train_timellm_ecg_classifier.py`

Responsibilities:

- Load ECG dataset
- Build model in classification mode
- Train teacher or student variant
- Save checkpoint
- Save validation/test predictions

### 4.2 Config support

Need configuration options for:

- Window size
- Lead selection policy
- Number of classes
- Pooling mode
- Teacher/student model size
- Learning rate
- Class weighting

### 4.3 Loss

Main loss:

- Multi-class cross-entropy

Optional:

- Weighted cross-entropy for imbalance

### 4.4 Metrics during training

Track:

- Accuracy
- Macro-F1
- Weighted-F1
- Per-class recall
- Confusion matrix summary

---

## 5. Distillation pipeline tasks

### 5.1 Add classification distillation mode

Extend the distillation pipeline to support:

- Classification teacher logits
- Classification student logits
- Classification distillation loss

### 5.2 Distillation runner

Create:

- `distillation/scripts/distill_mitbih_classifier.py`

Responsibilities:

- Load teacher checkpoint
- Load student config
- Compute supervised classification loss
- Compute distillation loss from teacher outputs
- Train distilled student

### 5.3 Distillation comparisons

We need three core systems:

- Teacher baseline
- Student baseline
- Distilled student

Optional later:

- Fairness-mitigated distilled student

---

## 6. External comparator tasks

### 6.1 Bring in friend baseline

Integrate friend’s model as external baseline.

Possible structure:

- `baselines/ecg_classifier_llm/`
- Or documented cloned repo location

### 6.2 Standardize evaluation

Make sure friend baseline uses:

- Our data split
- Our label mapping
- Our subgroup metadata
- Our fairness evaluation scripts

This is critical. Otherwise comparison is not meaningful.

---

## 7. Fairness analysis tasks

### 7.1 Build ECG classifier fairness analyzer

Create:

- `fairness/analyzers/ecg_classifier_fairness_analyzer.py`

Responsibilities:

- Load predictions and labels
- Compute subgroup performance
- Compute multi-class fairness summaries
- Compute one-vs-rest fairness for each class

### 7.2 Required fairness outputs

For each group:

- Support count
- Macro-F1
- Per-class recall
- TPR gaps
- EO Gap
- DP Gap
- FVO

### 7.3 Distillation comparison

Need fairness reports for:

- Teacher
- Student
- Distilled student
- External CNN baseline

---

## 8. Fairness mitigation tasks

After the core pipeline works, add mitigation experiments.

### Candidate transfers from BG work

1. Subgroup-aware weighting
2. Oversampling / replay ideas
3. Fairness-regularized distillation loss
4. Output-level calibration for classification probabilities
5. Per-class subgroup calibration studies

---

## 9. File creation order

### First batch

1. `data_processing/ecg_metadata_parser.py`
2. `data_processing/ecg_label_map.py`
3. `data_processing/ecg_mitbih_dataset.py`

### Second batch

1. Time-LLM classification head adaptation
2. `experiments/mitbih/train_timellm_ecg_classifier.py`

### Third batch

1. External baseline integration hooks
2. `fairness/analyzers/ecg_classifier_fairness_analyzer.py`

### Fourth batch

1. `distillation/scripts/distill_mitbih_classifier.py`
2. Fairness mitigation experiment scripts

---

## 10. Minimum viable milestone

The first real milestone is:

1. Beat-centered AAMI 5-class dataset exists
2. Time-LLM classification variant trains
3. Friend baseline trains on same split
4. Fairness report runs on both

The second milestone is:

1. Teacher/student/distilled comparison exists

---

## 11. Recommended immediate next step

Start with the **data layer**, because everything depends on it.

Immediate implementation order:

1. Metadata parser
2. AAMI label mapper
3. Beat-window dataset builder
4. Split generator

Only after that:

1. Classification head adaptation
2. Trainer
3. Fairness analyzer
4. Distillation runner
