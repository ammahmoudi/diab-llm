# MIT-BIH Time-LLM Distillation Roadmap

## Goal

Build a **Time-LLM classification variant** for **MIT-BIH AAMI 5-class beat classification**, then study:

- Teacher performance
- Student performance
- Distillation behavior
- Fairness before and after distillation
- Transfer of fairness-mitigation methods from the BG work

This roadmap is the ECG-classification analogue of the Time-LLM BG distillation and fairness pipeline.

---

## Phase 0: Scope lock

### Phase 0 objective

Lock the exact research direction before implementation.

### Phase 0 decision

Main system:

- Time-LLM classification variant

Main task:

- Beat-centered MIT-BIH AAMI 5-class classification

Main labels:

- `N`
- `S`
- `V`
- `F`
- `Q`

Primary architecture choice:

- Time-LLM-inspired classifier

Secondary architecture choice:

- Time-LLM encoder plus classifier head, only as an ablation or fallback

External comparator:

- Friend's ECG-Classifier-LLM baseline repo

---

## Phase 1: Data layer

### Phase 1 objective

Prepare a distillation-ready and fairness-ready ECG classification dataset.

### Phase 1 tasks

1. Parse record metadata from `.hea`.
2. Parse beat annotations from `.atr`.
3. Choose a primary lead per record.
4. Normalize waveform per record.
5. Extract beat-centered windows.
6. Map raw symbols to AAMI 5 classes.
7. Attach subgroup metadata.
8. Exclude duplicate subject record `202` to keep one-record-per-patient accounting.
9. Create record-level splits.

### Phase 1 output artifacts

- `metadata_records.csv`
- `beat_index.csv`
- Reusable ECG classification dataset loader

### Phase 1 success criteria

Each sample has:

- Record ID
- Beat anchor index
- Input window
- AAMI 5-class label
- Fairness subgroup attributes
- Split assignment

---

## Phase 2: External baseline integration

### Phase 2 objective

Bring in the friend's ECG classifier as a comparison system.

### Phase 2 tasks

1. Clone or vendor the reference implementation in a controlled location.
2. Document exact preprocessing assumptions.
3. Adapt data loading to our record-safe protocol if needed.
4. Train and evaluate it on our split policy.
5. Save outputs for fairness comparison.

### Phase 2 role

This model is:

- A comparison baseline
- Not the main system architecture

### Phase 2 success criteria

- External baseline runs on our processed data.
- Outputs are comparable to the internal pipeline.
- Fairness can be computed on it too.

---

## Phase 3: Time-LLM classification variant

### Phase 3 objective

Adapt Time-LLM from forecasting to multi-class beat classification.

### Phase 3 changes

1. Replace the forecasting head with a classification head.
2. Add pooling or center-aware aggregation.
3. Change the loss to a multi-class classification loss.
4. Change the metrics to classification metrics.
5. Support AAMI 5-class outputs.

### Phase 3 primary design

```text
ECG beat window -> Time-LLM-style backbone -> pooling -> classification logits
```

### Phase 3 ablation design

```text
ECG beat window -> Time-LLM encoder -> pooled hidden states ->
lightweight classifier head
```

### Phase 3 success criteria

- Model trains end to end.
- Outputs 5-class logits.
- Classification metrics are stable.

---

## Phase 4: Teacher and student variants

### Phase 4 objective

Define the distillation pair inside the Time-LLM classification family.

### Phase 4 tasks

1. Select the teacher backbone size.
2. Select one or more smaller student variants.
3. Standardize the output label space across all variants.
4. Make training and evaluation scripts consistent.

### Phase 4 success criteria

- Teacher baseline classifier
- Student baseline classifier
- Comparable model interfaces for distillation

---

## Phase 5: Distillation pipeline

### Phase 5 objective

Run classification distillation experiments analogous to the BG distillation setup.

### Phase 5 tasks

1. Supervised teacher training
2. Supervised student baseline training
3. Teacher-to-student distillation
4. Save probabilities, logits, and predictions
5. Compare accuracy and fairness across teacher, student, and distilled systems

### Phase 5 loss structure

Expected components:

- Supervised classification loss
- Distillation loss on teacher outputs
- Optional fairness-aware regularization later

### Phase 5 success criteria

- Distilled student trains successfully.
- A teacher, student, and distilled comparison table is produced.

---

## Phase 6: Fairness evaluation

### Phase 6 objective

Run fairness analysis on all classification systems.

### Phase 6 primary groups

- Sex

### Phase 6 secondary groups

- Age group
- Paced versus non-paced
- Quality or difficulty subgroup
- Lead subgroup

### Phase 6 metrics

- Macro-F1 by group
- Subgroup recall or TPR
- EO Gap
- DP Gap
- FVO
- Per-class subgroup recall gaps

### Phase 6 success criteria

- Fairness report for teacher
- Fairness report for student
- Fairness report for distilled student
- Comparison summary across all systems

---

## Phase 7: Fairness mitigation transfer

### Phase 7 objective

Test whether fairness-fixing methods from the BG work generalize to ECG classification distillation.

### Phase 7 candidate directions

1. Output-level calibration ideas adapted for classification
2. Subgroup-aware loss weighting
3. Sample rebalancing or subgroup-aware replay
4. Distillation-time fairness regularization
5. Post-hoc thresholding or calibration analysis if needed

### Phase 7 success criteria

- At least one mitigation sweep is completed.
- The fairness-performance tradeoff is documented.

---

## Phase 8: Comparative report

### Phase 8 objective

Produce a clear research story across systems.

### Systems to compare

- External CNN baseline
- Time-LLM teacher classifier
- Time-LLM student classifier
- Time-LLM distilled classifier
- Fairness-mitigated variants

### Questions to answer

1. Can a Time-LLM-family model be adapted cleanly to ECG classification?
2. Does distillation change fairness in this new domain?
3. Do our prior fairness-fixing ideas transfer?
4. How does the Time-LLM family compare with a standard ECG CNN baseline?

---

## Recommended file plan

### Data layer

- `data_processing/ecg_metadata_parser.py`
- `data_processing/ecg_label_map.py`
- `data_processing/ecg_mitbih_dataset.py`

### Time-LLM classification

- `models/` or `llms/` adaptation for a classification head
- `experiments/mitbih/train_timellm_ecg_classifier.py`
- `experiments/mitbih/eval_timellm_ecg_classifier.py`

### Distillation

- `distillation/` extensions for classification mode
- `distillation/scripts/distill_mitbih_classifier.py`

### Baselines

- External CNN baseline integration scripts

### Fairness

- `fairness/analyzers/ecg_classifier_fairness_analyzer.py`
- `scripts/fairness/run_mitbih_classifier_fairness.py`

### Docs

- `docs/MITBIH_TIMELLM_CLASSIFICATION_VARIANT.md`
- `docs/MITBIH_TIMELLM_DISTILLATION_ROADMAP.md`

---

## Priority order

### Priority 1

- Data layer
- AAMI 5-class mapping
- Record-safe split

### Priority 2

- External CNN comparison baseline
- Time-LLM classification variant

### Priority 3

- Teacher and student setup
- Distillation pipeline

### Priority 4

- Fairness analysis
- Fairness mitigation transfer

---

## First milestone

The first milestone is complete when we have:

1. An AAMI 5-class beat dataset
2. The external baseline running on our split
3. The Time-LLM classification variant running
4. Basic fairness evaluation for both

---

## Final decision summary

Main system:

- Time-LLM classification variant

Main task:

- MIT-BIH AAMI 5-class beat classification

Main architecture choice:

- Time-LLM-inspired classifier

Secondary architecture choice:

- Time-LLM encoder plus classifier head as an ablation or fallback

External comparator:

- Friend's ECG classifier repo

Research goal:

- Fairness and distillation generalization from BG forecasting to ECG classification
