# DEPRECATED — Replaced by Time-LLM MIT-BIH Roadmaps

## Status

This roadmap is deprecated.

Use these instead:

- `docs/MITBIH_TIMELLM_DISTILLATION_ROADMAP.md`
- `docs/MITBIH_TIMELLM_IMPLEMENTATION_PLAN.md`
- `docs/MITBIH_FAIRNESS_PLAN.md`

## Why this file is deprecated

This file reflects an older roadmap stage where the project was still framed as:

- generic ECG classification first
- binary-first progression
- Time-LLM adaptation later

That is no longer the final direction.

The current direction is:

- **Time-LLM-inspired classifier as the main system**
- **AAMI 5-class as the main target**
- **teacher/student/distillation/fairness transfer** as the core research framing
- friend ECG repo as **external baseline only**

## Keep or remove?

This file is kept temporarily only for traceability.

It can be removed later after implementation stabilizes.

---

## Original content below

### MIT-BIH Classification Roadmap

## Goal

Create a clean roadmap for adding an ECG classification pipeline to this repository and then applying the fairness framework to it.

---

## Phase 0: Scope lock

### Phase 0 objective

Decide the first supported task and evaluation scope.

### Phase 0 decision

First supported task:

- beat-centered binary classification
- `normal` vs `abnormal`

### Phase 0 deliverables

- label mapping spec
- subgroup metadata spec
- split policy spec

---

## Phase 1: Data layer

### Phase 1 objective

Prepare MIT-BIH data in a classification-friendly format.

### Phase 1 tasks

1. parse record metadata from `.hea`
2. parse beat annotations from `.atr`
3. choose primary lead per record
4. normalize signal per record
5. extract beat-centered windows
6. attach beat label and subgroup metadata
7. create record-level split

### Phase 1 output files

- `data/mit-bih-arrhythmia/metadata_records.csv`
- `data/mit-bih-arrhythmia/beat_index.csv`

### Phase 1 success criteria

Every training sample has:

- record ID
- anchor beat index
- fixed window
- class label
- subgroup labels
- split assignment

---

## Phase 2: Standard classifier baseline

### Phase 2 objective

Build a standard, simple classification runner before any Time-LLM variant.

### Phase 2 model

First baseline:

- 1D CNN

### Phase 2 input

- shape `[window_size, 1]`
- default `window_size = 256`

### Phase 2 output

- binary abnormal probability

### Phase 2 training

- weighted BCE or weighted cross-entropy
- validation split from record-level split
- early stopping

### Phase 2 evaluation

- accuracy
- precision
- recall
- F1
- AUROC
- confusion matrix

### Phase 2 success criteria

- stable baseline training
- saved checkpoint
- saved test predictions

---

## Phase 3: Fairness integration

### Phase 3 objective

Apply fairness analysis to classifier outputs.

### Phase 3 primary groups

- sex

### Phase 3 secondary groups

- age group
- paced vs non-paced
- quality/difficulty group
- lead subgroup

### Phase 3 metrics

- EO Gap
- DP Gap
- FVO
- subgroup TPR / FPR / F1

### Phase 3 thresholds

- default threshold `0.50`
- validation-optimized threshold
- recall-oriented threshold if needed

### Phase 3 success criteria

- one full fairness report for binary classification
- sex fairness table
- age fairness table

---

## Phase 4: Multi-class extension

### Phase 4 objective

Extend from binary to standard ECG class structure.

### Phase 4 label target

AAMI-like 5-class:

- `N`
- `S`
- `V`
- `F`
- `Q`

### Phase 4 evaluation additions

- macro-F1
- per-class recall
- per-class subgroup recall gaps
- one-vs-rest EO gaps

### Phase 4 success criteria

- stable 5-class baseline
- per-class fairness breakdown

---

## Phase 5: Time-LLM classification variant

### Phase 5 objective

Create a classification variant of the model.

### Phase 5 required change

Replace the forecast output head with a classification head:

```text
ECG window -> encoder -> pooled representation -> classifier logits
```

### Phase 5 variants to test

1. Time-LLM-inspired encoder + classification head
2. Time-LLM backbone as feature extractor
3. compare against 1D CNN baseline

### Phase 5 success criteria

- classifier variant trains end-to-end
- fair comparison with standard baseline

---

## Phase 6: Optional forecasting-transfer study

### Phase 6 objective

If desired, run the forecasting-based ECG transfer experiment separately.

### Phase 6 task

- predict future ECG waveform
- optionally derive beat events from forecasted waveform
- compare with direct classification approach

### Phase 6 rationale for optional status

- more faithful to original DiabLLM logic
- but weaker and more complex clinically

---

## Recommended file plan

### Data files

- `data_processing/ecg_metadata_parser.py`
- `data_processing/ecg_label_map.py`
- `data_processing/ecg_mitbih_dataset.py`

### Training files

- `experiments/mitbih/train_ecg_classifier.py`
- `experiments/mitbih/train_ecg_classifier_config.yaml` or similar

### Fairness files

- `fairness/analyzers/ecg_classifier_fairness_analyzer.py`
- `scripts/fairness/run_mitbih_classifier_fairness.py`

### Script files

- `scripts/mitbih/build_metadata.py`
- `scripts/mitbih/prepare_beat_dataset.py`
- `scripts/mitbih/run_classifier_pipeline.sh`

### Documentation files

- `docs/MITBIH_CLASSIFICATION_VARIANT.md`
- `docs/MITBIH_CLASSIFICATION_ROADMAP.md`

---

## Priority order

### Priority 1 tasks

- metadata parser
- beat dataset builder
- binary label mapping

### Priority 2 tasks

- standard 1D CNN classifier runner
- fairness integration for binary task

### Priority 3 tasks

- AAMI 5-class extension
- Time-LLM classification variant

---

## First milestone

The first milestone is complete when we have:

1. record-level beat dataset
2. binary normal vs abnormal classifier
3. saved predictions on test set
4. fairness report by sex and age group

---

## Final recommended path

1. build classification dataset
2. run standard classifier baseline
3. run fairness analysis
4. extend to multi-class
5. only then build Time-LLM classification variant

This is the safest and cleanest path.
