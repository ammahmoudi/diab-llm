# DEPRECATED — Replaced by Time-LLM MIT-BIH Docs

## Status

This document has been superseded by the newer Time-LLM-aligned MIT-BIH docs.

Use these instead:

- `docs/MITBIH_TIMELLM_CLASSIFICATION_VARIANT.md`
- `docs/MITBIH_TIMELLM_DISTILLATION_ROADMAP.md`
- `docs/MITBIH_TIMELLM_IMPLEMENTATION_PLAN.md`
- `docs/MITBIH_FAIRNESS_PLAN.md`

## Why this file is deprecated

This file reflects an older classifier-oriented planning stage before the project direction was fully locked.

It does **not** reflect the final intended architecture or research framing.

Main differences from the final direction:

- it predates the decision to make **Time-LLM-inspired classification** the main system
- it predates the decision to make **AAMI 5-class** the main target
- it predates the stronger focus on **teacher/student/distillation/fairness transfer**
- it does not frame the friend ECG repo purely as an **external comparator** as clearly as the new docs do

## Keep or remove?

This file is kept temporarily only for traceability.

Safe options:

1. keep it as deprecated history
2. delete it later after implementation starts and the new docs are stable

---

## Original content below

### MIT-BIH Time-LLM Classification Variant

## Goal

Build a **Time-LLM classification variant** for **MIT-BIH AAMI 5-class beat classification** inside this repository.

The purpose is to extend the existing blood glucose forecasting work into a classification setting that still supports the same higher-level research directions:

- teacher models
- student models
- knowledge distillation
- fairness evaluation
- fairness-aware generalization analysis

This document defines the preferred classification direction for ECG work in this codebase.

## Task

The target task is:

- **single-beat ECG classification**
- dataset family: **MIT-BIH Arrhythmia**
- label standard: **AAMI 5-class beat classification**

The model should take a beat-centered ECG representation and predict one of the five AAMI classes.

## Architecture Choice

### Recommended architecture

Use a **Time-LLM-inspired classifier**:

- keep the core Time-LLM design philosophy
- keep the time-series-to-token or patch-style representation logic where appropriate
- **replace the forecasting head with a classification head**
- train the full model directly for supervised beat classification

This is the primary direction.

### Non-primary alternative

A **Time-LLM encoder + classifier head** setup is allowed only as:

- an ablation
- a simplified fallback
- a debugging path if the full Time-LLM-inspired classifier is unstable

It is **not** the preferred final architecture.

## Model Changes

To adapt the current forecasting-oriented setup to classification, make these changes:

- remove next-step or horizon forecasting outputs
- replace the output layer with a **5-way classification head**
- use logits over AAMI classes instead of regression values
- change the loss from forecasting loss to **multiclass classification loss**
- update dataloading so each sample is a labeled beat example
- update metrics to classification metrics
- preserve enough modularity to support teacher, student, and distillation variants

## Classes

Use the standard **AAMI 5-class grouping**.

The five classes are:

- **N**: non-ectopic beats
- **S**: supraventricular ectopic beats
- **V**: ventricular ectopic beats
- **F**: fusion beats
- **Q**: unknown or paced or unclassifiable beats under the AAMI grouping used by the pipeline

The exact raw MIT-BIH beat-to-AAMI mapping should be implemented explicitly in preprocessing and documented in code.

## Beat Representation

The preferred input representation is **beat-centered ECG classification input**.

Recommended representation:

- extract a fixed window around each target beat
- center the sample on the annotated beat location
- use one consistent sampling and normalization policy across train, validation, and test
- keep the representation compatible with Time-LLM-style tokenization or patching

Recommended constraints:

- each sample corresponds to one labeled beat
- contextual signal around the beat is preserved
- the representation can be reused by teacher and student models
- subgroup metadata stays attached to each beat sample for fairness analysis

If needed, additional variants can be tested later, such as:

- narrower beat windows
- wider context windows
- morphology-only vs morphology-plus-context inputs

Those are secondary experiments, not the first implementation target.

## Data Work

The required data work includes:

- loading MIT-BIH ECG records and beat annotations
- mapping raw beat symbols into **AAMI 5 classes**
- constructing beat-centered examples
- defining patient-aware splits
- attaching subgroup metadata for fairness experiments
- handling class imbalance in a controlled and reproducible way
- storing processed artifacts in a reusable format for repeated training runs

Key preprocessing outputs should include:

- beat signal window
- class label
- patient identifier
- record identifier
- subgroup attributes used by fairness evaluation

## Fairness Goal

The fairness goal is to test whether the repository's fairness workflow can generalize from blood glucose forecasting to ECG classification.

This means the ECG classification pipeline should support:

- subgroup-aware evaluation
- performance comparison across groups
- disparity reporting
- compatibility with fairness analyses already used in the repository

The fairness objective is **not** to redesign the task around fairness first. The objective is to build a correct classification pipeline that can then be evaluated under the same fairness framework used in prior work.

## Distillation Goal

The distillation goal is to make the ECG classification variant compatible with the existing teacher-student research direction.

The pipeline should support:

- a stronger teacher classifier
- a smaller student classifier
- distillation using soft targets, logits, or related classification distillation signals
- comparison of accuracy, efficiency, and fairness behavior between teacher and student

The classification design should therefore avoid one-off choices that block later distillation experiments.

## Role of Friend's Repo

The friend's **ECG-Classifier-LLM** repository is an **external baseline or comparator only**.

Its role is limited to:

- inspiration for benchmarking
- architectural comparison
- result comparison
- sanity checking against another ECG-LLM-style approach

It is **not** the main architecture to port into this repository.

It is **not** the repository's primary design source.

The main implementation direction in this repository remains a **Time-LLM classification variant** aligned with the existing codebase and research goals.

## Model Stack

The intended model stack is:

1. **Primary model**
   - Time-LLM-inspired ECG classifier
   - classification head replaces forecasting head
   - used as the main teacher-capable architecture

2. **Student-compatible variants**
   - reduced-width or reduced-depth Time-LLM-style classifier
   - efficiency-oriented student versions for distillation experiments

3. **Ablation or fallback model**
   - Time-LLM encoder + lightweight classifier head
   - used only for simplification, debugging, or ablation

4. **External comparator**
   - friend's ECG-Classifier-LLM repo
   - used only as a comparison point outside the main internal stack

## Supported Labels

The supported labels for this variant are the **AAMI 5 classes**:

- `N`
- `S`
- `V`
- `F`
- `Q`

The output head should produce one prediction over these five labels.

No binary simplification is the main target in this document. The intended task here is the full **5-class** setup.

## Training Setup

Recommended training setup:

- supervised multiclass classification
- patient-aware train, validation, and test splits
- cross-entropy as the default starting loss
- class imbalance monitoring and, if needed, weighted loss or sampling
- macro and per-class reporting, not accuracy alone
- subgroup-aware fairness reporting added to the evaluation stage

Recommended metrics include:

- macro F1
- weighted F1
- per-class precision
- per-class recall
- confusion matrix
- subgroup performance gaps for fairness analysis

For distillation-enabled runs, later extensions can add:

- hard-label supervised loss
- soft-label distillation loss
- temperature-scaled teacher supervision

## Implementation Order

1. Define MIT-BIH beat extraction and AAMI 5-class label mapping.
2. Build beat-centered dataset objects with patient and subgroup metadata.
3. Implement the **Time-LLM-inspired classifier** with a 5-class head.
4. Add training and evaluation loops for multiclass classification.
5. Validate core metrics and confusion matrices on clean splits.
6. Add fairness evaluation hooks and subgroup reporting.
7. Add smaller student variants.
8. Add distillation training support.
9. Compare against external baselines, including the friend's repo if useful.
10. Run ablations, including the encoder-plus-classifier fallback if needed.

## Final Decision

The final decision is:

- this document supports a **MIT-BIH AAMI 5-class beat classification** pipeline
- the **main goal** is a **Time-LLM classification variant**
- the purpose is to support **teacher, student, distillation, and fairness generalization** from the existing blood glucose work
- the friend's **ECG-Classifier-LLM** repository is **only an external baseline or comparator**
- the **recommended architecture** is a **Time-LLM-inspired classifier with the forecasting head replaced by a classification head**
- a **Time-LLM encoder + classifier head** is **only an ablation or fallback**, not the primary design choice
