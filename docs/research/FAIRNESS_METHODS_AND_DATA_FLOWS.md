# T1, O2, and Data-Flow Reference

## Purpose and scope

This guide explains the implemented fairness methods used in the OhioT1DM
blood-glucose (BG) and MIT-BIH ECG studies. It distinguishes immutable data
preparation from method-specific changes so that a result labelled `T1`, `O2`,
or `T1+O2` is reproducible and not interpreted as a different evaluation
dataset.

BG is the primary mechanistic study. ECG is a cross-task classification study;
its AAMI-5 and binary-ectopy endpoints use different labels and utility
metrics, so their fairness values must never be pooled with BG values.

## Short answer

| Method | What it changes | What it does not change |
| --- | --- | --- |
| Baseline KD | Trains the student from the ordinary teacher with the task KD loss. | The prepared data, split, model task, or fairness metric. |
| T1 | Retrains the **teacher** with a fairness-aware training sampler, then distils that new teacher. | Input values/waveforms, labels, evaluation data, or the student output at inference. |
| O2 | Jointly learns a group-conditional affine correction on the **student output** and applies it at inference. | The training/test split, source data, target labels, or post-hoc evaluation thresholds. |
| T1+O2 | Uses the T1 teacher as the KD source and trains the O2 student-output head. | The shared preprocessing and locked evaluation protocol. |

T1 is therefore a **teacher-side exposure intervention**. O2 is a **student
output intervention**. Neither method is a new raw-data-cleaning pipeline.

## Shared methodological rules

- The demographic attribute is used only where a method needs it: T1 needs it
  while sampling the training set; O2 needs it during training and inference
  to select the learned group parameters.
- T1 and O2 do not alter the held-out labels or the test split.
- O2 is not the reported `EO_cal` metric. O2 changes predictions before the
  clinical/classification decision; `EO_cal` changes an evaluation decision
  threshold only.
- Checkpoint selection and any tuning use development/validation data. Locked
  ECG test predictions are final evaluation artifacts, not method-selection
  inputs.
- `T1+O2` is the completed BG O2 experiment. A pure BG O2 run has not been
  completed, whereas ECG includes both O2-only and T1+O2 variants.

## OhioT1DM BG forecasting

### Data preparation shared by every BG method

1. The source CSVs are standardized to the schema `item_id`, `timestamp`, and
   `target`. This is a schema conversion, not a z-score or min-max change to
   glucose values.
2. The canonical fairness study uses the 12 OhioT1DM patients. Each patient's
   pre-existing training and testing files are kept separate, then concatenated
   into `all_patients_training.csv` and `all_patients_testing.csv`. It is not a
   random row-level split.
3. The canonical teacher configuration uses the unscaled `target` values in
   mg/dL (`preprocess_input_features=False` and
   `preprocess_label=False`). The hypoglycemia threshold consequently remains
   clinically interpretable at 70 mg/dL.
4. The forecasting dataset sorts rows by timestamp and forms overlapping
   univariate windows: six historical values as input, six context values, and
   a nine-step forecast target at five-minute frequency. Calendar features are
   month, day, weekday, hour, and five-minute bin.
5. Teacher, student, Baseline KD, T1, and T1+O2 use the same prepared training
   and test files, BERT to BERT-tiny architecture family, ten epochs, and KD
   coefficients `alpha=0.5` and `beta=0.5` in the canonical runs.

### BG task and baseline KD

The teacher is trained on the combined training CSV. The student baseline is
trained on ground-truth forecasts alone. Baseline KD freezes the teacher and
trains the student with the continuous prediction loss

$$
\mathcal{L}_{KD} =
\alpha\operatorname{MSE}(\hat y_S,y) +
\beta\operatorname{MSE}(\hat y_S,\hat y_T).
$$

During per-patient test inference, the same student checkpoint is evaluated
on each patient's original held-out file. The raw BG fairness endpoint derives
hypoglycemia from the forecast values and labels at 70 mg/dL and reports the
absolute male/female true-positive-rate gap.

### BG T1: fair teacher sampling

T1 changes only how the **teacher's training windows** are sampled. For the
configured feature `gender`, each training window receives the majority gender
of the patient rows in its six-step history. A window is hypoglycemia-positive
when at least one value in its returned supervised span (context plus forecast
target) is below 70 mg/dL.

The sampler leaves all ordinary windows at weight 1. For a hypoglycemia window
in a group with lower observed hypoglycemia prevalence, it uses

$$
w_i = \min\left(\frac{\max_g r_g}{r_{A_i}},\;2.4\right),
$$

where $r_g$ is that group's hypoglycemia-window rate. A weighted sampler draws
with replacement for the teacher's training epoch. The sampler does not edit
the window's glucose values, labels, timestamps, or patient identity.

After retraining, the fair teacher is frozen. The T1 student is then trained
with the ordinary BG KD objective against that teacher. In other words, the
reported BG `T1` effect is teacher-side; it is not a second student sampler.

### BG O2: learned per-gender output calibration

For a student prediction $\hat y_S$ and group $A$, O2 learns a scale and bias
for each group:

$$
\hat y_{O2} = s_A\hat y_S + b_A.
$$

The calibrated prediction replaces $\hat y_S$ in **both** terms of the BG KD
loss:

$$
\mathcal{L}_{T1+O2} =
\alpha\operatorname{MSE}(\hat y_{O2},y) +
\beta\operatorname{MSE}(\hat y_{O2},\hat y_T).
$$

The two group scales initialize to 1 and the biases to 0. They are optimized
with the student, then written to `student_calibration_head.json`. Inference
loads that sidecar, constructs test-window gender labels using the same
window-alignment rule, and applies the saved scale and bias before predictions
are saved and fairness is calculated. Thus, O2 requires the chosen group
attribute at deployment time.

### BG evaluation boundary

- **Utility:** RMSE is primary and MAE is secondary on held-out forecasts.
- **Raw fairness:** male/female hypoglycemia TPR gap at the fixed 70 mg/dL
  decision boundary, using the actual saved predictions.
- **Calibrated fairness:** a separate leakage-controlled evaluation threshold
  procedure. It must not be confused with O2 and is not used to train O2.

## MIT-BIH ECG classification

### Data preparation shared by every ECG method

1. The curated fairness protocol uses one record per subject: record `201` is
   retained and duplicate-subject record `202` is excluded by default.
2. Each record's waveform uses lead `MLII` when available, otherwise the first
   usable lead. It is normalized independently per record by median/MAD, with
   mean/std fallback when MAD is near zero.
3. Each eligible annotated beat becomes one fixed 256-sample waveform window:
   128 samples before and 128 after the beat anchor. Boundary windows are
   reflect-padded when possible. Nearby beats are context only; the label is
   the centre beat.
4. Raw annotation symbols map to AAMI classes `N`, `S`, `V`, `F`, and `Q`.
   The binary-ectopy endpoint reuses the same index, maps `S/V/F` to ectopy,
   maps `N` to non-ectopy, and excludes `Q`.
5. Every sample retains record, beat index, sex, age group, paced group, and
   difficulty group. Previous/next RR intervals are derived from sorted beat
   indices, converted at 360 Hz, and clipped to 0--3 seconds. They are passed
   through an optional two-feature projection for the binary-ectopy protocol;
   waveform preprocessing itself is unchanged.
6. Splits are assigned at the record level, never at the beat level. The
   deterministic stratified protocol balances sex, class coverage, overall
   class counts, and sex-by-class counts. The current curated split has 28
   train, 9 validation, and 10 test records.

### ECG classifier and baseline KD

The ECG model normalizes an input window inside the model, produces patches,
projects them to a frozen pretrained LLM backbone, pools hidden states, and
predicts class logits. The patch embedding, projection, and classification
head are trainable; the backbone is frozen by default. When RR features are
enabled, log-clipped previous/next RR values are projected and added to the
pooled representation.

For logits $z_S$, teacher logits $z_T$, label $y$, and temperature $T$, ECG KD
uses

$$
\mathcal{L}_{ECG} =
\alpha\operatorname{CE}(z_S,y) +
\beta T^2\operatorname{KL}\left(
\operatorname{softmax}(z_T/T)\;\middle\|\;
\operatorname{softmax}(z_S/T)
\right).
$$

The class-imbalance correction is part of the shared ECG training protocol:

- teacher, student baseline, Baseline KD, and O2 use a class-only weighted
  sampler, with weights inverse to class frequency capped at 50x and
  normalized to mean 1;
- a run with a weighted sampler uses ordinary cross-entropy;
- a run without a weighted sampler uses inverse-frequency class-weighted
  cross-entropy instead;
- these two corrections are mutually exclusive, because applying both caused
  rare-class collapse in diagnostic runs.

### ECG T1: group-and-class fair teacher sampling

ECG T1 replaces the class-only sampler when the fair teacher is trained. It
counts training samples in each `(fairness group, class)` cell and assigns each
sample a weight proportional to the mean cell count divided by its own cell
count. The raw weight is capped at 4x, then all weights are normalized to mean
1 before weighted sampling with replacement.

For the primary fairness feature, this increases exposure to rare sex/class
combinations. It does not change the waveform samples, their AAMI labels,
their record-level split, or the class definition. The fair teacher checkpoint
becomes the KD source for ECG T1 and T1+O2 students.

### ECG O2: learned per-group, per-class logit calibration

ECG O2 cannot use a single scalar logit shift: adding the same value to every
class logit would disappear after softmax. Instead, it learns a scale and bias
for every group and class:

$$
z_{O2,c} = s_{A,c}z_{S,c} + b_{A,c}.
$$

These calibrated logits replace student logits in both the supervised
cross-entropy and KD KL terms. The scales initialize to 1 and biases to 0;
optional regularizers keep them near that identity mapping. The head trains
jointly with the student, the selected validation checkpoint's calibration
state is saved in `student_calibration_head.json`, and the same head is loaded
and applied before test probabilities and class predictions are written.

Consequently, ECG O2 also needs the selected group attribute at inference.
It is an in-model calibration step, not a post-hoc fit on the test set.

### ECG evaluation boundary

- **Utility:** AAMI-5 uses accuracy, macro-F1, weighted-F1, and class recall;
  binary ectopy emphasizes macro-F1.
- **Fairness:** each class is assessed one-vs-rest using the gap between group
  recalls. Aggregate EO includes only classes with at least 20 true examples
  in every compared group and best-group recall of at least 0.05. Raw gaps and
  exclusion reasons remain in the report.
- **Protocol:** validation selects checkpoints and configuration. Test
  predictions are locked final-evaluation artifacts. Binary ectopy is a
  secondary endpoint; it does not replace AAMI-5.

## What changes in each variant?

| Study | Variant | Prepared input and split | Training sampler | Student output at inference |
| --- | --- | --- | --- | --- |
| BG | Baseline KD | Shared all-patient mg/dL windows | Ordinary shuffled loader | Raw student forecast |
| BG | T1 | Unchanged | Fair sampler only while training the teacher | Raw student forecast from fair-teacher KD |
| BG | T1+O2 | Unchanged | T1 teacher sampler; ordinary KD loader for the student | Group-specific affine forecast |
| ECG | Baseline KD | Shared record-safe beat index | Class-only sampler, cap 50x | Raw student logits |
| ECG | T1 | Unchanged | Group/class sampler replaces class-only sampler, cap 4x | Raw student logits from fair-teacher KD |
| ECG | O2 | Unchanged | Class-only sampler, cap 50x | Group/class affine logits |
| ECG | T1+O2 | Unchanged | T1 group/class sampler for teacher | Group/class affine logits |

## Implementation map

| Area | Primary implementation |
| --- | --- |
| BG schema standardization and all-patient assembly | `scripts/data_formatting/core/standardize_data.py`, `data_processing/combine_all_patients.py` |
| BG windows and fair-teacher sampler | `data_processing/data_sets.py`, `data_processing/data_loader.py`, `fairness/utils/sampling.py` |
| BG KD, T1/O2 training, and sidecar inference | `distillation/core/distillation_trainer.py`, `distillation/core/distillation_wrapper.py`, `distillation/scripts/train_teachers.py` |
| ECG metadata, beat index, labels, splits, and windows | `scripts/mitbih/prepare_beat_dataset.py`, `data_processing/ecg/dataset.py` |
| ECG sampling and classifier | `fairness/utils/ecg_sampling.py`, `models/ecg/time_llm_classifier.py` |
| ECG KD, T1/O2 training, sidecar inference, and evaluation | `distillation/core/ecg_classification_wrapper.py`, `scripts/pipelines/run_mitbih_fairness_distillation_pipeline.sh`, `fairness/analyzers/ecg_classifier_fairness_analyzer.py` |

## Related documents

- BG method sweep and diagnostics: `docs/FAIRNESS_SOLUTIONS_ROADMAP.md`
- BG research synthesis and evidence gate: `docs/research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md`
- ECG protocol, endpoint limits, and source artifacts:
  `docs/mitbih/ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`
- ECG classifier design: `docs/mitbih/TIMELLM_CLASSIFICATION_VARIANT.md`
- ECG distillation protocol: `docs/mitbih/TIMELLM_DISTILLATION_ROADMAP.md`

This guide documents the implemented protocol; it does not replace the evidence-strength and reporting gates in the research plan.
