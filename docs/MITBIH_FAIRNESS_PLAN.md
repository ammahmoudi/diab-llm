# MIT-BIH Time-LLM Fairness Plan

## Goal

Build a fairness evaluation plan for the new MIT-BIH project direction.

The main project goal is:

- beat-centered MIT-BIH AAMI 5-class ECG classification
- using a Time-LLM-inspired classifier as the main architecture
- with teacher, student, and distilled student comparisons
- and fairness-transfer analysis adapted from the repository's prior blood
  glucose distillation work

This document replaces the old binary-baseline framing. The main benchmark is
not normal-versus-abnormal classification. The main benchmark is multi-class
AAMI beat classification with fairness analysis attached to the teacher,
student, and distilled student systems.

## Task Definition

### Primary task

Predict one AAMI class for each annotated heartbeat from a beat-centered ECG
window.

### Sample unit

Each sample is one annotated heartbeat represented by:

- a fixed-length ECG segment centered on the beat
- one target class from the AAMI 5-class label space
- subgroup metadata used for fairness analysis

### Modeling priority

1. Train a Time-LLM-inspired classifier as the main teacher model.
2. Train a smaller student model on the same task.
3. Train a distilled student from the teacher.
4. Evaluate fairness transfer across teacher, student, and distilled student.
5. Use a simpler encoder plus classifier head only as a fallback or ablation.

### Non-goal

The fairness benchmark is not defined as a binary baseline. Binary views may be
used inside class-wise fairness analysis, but the benchmark itself remains a
5-class classification problem.

## Class Definition

### AAMI 5-class target space

The main target space is the standard AAMI-inspired 5-class beat taxonomy:

- `N`: non-ectopic beats
- `S`: supraventricular ectopic beats
- `V`: ventricular ectopic beats
- `F`: fusion beats
- `Q`: unknown or unclassifiable beats

### Intended MIT-BIH symbol mapping

Use a documented raw-symbol-to-AAMI mapping. The expected mapping is:

- `N` class:
  - `N` normal beat
  - `L` left bundle branch block beat
  - `R` right bundle branch block beat
  - `e` atrial escape beat
  - `j` nodal or junctional escape beat
- `S` class:
  - `A` atrial premature beat
  - `a` aberrated atrial premature beat
  - `J` nodal or junctional premature beat
  - `S` supraventricular premature or ectopic beat
- `V` class:
  - `V` premature ventricular contraction
  - `E` ventricular escape beat
- `F` class:
  - `F` fusion of ventricular and normal beat
- `Q` class:
  - `/` paced beat
  - `f` fusion of paced and normal beat
  - `Q` unclassifiable beat

### Exclusion policy

Exclude annotations that are not clean beat targets for the main benchmark,
including:

- rhythm markers
- comments
- artifact-only markers
- symbols that do not map cleanly into the AAMI 5-class space

### Threshold reporting rule

All main performance, distillation, and fairness results should be reported on
this same 5-class label space. If any labels are excluded, record the count and
the reason.

## Beat-Centered Input Representation

### Core representation

Represent each example as a beat-centered ECG window extracted around the
annotated beat location.

### Default waveform policy

- sampling rate: `360 Hz`
- default window length: `256` samples
- left context: `128` samples before the beat
- right context: `128` samples after the beat
- primary channel rule: use `MLII` when available; otherwise use the first
  usable lead

### Rationale

This representation matches the task better than long forecasting windows
because the label belongs to one heartbeat morphology with local temporal
context.

### Optional sensitivity analysis

After the main pipeline works, test alternative window sizes such as:

- `180` samples
- `300` samples
- `360` samples

Keep the fairness definitions fixed across window-size variants.

## Data and Preprocessing

### Data source

Primary dataset path:

`data/mit-bih-arrhythmia/`

Use MIT-BIH waveform, header, and annotation files to build:

- beat-centered windows
- AAMI labels
- record-level metadata
- subgroup assignments

### Split policy

Split by record, never by beat.

This is required to prevent leakage across training, validation, and test sets.

### Metadata to retain

Store subgroup-relevant metadata for each sample or parent record, including:

- `record_id`
- `sex`
- `age`
- `age_group`
- `primary_lead_used`
- `paced_group`
- `difficulty_group`
- optional note fields derived from headers

### Preprocessing policy

Use a minimal, auditable preprocessing stack for the main benchmark:

- no resampling from `360 Hz`
- no learned beat detector
- use provided annotations directly
- per-record normalization on the selected lead
- fixed-length extraction with boundary padding when needed

### Normalization recommendation

Use per-record robust normalization first:

```text
x_norm = (x - median(record)) / max(MAD(record), eps)
```

Fallback if the robust scale is degenerate:

```text
x_norm = (x - mean(record)) / max(std(record), eps)
```

Optional clipping after normalization:

```text
clip to [-5, 5]
```

### Augmentation policy

Do not make augmentation part of the first fairness benchmark. Add augmentation
only as a later robustness study.

## Fairness Goals and Metrics

### Main fairness goal

Measure whether subgroup performance gaps appear in the AAMI 5-class task and
whether distillation preserves, reduces, or worsens those gaps.

### Fairness-transfer goal

Adapt the repository's blood glucose fairness-distillation logic to multi-class
ECG classification.

The main question is:

- what fairness properties does the teacher have
- what fairness properties does the student have
- what fairness properties does the distilled student inherit or change

### System comparison goal

Compare fairness across four model roles:

1. Time-LLM teacher
2. student trained without distillation
3. distilled student
4. external ECG baseline used as comparator only

### Performance metrics

Report overall and subgroup values for:

- accuracy
- macro-F1
- weighted-F1
- per-class precision
- per-class recall
- per-class F1
- confusion matrix

### Fairness metrics

For each subgroup and class-aware fairness view, report:

- recall gap or TPR gap
- equal opportunity gap
- demographic parity gap
- support counts per subgroup and class
- worst-group performance
- worst-subgroup macro-F1
- average class-wise fairness gap

### Recommended fairness summary format

Compute fairness in a one-vs-rest manner for each AAMI class, then summarize:

- per-class subgroup metric tables
- maximum subgroup gap for each class
- average gap across classes
- worst-case class and subgroup pair

## Distillation Framing

### Project framing

Treat this MIT-BIH effort as the ECG classification extension of the
repository's earlier blood glucose teacher-student fairness work.

### Role of distillation

Distillation is not only an efficiency step. It is part of the fairness study.

The key fairness-transfer question is whether the distilled student:

- inherits subgroup disparities from the teacher
- amplifies subgroup disparities
- reduces subgroup disparities
- changes which class or subgroup has the worst gap

### Required model variants

At minimum, compare:

- teacher baseline
- student baseline
- distilled student

Optional later variant:

- fairness-aware distilled student

### Main architecture choice

The main teacher architecture should be a Time-LLM-inspired classifier adapted
from the repository's sequence modeling stack.

### Fallback or ablation architecture

If the full Time-LLM-inspired classifier is unstable, too slow, or not yet
ready, use a simpler encoder plus classifier head as:

- a fallback implementation
- a debugging baseline
- an ablation

That simpler architecture should not replace the main project direction.

## Thresholding Policy for Multiclass Fairness

### Primary decision rule

Use `argmax` over the 5 class logits or probabilities for the official predicted
class.

This is the primary decision rule for:

- accuracy
- macro-F1
- confusion matrix
- subgroup performance reporting
- headline fairness tables

### Class-wise fairness decomposition

For fairness metrics that require binary events, decompose the multi-class task
into five one-vs-rest views:

- `N` versus not `N`
- `S` versus not `S`
- `V` versus not `V`
- `F` versus not `F`
- `Q` versus not `Q`

Under the main policy, define these one-vs-rest events from the final `argmax`
prediction so the fairness tables remain consistent with the multi-class model's
official output.

### Secondary threshold analysis

If score-threshold analysis is needed, use per-class softmax scores only in a
secondary report.

Recommended secondary checks:

- threshold `0.5`
- class-specific validation-tuned thresholds
- calibration-aware threshold sensitivity

### Reporting rule

Keep one fixed policy for the main benchmark table. Put threshold sensitivity in
a secondary table, not in the headline results table.

## Subgroup Definitions

### Primary subgroup

Sex should be the first subgroup because it aligns with the repository's prior
fairness-transfer work.

Recommended values:

- `M`
- `F`
- `unknown`, if needed

### Secondary subgroups

Add secondary subgroup analyses when metadata coverage supports them:

- age group:
  - `<50`
  - `50-69`
  - `70+`
  - `unknown`
- paced status:
  - `paced`
  - `non_paced`
- signal difficulty:
  - `clean_or_mostly_clean`
  - `noisy_or_difficult`
- lead subgroup:
  - `MLII_primary`
  - `non_MLII_primary`

### Priority order

Use this reporting order:

1. sex
2. age group
3. paced status
4. signal difficulty
5. lead subgroup

### Data sufficiency rule

Only report subgroup fairness metrics where support is large enough to avoid
misleading instability. If a subgroup or class has very low support, keep the
count visible and mark the estimate as low-confidence.

## External Baseline Role

### Comparator status

The friend's ECG classifier is an external baseline or comparator only.

It is not:

- the main project architecture
- the default fallback architecture
- the primary distillation teacher unless explicitly chosen for a separate
  comparison

### Required evaluation constraint

If included, evaluate the external classifier on:

- the same record-level split
- the same AAMI mapping
- the same subgroup metadata
- the same fairness scripts
- the same reporting protocol

Without this alignment, the comparison is not valid.

### Intended use

Use the external model to answer two questions:

1. Does the Time-LLM-inspired classifier achieve competitive classification
   quality?
2. Does it show better, worse, or comparable subgroup fairness behavior?

## Evaluation Plan

### Stage 1: data and labeling validation

Confirm:

- record-level metadata extraction
- raw-symbol-to-AAMI mapping
- beat counts per class
- subgroup counts per split

### Stage 2: teacher benchmark

Train and evaluate the Time-LLM-inspired teacher classifier.

Report:

- overall multi-class performance
- per-class metrics
- subgroup metrics
- one-vs-rest fairness tables

### Stage 3: student benchmark

Train a smaller student without distillation on the same task and evaluate with
the same protocol.

### Stage 4: distillation benchmark

Train the distilled student and compare it directly against teacher and student
baselines.

Main question:

Does distillation improve efficiency while preserving or improving subgroup
fairness?

### Stage 5: external baseline comparison

Run the friend's ECG classifier under the same split and reporting rules.

### Stage 6: robustness checks

Add secondary analyses for:

- window-size sensitivity
- threshold sensitivity for one-vs-rest fairness
- subgroup support sensitivity
- optional calibration checks

### Final comparison table

The final summary should place the following side by side:

- teacher
- student
- distilled student
- external baseline

For each model, include:

- macro-F1
- weighted-F1
- per-class recall
- worst subgroup gap
- average class-wise fairness gap
- subgroup support notes

## Final Recommendation

Use the Time-LLM-inspired 5-class MIT-BIH classifier as the main project path.

Treat fairness as a multi-class teacher-student-distillation evaluation problem,
not as a binary screening add-on.

Use the simpler encoder plus classifier head only as a fallback or ablation.
Keep the friend's ECG classifier in the project as an external comparator only.

For the main project decision path, use this benchmark order:

1. Time-LLM teacher on AAMI 5-class classification
2. student baseline on the same split
3. distilled student fairness-transfer analysis
4. external baseline comparison

This keeps the project aligned with the new direction:

- Time-LLM classification first
- distillation second
- fairness-transfer throughout
