# Fairness Under Compression: Decision Brief

## Current conclusion

Knowledge distillation (KD) can preserve useful blood-glucose forecasting
accuracy while worsening gender fairness in hypoglycemia detection. In the
completed OhioT1DM repeated evaluation, fair-teacher distillation with learned
group-aware output calibration (`T1+O2`) is the only completed method that
consistently reduces the raw fairness gap without a meaningful RMSE penalty.

## Problem

The project compresses a Time-LLM BERT teacher into a smaller BERT-tiny
student for efficient clinical time-series inference. Average error alone is
insufficient: a model can forecast well overall while detecting true
hypoglycemia less often for one gender.

The primary fairness measure is the absolute difference in hypoglycemia
true-positive rates between male and female patients (raw EO gap). Lower is
better. Standard KD has a mean raw EO gap of `0.2174`, which leaves a clinically
important disparity despite acceptable forecasting accuracy.

## Study design

| Element | BG primary study | ECG cross-task study |
| --- | --- | --- |
| Task | Nine-step blood-glucose forecasting | AAMI-5 and binary-ectopy beat classification |
| Data | 12 OhioT1DM patients; original patient train/test splits preserved | Curated MIT-BIH record split; duplicate-subject record 202 excluded |
| Model | BERT teacher -> BERT-tiny student | Time-LLM classifier teacher -> smaller classifier student |
| Utility | RMSE primary, MAE secondary | Accuracy/macro-F1 for AAMI-5; macro-F1 for binary ectopy |
| Fairness | Male/female hypoglycemia TPR gap at 70 mg/dL | Sex-based, support-qualified class-recall gap |
| Evidence | Repeated-run results for Baseline KD, T1, and T1+O2 | Repeated-run results for all main variants |

BG and ECG metrics are not numerically comparable. ECG tests whether the
intervention family transfers to another task; it is not an independent BG
clinical cohort.

## Methods tested

| Method | Main idea | Current evidence |
| --- | --- | --- |
| Teacher | Larger reference model trained on task labels. | Reference evidence; BG preliminary, ECG repeated-run results. |
| Student | Smaller model trained only on task labels. | Reference evidence; BG preliminary, ECG repeated-run results. |
| Baseline KD | Student learns from ground truth and frozen teacher outputs. | Repeated-run results in BG and ECG. |
| T1: fair teacher | Retrain the teacher with fair sampling, then distil it. BG upweights lower-prevalence hypoglycemia windows; ECG upweights rare sex/class cells. | Repeated-run results in BG and ECG. |
| O2: output calibration | Jointly learn group-specific student-output corrections and apply them at inference. BG corrects continuous forecasts; ECG corrects per-class logits. | BG only in T1+O2; ECG O2 and T1+O2 have repeated-run results. |
| Secondary BG mechanism screen | Test direct fairness losses, data weighting, constrained optimization, target correction, representation methods, and group-specific teachers. | Preliminary results and scoped conclusions below. |

`T1` changes training-sample exposure only. `O2` changes the student output
and requires the selected group attribute during inference. Neither method
changes the held-out data, labels, or split.

## Main evidence

### Confirmed BG results

| Method | RMSE | Raw EO gap | Change versus Baseline KD |
| --- | ---: | ---: | ---: |
| Baseline KD | 23.5284 +/- 0.2702 | 0.2174 +/- 0.0061 | Reference |
| T1 | 22.8454 +/- 0.3305 | 0.2013 +/- 0.0077 | -7.4% |
| T1+O2 | 22.9420 +/- 0.6669 | 0.1179 +/- 0.0325 | -45.8%; lower gap in every completed repetition |

Interpretation: T1 alone gives a small directional fairness improvement. The
combined T1+O2 approach produces a large, consistent reduction in the BG raw
EO gap while maintaining similar average RMSE.

### ECG evidence

#### Binary ectopy: locked-test results

| Method | Macro-F1 | EO gap |
| --- | ---: | ---: |
| Teacher | 0.6951 +/- 0.0560 | 0.0535 +/- 0.0398 |
| Student, no KD | 0.7063 +/- 0.0715 | 0.0861 +/- 0.0599 |
| Baseline KD | 0.7000 +/- 0.0314 | 0.0596 +/- 0.0964 |
| T1 | 0.6890 +/- 0.0536 | 0.0516 +/- 0.0672 |
| O2 | 0.7127 +/- 0.0160 | 0.0239 +/- 0.0211 |
| T1+O2 | 0.7265 +/- 0.0345 | 0.0349 +/- 0.0400 |

O2 and T1+O2 improve mean locked-test macro-F1 and mean EO versus Baseline KD.
This is supportive rather than definitive: validation averages favor Baseline
KD, and one unusually high Baseline KD gap strongly affects the mean test EO.

- **AAMI-5:** effects are mixed. T1 has the best directional EO result, while
  O2 does not improve mean EO over Baseline KD. Sparse classes and low S-class
  recall limit the endpoint.

## Secondary BG methods: preliminary conclusions

The methods below test different explanations for the BG disparity. Each has
initial evidence only, so the conclusion describes what the current
implementation did or did not demonstrate; it does not establish a general
method failure.

| Method | Initial BG result | What the result can conclude |
| --- | --- | --- |
| Direct fairness losses | Raw EO `0.2194` and `0.2300`; RMSE `23.672` and `24.426`. | The implemented differentiable TPR/EO losses did not reduce the raw gap in the tested configurations and increased error. |
| Oversampling only | Raw EO `0.2326`; RMSE `24.918`. | Increasing exposure to minority-group hypoglycemia windows alone was insufficient to improve raw EO. |
| Oversampling + TPR loss | Raw EO `0.2225`; RMSE `24.141`. | Combining data reweighting with the tested fairness loss still did not repair the raw gap. |
| T1+O1 constrained EO | Raw EO `0.2135`; RMSE `23.141`. | The projected dual-ascent constraint was not sufficient, by itself, to move the model out of the high-gap range. |
| K1 corrected teacher targets | K1-only: raw EO `0.2326`, RMSE `25.677`; T1+K1: raw EO `0.1981`, RMSE `22.654`. | A fairer teacher helps, but static group-specific target offsets appear weaker than learned output calibration. |
| T1+K3 feature alignment | Best run: raw EO `0.1919`, RMSE `22.438` at weight 200. | Representation alignment can improve utility, but the tested alignment did not produce the large raw-EO reduction seen with O2. |
| T1+K4 selective KD replay | Raw EO `0.2056`; RMSE `23.413`. | Giving more KD weight to minority-group critical events did not by itself close the raw gap. |
| T1+O3 adversarial erasure | Raw EO `0.2118`; RMSE `23.048`; adversary performance near chance. | Removing readily predictable group information from the hidden representation was not sufficient to remove the output fairness gap. |
| T2 group-specific teachers | Raw EO `0.2197`; RMSE `22.956`. | Group-specialized teachers did not improve raw EO in this run and may preserve group-specific event patterns; subgroup data support remains a concern. |

## What the results support

- KD fairness must be evaluated at the clinical event level, not only with
  average error or classification accuracy.
- For BG forecasting, a fairer teacher helps but transfers only part of its
  benefit to the student.
- For BG forecasting, group-aware output calibration is the strongest
  completed intervention.
- O2 should be described as task- and endpoint-dependent, not universally
  superior across clinical time series.
- The current secondary BG methods are useful mechanism tests, but their
  preliminary results do not establish robust failure or success.

## Limits and risks

- The BG cohort has 12 patients, so subgroup estimates remain limited.
- Most secondary BG methods need repeated-run confirmation.
- O2 requires a group attribute at inference, creating deployment, governance,
  privacy, and missing-data considerations.
- ECG is a different task on one dataset; it does not independently validate
  BG clinical performance.
- Locked ECG test results must not be used to choose another method or retune
  the pipeline.

## Choices for next work

| Choice | Purpose | Resulting scope |
| --- | --- | --- |
| Confirm the secondary BG methods | Establish whether the preliminary O1, K1, K3, K4, O3, T2, and data/loss findings hold under repeated evaluation. | A complete comparison of intervention families. |
| Focus on the current BG finding | Present T1+O2 as the main completed BG result and keep all secondary methods clearly preliminary. | A narrower output-calibration study. |
| Assess deployment feasibility | Decide whether the required group attribute can be used at inference and, if not, investigate group-agnostic alternatives. | A clinically actionable calibration strategy. |
| Extend external evidence | Keep the current ECG results fixed, then evaluate the approach on an independent ECG dataset. | Stronger evidence for cross-task generalization. |

The current evidence supports the BG T1+O2 finding. It does not yet support a
general claim that O2 is the best method for every task or that all other
interventions fail.

## Source documents

- `docs/research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md`
- `docs/research/FAIRNESS_METHODS_AND_DATA_FLOWS.md`
- `docs/FAIRNESS_KD_JOURNEY_REPORT.md`
- `docs/mitbih/ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`
