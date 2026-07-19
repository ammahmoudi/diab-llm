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

## Terms used in this brief

| Term | Plain-language meaning | Mathematical form |
| --- | --- | --- |
| KD: knowledge distillation | Train a smaller student model to learn from both the real labels and the predictions of a larger teacher model. | $\mathcal{L}_{KD}=\alpha\mathcal{L}_{task}(S,y)+\beta\mathcal{L}_{teacher}(S,T)$ |
| Raw EO gap | Absolute difference between groups in true-positive rate for the clinically important event. For BG, the event is hypoglycemia at or below 70 mg/dL. Lower is better. | $\Delta EO=\lvert TPR_{group\;1}-TPR_{group\;2}\rvert$ |
| Calibrated EO gap (`EO_cal`) | EO gap after using a separate decision threshold for each group. The thresholds are fitted on calibration patients and evaluated on different held-out patients, so the evaluation avoids fitting and testing on the same patients. It does not change model predictions. | $\Delta EO_{cal}=\lvert TPR_M(\hat y<\tau_M)-TPR_F(\hat y<\tau_F)\rvert$ |
| RMSE | Root mean squared error for BG forecasts. It measures average numerical prediction error; lower is better. | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)^2}$ |
| Macro-F1 | Classification quality averaged equally across classes, so common classes do not dominate the score. Higher is better. | $\frac{1}{C}\sum_{c=1}^{C}F1_c$ |
| AAMI-5 | Five ECG beat groups: `N` normal, `S` supraventricular ectopic, `V` ventricular ectopic, `F` fusion, and `Q` paced/unknown. | $y\in\{N,S,V,F,Q\}$ |
| Binary ectopy | A simpler ECG endpoint: `N` is non-ectopic; `S`, `V`, and `F` are ectopic; `Q` is excluded. An ectopic beat is a beat occurring outside the normal rhythm pattern. | $y_{ectopy}=\mathbb{1}[y\in\{S,V,F\}]$ |
| T1: fair teacher | Retrain the teacher so underrepresented clinically relevant group/event examples are sampled more often, then distil that teacher into the student. | $p_i\propto w_i$; BG uses larger $w_i$ for lower-prevalence hypoglycemia windows, and ECG uses $w_i\propto1/n_{group,class}$ |
| T2: group-specific teachers | Train a separate teacher for each group and route a sample to its matching teacher during KD. | $\hat{y}_{T,i}=f_{T,A_i}(x_i)$ |
| O1: constrained EO | Add a fairness constraint during training and increase its penalty when the model exceeds the allowed EO gap. | $\mathcal{L}=\mathcal{L}_{KD}+\lambda\max(0,\Delta EO-\epsilon)$ |
| O2: output calibration | Learn a small group-specific correction to the final student prediction. For BG it adjusts the predicted glucose value; for ECG it adjusts each class logit before the predicted class is chosen. | BG: $\hat{y}=s_A\hat{y}_S+b_A$; ECG: $z_c'=s_{A,c}z_c+b_{A,c}$ |
| T1+O2 | Distil from the fair T1 teacher and apply the learned O2 output correction to the student. | $\mathcal{L}=\alpha\mathcal{L}_{task}(O2(S),y)+\beta\mathcal{L}_{teacher}(O2(S),T_{T1})$ |
| O3: adversarial erasure | Train the student to make its hidden representation less predictive of group membership. | $\min_{\theta}\max_{\phi}\;\mathcal{L}_{KD}-\gamma\mathcal{L}_{group}(g_{\phi}(h_{\theta}(x)),A)$ |
| K1: corrected teacher targets | Shift the teacher target by group before the student learns from it. | BG: $\tilde{y}_T=\hat{y}_T+\delta_A$; ECG: $\tilde{z}_{T,c}=z_{T,c}+\delta_{A,c}$ |
| K3: feature alignment | Penalize differences between group hidden representations during KD. | $\mathcal{L}=\mathcal{L}_{KD}+\lambda\left\|C_{group\;1}-C_{group\;2}\right\|_F^2$ |
| K4: selective KD replay | Give more KD weight to clinically important examples from the underrepresented group. | $\mathcal{L}=\alpha\mathcal{L}_{task}+\beta\frac{1}{n}\sum_iw_i\mathcal{L}_{teacher,i}$, with $w_i>1$ for target events |

`T1+O2` means that the student is distilled from the fair T1 teacher and also
uses the learned O2 output correction. The labels `T`, `O`, and `K` are short
method identifiers only; they do not represent clinical categories.

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

| Method | RMSE | Raw EO gap | Calibrated EO gap | Raw-EO change versus Baseline KD |
| --- | ---: | ---: | ---: | ---: |
| Baseline KD | 23.5284 +/- 0.2702 | 0.2174 +/- 0.0061 | 0.0675 +/- 0.0078 | Reference |
| T1 | 22.8454 +/- 0.3305 | 0.2013 +/- 0.0077 | 0.0568 +/- 0.0057 | -7.4% |
| T1+O2 | 22.9420 +/- 0.6669 | 0.1179 +/- 0.0325 | 0.0445 +/- 0.0041 | -45.8%; lower gap in every completed repetition |

Interpretation: T1 alone gives a small directional fairness improvement. The
combined T1+O2 approach produces a large, consistent reduction in the BG raw
EO gap while maintaining similar average RMSE.

`EO_cal` is reported as a secondary decision-threshold analysis. It shows how
much gap remains after leakage-controlled group-specific threshold adjustment;
it does not replace the primary raw-EO result at the fixed 70 mg/dL clinical
threshold. O2 is different: it learns and applies a correction to the model
outputs themselves before either EO metric is calculated.

### ECG evidence

#### Binary ectopy: locked-test results

| Method | Macro-F1 | Macro-F1 change versus KD | EO gap | EO change versus KD |
| --- | ---: | ---: | ---: | ---: |
| Teacher | 0.6951 +/- 0.0560 | -0.0049 | 0.0535 +/- 0.0398 | -0.0062 |
| Student, no KD | 0.7063 +/- 0.0715 | +0.0063 | 0.0861 +/- 0.0599 | +0.0264 |
| Baseline KD | 0.7000 +/- 0.0314 | Reference | 0.0596 +/- 0.0964 | Reference |
| T1 | 0.6890 +/- 0.0536 | -0.0110 | 0.0516 +/- 0.0672 | -0.0081 |
| O2 | 0.7127 +/- 0.0160 | +0.0127 | 0.0239 +/- 0.0211 | -0.0357 |
| T1+O2 | 0.7265 +/- 0.0345 | +0.0264 | 0.0349 +/- 0.0400 | -0.0248 |

For binary ectopy, positive macro-F1 change and negative EO change are better.
O2 and T1+O2 improve both mean utility and mean EO versus Baseline KD. This is
supportive rather than definitive: validation averages favor Baseline KD, and
one unusually high Baseline KD gap strongly affects the mean test EO.

#### AAMI-5: locked-test results

| Method | Accuracy | Macro-F1 | Macro-F1 change versus KD | EO gap | EO change versus KD |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher | 0.6633 +/- 0.0833 | 0.4222 +/- 0.0248 | -0.0194 | 0.1124 +/- 0.0255 | -0.0326 |
| Student, no KD | 0.6175 +/- 0.0602 | 0.4236 +/- 0.0281 | -0.0180 | 0.2187 +/- 0.0524 | +0.0737 |
| Baseline KD | 0.6939 +/- 0.0865 | 0.4416 +/- 0.0232 | Reference | 0.1450 +/- 0.0908 | Reference |
| T1 | 0.7019 +/- 0.0645 | 0.4375 +/- 0.0174 | -0.0041 | 0.1259 +/- 0.0536 | -0.0191 |
| O2 | 0.7035 +/- 0.1160 | 0.4440 +/- 0.0251 | +0.0024 | 0.1484 +/- 0.0961 | +0.0034 |
| T1+O2 | 0.7196 +/- 0.0468 | 0.4416 +/- 0.0120 | +0.0000 | 0.1349 +/- 0.0540 | -0.0101 |

For AAMI-5, T1 has the strongest fairness direction, but its macro-F1 is
slightly lower than Baseline KD. O2 improves macro-F1 but makes mean EO
slightly worse. T1+O2 improves accuracy and mean EO, but does not provide a
consistent all-metric improvement. Sparse classes and low S-class recall limit
the endpoint.

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

## Future Work Methodology

Future work should use the existing BG T1+O2 result as a fixed reference and
evaluate the preliminary intervention families under the same patient splits,
data processing, model architecture, training settings, and event-level
fairness definition. The evaluation should compare each method with Baseline
KD, T1, and T1+O2 using paired utility and raw-EO changes with uncertainty
intervals.

The secondary BG methods should be evaluated as distinct mechanism tests:

- **Data and loss methods:** repeat oversampling and direct TPR/EO losses to
  test whether increasing minority-event exposure or penalizing disparity can
  reduce the output gap.
- **Objective and transfer methods:** repeat O1, K1, and K4 to test constrained
  optimization, group-corrected teacher targets, and selective KD replay.
- **Representation and source methods:** repeat K3, O3, and T2 to test feature
  alignment, removal of group-predictive representations, and group-specific
  teachers. Report the corresponding mechanism diagnostics, including the
  alignment loss, adversary performance, routing behavior, and subgroup support.

ECG work should keep the current protocol fixed and use validation data for
all model or hyperparameter selection. A future external ECG evaluation should
use a record-safe split and the same support-qualified fairness definition;
the current locked test results must not be used for retuning.

Finally, deployment-oriented work should assess whether the group attribute
needed by O2 is available, reliable, and permissible at inference. If it is
not, group-agnostic alternatives should be compared with the same utility and
fairness criteria.

## Source documents

- `docs/research/FAIRNESS_KD_ARTICLE_EVIDENCE_PLAN.md`
- `docs/research/FAIRNESS_METHODS_AND_DATA_FLOWS.md`
- `docs/FAIRNESS_KD_JOURNEY_REPORT.md`
- `docs/mitbih/ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`
