# Fairness Under Compression: Evidence and Experiment Plan

> **For agentic workers:** Use the `executing-plans` workflow to implement the manifest-driven confirmatory suite phase by phase, validating each gate before launching expensive runs.

**Goal:** Build balanced, auditable evidence for a BG-led, cross-domain fairness-under-KD article before any manuscript drafting begins.

**Architecture:** Use a manifest-driven five-seed BG suite as the confirmatory core, preserve the existing complete runs, and add only missing method-by-seed cells. Generate all statistics and claim decisions from one canonical evidence package, then integrate the already locked AAMI-5 and binary ECG results without retuning.

**Tech stack:** Bash orchestration, Python, PyTorch, pandas/SciPy statistics, Markdown reports, and pytest protocol checks.

---

**Status:** Evidence-building stage; not yet cleared for article drafting.

**Format decision:** Markdown research plan only. Do not start LaTeX until the readiness gate in this document passes.

**Working title:** *Fairness Under Compression: Knowledge Distillation Across Clinical Forecasting and Classification Tasks*

## 1. Decision

The repository contains a strong result, but not yet a fully balanced article-level comparison.

The strongest current finding is the five-seed OhioT1DM result: fair-teacher distillation with a learned group calibration head (T1+O2) reduces the mean raw hypoglycemia equal-opportunity (EO) gap from 0.2174 to 0.1179, a 45.8% reduction, and improves the gap in all five seeds. The full blood-glucose (BG) intervention grid also contains several useful negative results.

However, most BG methods have only one completed seed. Only Baseline KD, T1, and T1+O2 currently have five-seed BG evidence. Therefore:

- do not claim that every intervention fails robustly across seeds;
- do not claim that O2 is universally superior across tasks;
- do not use the complete single-seed grid as if every row had the same evidence strength;
- do not begin the final article or LaTeX manuscript yet;
- first repeat the essential BG method families and complete paired statistical analysis.

This plan treats the paper as evidence-gated. A negative result is acceptable and potentially valuable, but it must be reproducible, correctly scoped, and supported by mechanism checks.

## 2. Research Question

Primary question:

> How does knowledge distillation affect subgroup fairness in clinical time-series models, and which intervention points can reliably mitigate any degradation without unacceptable utility loss?

Secondary questions:

1. Does ordinary KD change fairness relative to the teacher and independently trained student?
2. Does improving the teacher transfer fairness to the student?
3. Are data reweighting, transfer-level changes, constrained objectives, representation alignment, or group erasure effective?
4. Is explicit group-aware output calibration uniquely effective for BG, or does its benefit transfer to ECG classification?
5. Which conclusions are common across BG forecasting and ECG classification, and which are task-specific?

## 3. Fixed Scope

### 3.1 Domains and endpoints

| Domain | Primary utility | Primary fairness endpoint | Role in article |
| --- | --- | --- | --- |
| OhioT1DM BG forecasting | RMSE, with MAE secondary | Male/female TPR gap for hypoglycemia at 70 mg/dL | Primary mechanistic study and complete intervention grid |
| MIT-BIH AAMI-5 classification | Accuracy and macro-F1 | Performance-qualified N/V recall gap by sex | Multiclass cross-task comparison; mixed evidence |
| MIT-BIH binary ectopy | Macro-F1 | S/V/F ectopy-recall gap by sex, Q excluded | Supportive secondary endpoint |

BG and ECG utility values and EO definitions are not numerically interchangeable. Cross-domain conclusions must concern direction, consistency, and intervention behavior, not pooled metric magnitudes.

### 3.2 Canonical seeds and protocol

Use the five fixed seeds from `scripts/utilities/seeds.py`:

```text
831363, 809906, 427368, 238822, 247659
```

For the confirmatory BG matrix, keep the canonical BERT teacher, BERT-tiny student, OhioT1DM patient set, preprocessing, forecast horizon, ten epochs, optimizer settings, and KD coefficients fixed. Do not tune a method on the test outputs. If a hyperparameter must be selected, use a separate development screen and freeze it before the five-seed confirmatory run.

## 4. Current Evidence Inventory

### 4.1 Five-seed BG evidence already complete

| Method | RMSE, mean +/- SD | EO raw, mean +/- SD | EO calibrated, mean +/- SD | Current conclusion |
| --- | ---: | ---: | ---: | --- |
| Baseline KD | 23.5284 +/- 0.2702 | 0.2174 +/- 0.0061 | 0.0675 +/- 0.0078 | Stable reference |
| T1 fair teacher | 22.8454 +/- 0.3305 | 0.2013 +/- 0.0077 | 0.0568 +/- 0.0057 | Small directional improvement |
| T1+O2 calibration head | 22.9420 +/- 0.6669 | 0.1179 +/- 0.0325 | 0.0445 +/- 0.0041 | Strong positive result; EO improves in 5/5 seeds |

The direction of the T1+O2 raw-EO improvement is consistent, but its magnitude varies by seed. That variation must be shown rather than hidden.

### 4.2 Complete BG single-seed method grid

| Intervention family | Methods already run | Single-seed observation | Evidence status |
| --- | --- | --- | --- |
| Baselines | Teacher, Student, Baseline KD | KD is worse than the standalone student on RMSE and EO in the matched run | Descriptive until teacher/student references are aligned with repeated evidence |
| Teacher/source | T1, T2 | T1 helps at teacher stage; per-group teachers T2 do not close the gap | T1 headline repeated; T2 single seed |
| Data weighting | Oversampling; oversampling + TPR loss | Utility worsens and EO remains high | Single seed |
| Fairness loss | Equalized-odds v1; soft/focal TPR v2 | EO remains high | Single seed |
| Transfer labels | K1; T1+K1 | K1-only performs poorly; T1+K1 partially recovers T1 | Single seed |
| Representation | K3 at weights 100 and 200 | Best RMSE among single runs, but EO remains near baseline | Two weights, one seed each |
| Selective transfer | K4 | Engaged loss does not close EO | Single seed |
| Student constraint | O1 | Stable run, but EO remains high | Single seed |
| Output correction | T1+O2 | Large utility-compatible EO reduction | Five seeds complete |
| Group erasure | O3 | Adversary reaches chance while EO remains high | Strong mechanism check, but one seed |

### 4.3 ECG evidence already complete

AAMI-5 has five-seed Teacher, Student, Baseline KD, T1, O2, and T1+O2 results. No paired fairness comparison is statistically significant; T1 has the strongest directional N/V EO reduction versus Baseline KD.

Binary ectopy also has five-seed locked-test results. O2 and T1+O2 improve mean utility and mean EO relative to Baseline KD, but the result is supportive rather than decisive because validation favors Baseline KD and one high-gap seed influences the mean test reduction.

The defensible cross-domain statement today is:

> Group-aware output correction is strongly effective for BG and directionally promising for binary ECG, but not consistently beneficial for AAMI-5. Fairness mitigation is task- and endpoint-dependent.

## 5. Methods That Belong in the BG Comparison

### 5.1 Required confirmatory matrix

The final BG main table should contain these five-seed rows:

| Tier | Method | Why it is required |
| --- | --- | --- |
| Reference | Teacher | Establish source-model utility and fairness |
| Reference | Student, no KD | Separate compression effects from KD effects |
| Reference | Baseline KD | Primary comparison for every intervention |
| Source | T1 | Test whether a fairer teacher transfers its gain |
| Data/loss | Oversampling + TPR loss | Representative data-reweighting plus fairness-loss method |
| Transfer | T1+K1 | Representative teacher-label correction |
| Representation | T1+K3, locked at weight 200 | Representative alignment method; weight 200 has the better single-seed EO and RMSE |
| Selective transfer | T1+K4 | Representative minority-event KD reweighting |
| Constraint | T1+O1 | Representative constrained-optimization method |
| Output | T1+O2 | Primary positive method |
| Erasure | T1+O3 | Essential mechanism test: group information can be erased without closing EO |
| Source specialization | T2 | Essential mechanism test: group-specific teachers can preserve base-rate disparity |

This is the preferred strong-result matrix: 12 rows x 5 seeds. Existing complete seed runs must be reused, not retrained.

### 5.2 Secondary single-seed ablations

The following rows can remain in a secondary ablation table unless compute permits repeating them:

- equalized-odds loss v1;
- soft/focal TPR loss v2;
- oversampling only;
- K1-only without T1;
- T1+K3 weight 100.

They are useful for explaining design evolution, but they do not all need to be headline rows if the representative family-level methods above are repeated.

### 5.3 Explicit exclusions

- K2 group-conditional KD temperature is not defined for point-valued BG regression and must not be relabeled as a new method.
- T3 synthetic counterfactual augmentation requires physiological validation and is a separate research project.
- Do not add new methods merely to enlarge the table. The present intervention taxonomy is already broad enough.

## 6. Strong-Result Gates

Article drafting may begin only after Gates A-E pass. Gate F controls the strength of cross-domain claims.

### Gate A: Protocol integrity

All confirmatory BG runs must use:

- the same 12 OhioT1DM patients and preprocessing;
- the same five fixed seeds;
- identical training budget and checkpoint-selection rule;
- frozen method hyperparameters;
- complete per-patient/window predictions;
- a run manifest recording commit, seed, method, teacher checkpoint, arguments, and completion state.

A partial directory or checkpoint alone is not a completed run.

### Gate B: Evidence completeness

Minimum requirement:

- Baseline KD, T1, T1+O2: 5/5 complete, already satisfied;
- Teacher and Student: 5/5 comparable references;
- T1+O1, T1+K1, T1+K3, T1+K4, T1+O3, T2, and the representative data/loss method: 5/5 complete;
- all required methods have the same seed set and auditable prediction files.

If compute is constrained, run this priority order:

1. T1+O3 and T2, because they support the base-rate mechanism argument;
2. T1+O1, T1+K3, and T1+K4, covering objective, representation, and selective-transfer failures;
3. T1+K1 and oversampling + TPR loss;
4. repeated Teacher and Student references.

The paper may instead use a smaller confirmatory matrix only if the omitted methods are explicitly labeled exploratory single-seed ablations and the conclusions are narrowed accordingly.

### Gate C: Statistical support

For every repeated BG method, report:

- mean, sample SD, median, and range across seeds;
- paired per-seed delta versus Baseline KD for RMSE and EO;
- number of seeds improved versus Baseline KD;
- 95% confidence interval for the paired mean or median delta;
- a paired permutation test or Wilcoxon signed-rank test, with exact small-sample handling;
- Holm correction within each prespecified family of comparisons;
- effect size, not only a p-value.

Five seeds provide limited power. A non-significant result must be described as inconclusive or directional, not proof of equivalence. For key negative claims, report the confidence interval and define a smallest effect size of interest before analysis.

Recommended practical thresholds for claim wording, to freeze before running the missing seeds:

- utility non-inferiority: mean RMSE increase no greater than 1.0 mg/dL versus the relevant baseline;
- meaningful raw-EO improvement: absolute reduction at least 0.03 and relative reduction at least 15%;
- robust direction: improvement in at least 4/5 seeds;
- strong positive result: robust direction plus a confidence interval excluding zero, when the test has enough resolution;
- strong negative mechanism result: mechanism diagnostic succeeds in at least 4/5 seeds while the EO improvement remains below the prespecified meaningful threshold.

These are research decision rules, not clinical safety thresholds.

### Gate D: Mechanism validation

A method counts as a valid negative result only if its mechanism engaged:

| Method | Required diagnostic |
| --- | --- |
| Oversampling | effective sample/group/event proportions |
| TPR fairness loss | nonzero loss and gradient contribution |
| K1 | offsets loaded and teacher targets changed by group |
| K3 | alignment loss magnitude and trajectory |
| K4 | minority-hypoglycemia KD weights and weighted-loss change |
| O1 | constraint violation and dual-variable trajectory |
| O2 | learned scales/biases, identity-regularization term, and inference sidecar use |
| O3 | adversary accuracy or cross-entropy near chance, plus stable predictive training |
| T2 | both teacher checkpoints, correct group routing, and per-group sample counts |

Without these diagnostics, "method failed" could mean "implementation was inactive."

### Gate E: Fairness and clinical completeness

The main BG result must include more than one aggregate gap:

- group-specific TPR and FNR for hypoglycemia;
- false-positive rate and precision or positive predictive value;
- event support counts by group;
- overall and group-specific RMSE/MAE;
- raw threshold result as primary;
- leakage-free patient-holdout calibrated result as secondary;
- per-patient distributions or bootstrap intervals clustered by patient;
- utility-fairness Pareto plot;
- calibration parameters and deployment requirement for group membership.

A reduced EO gap caused by uniformly poor detection is not a fairness success. TPR levels and event counts must accompany every EO value.

### Gate F: Cross-domain claim strength

The final cross-domain conclusion must satisfy all of the following:

- no pooling of BG and ECG metric values;
- AAMI-5 and binary endpoints reported separately;
- binary test results described as locked supportive evidence, not tuning evidence;
- AAMI-5 null/mixed results retained, not hidden;
- no claim of independent clinical replication because both ECG endpoints use MIT-BIH;
- no universal-superiority language for O2.

## 7. Execution Plan

### Phase 1: Build a manifest-driven BG confirmatory runner

Create a single resumable runner rather than launching the existing one-method scripts manually.

**Files to create or modify:**

- create `scripts/pipelines/run_bg_fairness_confirmatory_suite.sh`;
- modify or wrap the existing method runners under `scripts/pipelines/`;
- create `scripts/fairness/aggregate_bg_confirmatory_suite.py`;
- add protocol tests under `tests/`.

Required behavior:

1. resolve the five seeds from `scripts/utilities/seeds.py`;
2. define one locked method registry with exact flags and expected run-directory names;
3. reuse complete existing runs;
4. validate checkpoint, sidecar, all 12 patient outputs, and expected row counts before skipping;
5. write a method/seed manifest before training;
6. write `.training_complete` only after checkpoint and diagnostics exist;
7. write `.complete` only after inference and artifact validation pass;
8. fail closed on partial or mismatched artifacts;
9. support `METHODS=...`, `SEEDS=...`, and `DRY_RUN=1` for safe scheduling;
10. never select or tune a method from test metrics.

Validation commands for the implementation stage should include:

```bash
bash -n scripts/pipelines/run_bg_fairness_confirmatory_suite.sh
pytest -q tests/test_bg_fairness_confirmatory_protocol.py
DRY_RUN=1 METHODS=baseline_kd,t1_o3 SEEDS=831363 \
  bash scripts/pipelines/run_bg_fairness_confirmatory_suite.sh
```

### Phase 2: Complete missing method x seed cells

Run only cells absent from the validated manifest. Keep existing Baseline KD, T1, and T1+O2 five-seed artifacts.

Expected minimum new runs under the preferred matrix:

- Teacher and Student repeated references: up to 10 runs;
- nine missing rows at five seeds (two references and seven interventions), less any reusable seed-42-equivalent artifacts: up to 45 runs.

Before launching, the dry-run manifest must print the exact number of missing training and inference cells. The article plan should use that count rather than an estimate.

### Phase 3: Produce one canonical BG evidence package

The aggregator should produce:

```text
<canonical-bg-pipeline>/confirmatory/
  protocol_manifest.json
  run_status.csv
  per_seed_metrics.csv
  aggregate_metrics.csv
  paired_comparisons.csv
  mechanism_diagnostics.csv
  patient_bootstrap_intervals.csv
  utility_fairness_pareto.csv
  BG_CONFIRMATORY_ANALYSIS.md
  .complete
```

The aggregator must use sample SD for seed summaries and preserve seed-level values. It must refuse to produce `.complete` if a required method lacks a seed or prediction artifact.

### Phase 4: Integrate existing ECG evidence without retuning

Use the current canonical AAMI-5 and binary artifacts. Regenerate the cross-domain report only after the BG confirmatory package is complete.

The combined table should distinguish:

- exact matched methods: Baseline KD, T1, and T1+O2;
- methods available only in BG;
- pure O2, available in ECG but not currently in BG;
- primary versus supportive endpoints;
- statistically supported, directional, null, and mechanism-only findings.

Pure O2 for BG is optional. Run it only if the paper needs to separate the O2 main effect from the T1 interaction. If added, predeclare it before inspecting its five-seed results and include both O2 and T1+O2 in the confirmatory family.

### Phase 5: Freeze conclusions before article writing

Create a claim-evidence table with one row per proposed abstract claim:

| Claim | Required artifact | Allowed wording if gate passes | Fallback wording |
| --- | --- | --- | --- |
| Ordinary KD changes BG fairness | paired Teacher/Student/KD evidence | "KD increased the EO gap relative to the standalone student" | "In the available comparison, KD was associated with a larger gap" |
| T1+O2 improves BG fairness | five-seed paired analysis | "reduced raw EO consistently across seeds" | "showed a directional mean reduction" |
| Most training-time methods do not close BG EO | repeated representative families plus diagnostics | "did not achieve the prespecified meaningful reduction" | "single-seed screens did not suggest improvement" |
| O3 supports a base-rate mechanism | repeated chance-level adversary plus null EO effect | "group erasure was insufficient to close the gap" | "one run suggests group erasure may be insufficient" |
| O2 transfers across domains | BG plus both ECG endpoints | "benefit was endpoint-dependent" | "binary ECG was supportive while AAMI-5 was null" |

Only after this table is frozen should the article outline, prose, figures, and eventual LaTeX source be started.

## 8. Analyses and Figures Required Before Writing

### Main tables

1. BG five-seed confirmatory utility/fairness table.
2. Paired deltas versus Baseline KD with confidence intervals and corrected tests.
3. Mechanism-diagnostic table.
4. Matched BG/AAMI-5/binary table for Baseline KD, T1, and T1+O2.

### Main figures

1. BG utility versus raw-EO Pareto scatter with seed points and method means.
2. Paired seed-slope plot for Baseline KD, T1, and T1+O2.
3. Intervention taxonomy diagram: source, data, transfer, objective, representation, output.
4. Cross-domain effect-direction plot using within-domain standardized or percentage deltas, never raw pooled metrics.

### Supplementary analyses

- group-specific event support and confusion metrics;
- patient-clustered bootstrap sensitivity analysis;
- raw versus patient-holdout calibrated thresholds;
- K3 weight and mechanism traces;
- O1 dual trajectory;
- O2 parameter distributions;
- O3 adversary trajectory;
- T2 routing and group sample audit;
- all single-seed ablations clearly labeled exploratory.

## 9. Conclusions Currently Allowed

The following statements are already supported, with scope labels:

1. **BG, five seeds:** T1+O2 lowers mean raw EO from 0.2174 to 0.1179 versus Baseline KD and improves all five seeds while maintaining similar mean RMSE.
2. **BG, five seeds:** T1 alone provides a smaller directional improvement than T1+O2.
3. **BG, single-seed grid:** many diverse interventions did not materially improve raw EO in the existing screen.
4. **AAMI-5, five seeds:** mitigation effects are mixed and no paired fairness result is statistically significant.
5. **Binary ECG, five seeds:** O2-family methods are supportive on the locked test set, but validation and seed sensitivity prevent a universal claim.
6. **Cross-domain:** fairness behavior under KD depends on task, endpoint, support, and intervention point.

## 10. Conclusions Not Yet Allowed

Do not write any of these as established findings:

- "All training-time fairness methods fail."
- "The disparity is proven to be caused only by base rates."
- "O2 is the best fairness method for clinical time series."
- "The intervention generalizes across diseases or cohorts."
- "A non-significant difference proves equivalence."
- "Calibrated EO is near zero."
- "The ECG experiments independently replicate the BG result."

Preferred cautious mechanism wording, even after repetition:

> The results are consistent with subgroup event prevalence and operating-point differences being important drivers of the observed BG EO gap; successful representation erasure alone is insufficient to remove it.

## 11. Stop Rules

Stop expanding experiments and start the article only when:

- Gates A-E pass for BG;
- Gate F passes for cross-domain reporting;
- the canonical BG package has `.complete`;
- every main-table number is generated by a script from preserved artifacts;
- every proposed claim maps to an artifact and evidence tier;
- exploratory rows are visibly separated from confirmatory rows;
- no result is selected because it looked best on the held-out test set.

If a method has unstable or contradictory five-seed behavior, keep the result. The correct article may be a mixed or negative-results paper; strength comes from protocol discipline and mechanistic evidence, not from making every method positive.

## 12. Source of Truth

- BG method definitions and single-seed grid: `docs/FAIRNESS_SOLUTIONS_ROADMAP.md`
- Canonical BG artifact paths: `docs/EXPERIMENT_FOLDER_REGISTRY.md`
- BG single-seed table: `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/fairness_comparison_results.csv`
- BG current five-seed table: `distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/multiseed_robustness_results.csv`
- BG five-seed runner: `scripts/pipelines/run_multiseed_robustness_experiment.sh`
- AAMI-5/binary detailed comparison: `experiments/mitbih_fairness_pipeline_protocol_fixed_all_seeds_20260712/BG_ECG_CROSS_DOMAIN_COMPARISON.md`
- Binary canonical comparison: `experiments/mitbih_binary_ectopy_five_seed/BG_AAMI5_BINARY_COMPARISON.md`
- MIT-BIH protocol and limitations: `docs/mitbih/ECG_TUNING_AND_CLASS_LIMITATIONS_ROADMAP.md`

## 13. Immediate Next Action

Do not launch experiments directly from the individual legacy scripts. First implement and dry-run the manifest-driven BG confirmatory suite in Phase 1. Its status table will identify exactly which method x seed cells can be reused and which must be trained. Then run the missing cells in the priority order from Gate B.
