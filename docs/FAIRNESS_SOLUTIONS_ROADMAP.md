# Fairness in Clinical KD — Solutions Roadmap

> Context: Standard fairness interventions (loss regularization, oversampling) fail in our
> BERT→BERT-tiny glucose forecasting distillation pipeline because the raw EO Gap is driven
> by a clinically real 3:1 male:female hypoglycemia prevalence imbalance, not model bias.
> This document tracks proposed solutions for future experiments / paper contributions.

---

## Status Legend

- [ ] Not started
- [~] In progress
- [x] Done

---

## Intervention Point 1 — Fix the Teacher

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| T1 | [x] | **Fair teacher retraining** | Retrained the teacher with fair sampling and evaluated both the teacher and its distilled student. The teacher improves substantially, but the student only inherits part of that gain. | Teacher-side source fix works, but KD transfer is lossy. |
| T2 | [ ] | **Per-group teachers** | Train one teacher per gender group, then distill both into a single shared student via multi-teacher KD. Each teacher specializes on its group's hypo pattern. | Eliminates prevalence mismatch in soft labels entirely. More compute-heavy. |
| T3 | [ ] | **Counterfactual data augmentation** | Synthesize additional female hypoglycemia windows (e.g., via interpolation or GAN) to match male hypo prevalence before teacher training. Addresses root cause directly. | Most principled fix. Requires validation that synthetic data is realistic. |

---

## Intervention Point 2 — Fix the Knowledge Transfer

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| K1 | [x] | **Calibrated soft labels** | Implemented calibrated soft-label KD using per-gender teacher offsets and evaluated it end to end, both standalone (K1-only) and on the fair teacher (T1+K1). K1-only is the worst run in the sweep (RMSE 25.677). T1+K1 recovers most of the T1 fairness gain (EO_raw 0.1981) but does not beat O2 and leaves the calibrated gap higher (EO_cal 0.0719). | Negative result. Source-level label shifting alone is not enough; even on a fair teacher it underperforms the learned calibration head. |
| K2 | [ ] | **Group-conditional KD temperature** | Use higher softmax temperature for male batches during KD. Higher temperature flattens overconfident male hypo predictions, reducing the information the student extracts from majority-group patterns. | Simple, interpretable. Easy to ablate temperature values. |
| K3 | [ ] | **Fairness-aware feature alignment** | Penalize divergence of intermediate hidden-state distributions across groups (not just output predictions). Add MMD or CORAL loss between male/female intermediate representations. Forces student to learn group-invariant features. | Works at representation level. Stronger than output-level constraints. |
| K4 | [ ] | **Selective KD replay** | Upweight the KD loss specifically on minority-group (female) hypoglycemic windows. Asymmetric distillation — student is pushed harder to match the teacher on rare events for the underrepresented group. | Targeted. Doesn't distort majority-group learning. |

---

## Intervention Point 3 — Fix the Student Objective

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| O1 | [x] | **Projected dual-ascent EO constraint** | Implemented a constrained distillation objective on top of the fair teacher with an adaptive dual variable and EO-gap violation penalty. The run completed end to end, but the student still stayed in the critical raw-gap range. | Scientifically negative result: operationally valid, but not sufficient to close the fairness gap. |
| O2 | [x] | **Learned calibration head** | Trained a small per-group linear calibration layer jointly with distillation (`y_final = W_group * y_student + b_group`) on top of the fair teacher. The learned head is persisted alongside the student checkpoint and applied automatically at inference time. | Best completed student-side method so far. Improves both RMSE and EO Gap materially relative to T1-only and T1+O1. |
| O3 | [ ] | **Adversarial group erasure** | Add a gradient-reversal discriminator head that tries to predict gender from the student's hidden states. Penalize the student for being gender-predictable at the representation level. | Prevents encoding group information entirely. Strong fairness guarantee. Risk of underfitting. |

---

## Recommended Experiment Order for Paper

### Phase 1 (Diagnosis) — DONE

- [x] Run baseline + 4 fairness variants
- [x] Show all training-time interventions fail (EO Gap unchanged)
- [x] Re-evaluate Fix B with leakage-free patient-holdout calibration

### Phase 2 (Source Fix)

- [x] **T1**: Retrain fair teacher → teacher improves, student inherits only part of the gain
- [x] **K1**: Calibrated soft labels → negative result, worse than baseline KD and T1

### Phase 3 (Best Combined)

- [x] **T1 + K1**: Fair teacher + calibrated soft labels → completed as completeness check; recovers most of the T1 fairness gain (EO_raw 0.1981) but does not beat O2 and leaves the calibrated gap higher (EO_cal 0.0719)
- [x] **T1 + O1**: Fair teacher + projected dual-ascent EO constraint → completed, but still critical on raw EO Gap
- [x] **T1 + O2**: Fair teacher + learned calibration head → completed, best student-side fairness result so far
- [x] Compare completed Phase 2+3 runs on RMSE vs EO Gap tradeoff curve

### Current Combined Findings

Full comparison (OhioT1DM, BERT→BERT-tiny, all-patients pipeline, leakage-free
patient-holdout calibration, folds=2). EO Gap = |Male_TPR − Female_TPR| for
hypoglycemia detection; lower is fairer.

| Model | RMSE | EO_raw | EO_cal | Assessment (raw) |
| --- | --- | --- | --- | --- |
| Teacher baseline (BERT) | 24.162 | 0.2302 | 0.0761 | ❌ CRITICAL |
| Teacher fair sampling (T1 teacher) | 22.564 | 0.1925 | 0.0478 | ⚠️ CONCERNING |
| Student baseline (no KD) | 21.989 | 0.1948 | 0.0534 | ⚠️ CONCERNING |
| Distilled — no fairness | 23.361 | 0.2171 | 0.0766 | ❌ CRITICAL |
| Distilled + HypoglycemiaTPR loss (v1, EqualizedOdds) | 23.672 | 0.2194 | 0.0629 | ❌ CRITICAL |
| Distilled + HypoglycemiaTPR loss (v2, focal+soft) | 24.426 | 0.2300 | 0.0764 | ❌ CRITICAL |
| Distilled + Oversampling only | 24.918 | 0.2326 | 0.0752 | ❌ CRITICAL |
| Distilled + Oversampling + HypoglycemiaTPR loss | 24.141 | 0.2225 | 0.0650 | ❌ CRITICAL |
| Distilled from Fair Teacher (T1) | 23.828 | 0.2089 | 0.0579 | ❌ CRITICAL |
| Distilled from Fair Teacher + O1 Constraint | 23.141 | 0.2135 | 0.0656 | ❌ CRITICAL |
| Distilled from Fair Teacher + K1 Calibrated Soft Labels | 22.654 | 0.1981 | 0.0719 | ⚠️ CONCERNING |
| **Distilled from Fair Teacher + O2 Calibration Head** | **22.645** | **0.1189** | **0.0286** | ⚠️ MODERATE |
| Distilled + K1 Calibrated Soft Labels | 25.677 | 0.2326 | 0.0706 | ❌ CRITICAL |

Interpretation: T1 is effective at the teacher stage, but most KD variants still fail
to preserve the full teacher fairness/accuracy gain. Every training-time intervention
(TPR losses v1/v2, oversampling, and combinations) leaves the raw EO Gap in the critical
range. O1 improved RMSE relative to baseline KD and T1-only distillation, but it still
failed to reduce the raw EO Gap below the critical range. T1+K1 recovers most of the T1
fairness gain (EO_raw 0.1981) yet does not beat O2 and leaves the calibrated gap higher,
while K1-only is the worst run overall. O2 is the only completed student-side method that
materially improves both fairness and accuracy at once, lowering the raw EO Gap to 0.1189
and the leakage-free calibrated EO Gap to 0.0286 while keeping RMSE near the fair teacher.
A second quiet finding: per-gender threshold calibration flattens almost every model's
EO_cal into the 0.05–0.08 band regardless of training-time intervention, which further
undercuts the loss-based methods — only O2 stands apart on both raw and calibrated gaps.
The current evidence is that teacher quality matters, but the transfer mechanism itself
must also encode group-aware calibration to recover a substantial fairness gain.

### Recommended Next Experiments

1. Run a small multi-seed robustness check on baseline KD, T1, and T1+O2.
2. Treat `T1 + K1` as low priority unless you need a complete combinational grid for the paper.
3. If you want one stronger follow-on method beyond O2, prioritize `T2` or `K3` rather than more output-level penalties.

### Paper Narrative

> "Standard fairness regularization mostly fails in clinical knowledge distillation under real
> subgroup prevalence imbalance. Fair-teacher retraining improves the teacher materially,
> but only part of that gain transfers to the student through ordinary KD. A projected
> dual-ascent EO constraint remains a negative training-time result, while a jointly learned
> group calibration head is the first student-side intervention that substantially improves
> both raw and leakage-free calibrated fairness without sacrificing accuracy."

---

## Implementation Notes

### T1: Fair Teacher Retraining

- Teacher script: `scripts/time_llm/run_experiments.py` + `config_generator.py`
- Add `--fairness-feature gender --fairness-weight <w>` to teacher training
- Or: add WeightedRandomSampler to teacher DataLoader in `data_processing/data_sets.py`
- Teacher checkpoint dir: `pipeline_2025-10-28_14-20-17/phase_1_teacher/`
- Completed artifact: `pipeline_2025-10-28_14-20-17/phase_1_teacher/bert_all_patients_10epochs_fair_gender/`
- Comparable inference artifact: `pipeline_2025-10-28_14-20-17/phase_1_teacher/per_patient_inference_fair_teacher/`

### K1: Calibrated Soft Labels

- Completed script: `scripts/fairness/compute_k1_teacher_offsets.py`
- Distillation implementation: `distillation/core/distillation_trainer.py`
- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/`
- Outcome: negative result; K1 worsens RMSE and does not beat T1 on fairness

### O1: Projected Dual-Ascent KD Constraint

- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/`
- Student checkpoint: `logs/logs_2026-06-15_09-20-01/student_distilled.pth`
- Outcome: RMSE 23.141, EO_raw 0.2135, leakage-free EO_cal 0.0656
- Training note: the run completed stably, but the fairness term did not materially pull the student out of the raw-gap regime

### O2: Learned Calibration Head

- Completed artifact: `phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/`
- Student checkpoint: `logs/logs_2026-06-15_16-10-49/student_distilled.pth`
- Calibration sidecar: `logs/logs_2026-06-15_16-10-49/student_calibration_head.json`
- Outcome: RMSE 22.645, EO_raw 0.1189, leakage-free EO_cal 0.0286
- Training note: this is the strongest completed student result so far and is the main positive method result from the current roadmap

---

## Related Files

- `fairness/loss_functions/fairness_losses.py` — current fairness loss implementations
- `distillation/core/distillation_trainer.py` — KD training loop
- `distillation/core/distillation_wrapper.py` — wraps training, applies demographics
- `fairness/utils/analyzer_utils.py` — demographics + shared utilities
- `scripts/fairness/compute_fairness_comparison.py` — generates comparison table
- `scripts/pipelines/run_fairness_distillation_experiments.sh` — runs all 4 variants
