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
| K1 | [x] | **Calibrated soft labels** | Implemented calibrated soft-label KD using per-gender teacher offsets and evaluated it end to end. Result: worse RMSE in all 12 patients and no fairness improvement over T1. | Negative result. Useful as evidence that source-level label shifting alone is not enough. |
| K2 | [ ] | **Group-conditional KD temperature** | Use higher softmax temperature for male batches during KD. Higher temperature flattens overconfident male hypo predictions, reducing the information the student extracts from majority-group patterns. | Simple, interpretable. Easy to ablate temperature values. |
| K3 | [ ] | **Fairness-aware feature alignment** | Penalize divergence of intermediate hidden-state distributions across groups (not just output predictions). Add MMD or CORAL loss between male/female intermediate representations. Forces student to learn group-invariant features. | Works at representation level. Stronger than output-level constraints. |
| K4 | [ ] | **Selective KD replay** | Upweight the KD loss specifically on minority-group (female) hypoglycemic windows. Asymmetric distillation — student is pushed harder to match the teacher on rare events for the underrepresented group. | Targeted. Doesn't distort majority-group learning. |

---

## Intervention Point 3 — Fix the Student Objective

| ID | Status | Method | Description | Expected Impact |
| --- | --- | --- | --- | --- |
| O1 | [x] | **Projected dual-ascent EO constraint** | Implemented a constrained distillation objective on top of the fair teacher with an adaptive dual variable and EO-gap violation penalty. The run completed end to end, but the student still stayed in the critical raw-gap range. | Scientifically negative result: operationally valid, but not sufficient to close the fairness gap. |
| O2 | [ ] | **Learned calibration head** | Train a small per-group linear calibration layer jointly with distillation (`y_final = W_group * y_student + b_group`). Bakes Fix B into model weights — no inference-time lookup needed. Joint training ensures calibration doesn't hurt RMSE. | Inference-simple. Avoids post-hoc step. |
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

- [ ] **T1 + K1**: Fair teacher + calibrated soft labels → still unrun as a completeness check only
- [x] **T1 + O1**: Fair teacher + projected dual-ascent EO constraint → completed, but still critical on raw EO Gap
- [x] Compare completed Phase 2+3 runs on RMSE vs EO Gap tradeoff curve

### Current Combined Findings

- Teacher baseline: RMSE 24.162, EO_raw 0.2302, leakage-free EO_cal 0.0761
- Teacher fair sampling (T1 teacher): RMSE 22.564, EO_raw 0.1925, leakage-free EO_cal 0.0478
- Baseline KD student: RMSE 23.361, EO_raw 0.2171, leakage-free EO_cal 0.0766
- T1 distilled student: RMSE 23.828, EO_raw 0.2089, leakage-free EO_cal 0.0579
- T1 + O1 distilled student: RMSE 23.141, EO_raw 0.2135, leakage-free EO_cal 0.0656

Interpretation: T1 is effective at the teacher stage, but knowledge distillation does not preserve the full teacher fairness/accuracy gain. O1 improved RMSE relative to baseline KD and T1-only distillation, but it still failed to reduce the raw EO Gap below the critical range. The current evidence is that the fairness bottleneck is not only in teacher quality, but also in the transfer process and the underlying prevalence imbalance.

### Recommended Next Experiments

1. Run `O2` as the highest-value remaining method experiment if you want one more method contribution.
2. Treat `T1 + K1` as low priority unless you need a complete combinational grid for the paper.
3. Run a small multi-seed robustness check on baseline KD, T1, and T1+O1.

### Paper Narrative

> "Standard fairness regularization fails in clinical knowledge distillation under real
> subgroup prevalence imbalance. Fair-teacher retraining improves the teacher materially,
> but only part of that gain transfers to the student. Even a projected dual-ascent EO
> constraint remains a negative result at training time. Leakage-free post-hoc calibration
> remains effective as a deployment-time mitigation, but not as a trivial zero-gap solution."

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

---

## Related Files

- `fairness/loss_functions/fairness_losses.py` — current fairness loss implementations
- `distillation/core/distillation_trainer.py` — KD training loop
- `distillation/core/distillation_wrapper.py` — wraps training, applies demographics
- `fairness/utils/analyzer_utils.py` — demographics + shared utilities
- `scripts/fairness/compute_fairness_comparison.py` — generates comparison table
- `scripts/pipelines/run_fairness_distillation_experiments.sh` — runs all 4 variants
