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
|----|--------|--------|-------------|-----------------|
| T1 | [ ] | **Fair teacher retraining** | Retrain teacher (BERT) with group-balanced sampling or a fairness loss on the teacher's training objective. If the teacher is fair, students inherit fairness via KD with no other changes. | Reduces raw EO Gap at the source. Clean ablation to isolate where bias enters. |
| T2 | [ ] | **Per-group teachers** | Train one teacher per gender group, then distill both into a single shared student via multi-teacher KD. Each teacher specializes on its group's hypo pattern. | Eliminates prevalence mismatch in soft labels entirely. More compute-heavy. |
| T3 | [ ] | **Counterfactual data augmentation** | Synthesize additional female hypoglycemia windows (e.g., via interpolation or GAN) to match male hypo prevalence before teacher training. Addresses root cause directly. | Most principled fix. Requires validation that synthetic data is realistic. |

---

## Intervention Point 2 — Fix the Knowledge Transfer

| ID | Status | Method | Description | Expected Impact |
|----|--------|--------|-------------|-----------------|
| K1 | [ ] | **Calibrated soft labels** | Before distillation: run teacher on training set, compute per-gender threshold offsets that equalize TPR (same as post-hoc Fix B but applied to teacher outputs). During KD loss, shift teacher logits by this offset per patient group. Student learns from a "fair teacher". | Novel contribution. Targets the label-level source of bias. No change to loss objective needed. |
| K2 | [ ] | **Group-conditional KD temperature** | Use higher softmax temperature for male batches during KD. Higher temperature flattens overconfident male hypo predictions, reducing the information the student extracts from majority-group patterns. | Simple, interpretable. Easy to ablate temperature values. |
| K3 | [ ] | **Fairness-aware feature alignment** | Penalize divergence of intermediate hidden-state distributions across groups (not just output predictions). Add MMD or CORAL loss between male/female intermediate representations. Forces student to learn group-invariant features. | Works at representation level. Stronger than output-level constraints. |
| K4 | [ ] | **Selective KD replay** | Upweight the KD loss specifically on minority-group (female) hypoglycemic windows. Asymmetric distillation — student is pushed harder to match the teacher on rare events for the underrepresented group. | Targeted. Doesn't distort majority-group learning. |

---

## Intervention Point 3 — Fix the Student Objective

| ID | Status | Method | Description | Expected Impact |
|----|--------|--------|-------------|-----------------|
| O1 | [ ] | **Lagrangian constrained KD** | Replace the soft additive fairness penalty (`total = kd + alpha*gt + beta*fair`) with a hard Lagrangian constraint: `min KD+GT s.t. EO_Gap <= epsilon`. Use an adaptive multiplier (dual variable) that increases when constraint is violated. Solves the gradient competition problem. | Theoretically grounded. Principled solution to loss competition. Requires constrained optimizer (e.g. `cooper` library). |
| O2 | [ ] | **Learned calibration head** | Train a small per-group linear calibration layer jointly with distillation (`y_final = W_group * y_student + b_group`). Bakes Fix B into model weights — no inference-time lookup needed. Joint training ensures calibration doesn't hurt RMSE. | Inference-simple. Avoids post-hoc step. |
| O3 | [ ] | **Adversarial group erasure** | Add a gradient-reversal discriminator head that tries to predict gender from the student's hidden states. Penalize the student for being gender-predictable at the representation level. | Prevents encoding group information entirely. Strong fairness guarantee. Risk of underfitting. |

---

## Recommended Experiment Order for Paper

### Phase 1 (Diagnosis) — DONE
- [x] Run baseline + 4 fairness variants
- [x] Show all training-time interventions fail (EO Gap unchanged)
- [x] Show Fix B (post-hoc calibration) works trivially for all models

### Phase 2 (Source Fix)
- [ ] **T1**: Retrain fair teacher → show student inherits fairness
- [ ] **K1**: Calibrated soft labels → compare to T1, show similar effect with less compute

### Phase 3 (Best Combined)
- [ ] **T1 + K1**: Fair teacher + calibrated soft labels → lower bound on achievable EO Gap
- [ ] **T1 + O1**: Fair teacher + Lagrangian constraint → principled training-time guarantee
- [ ] Compare all Phase 2+3 runs on RMSE vs EO Gap tradeoff curve

### Paper Narrative
> "Standard fairness regularization fails in clinical knowledge distillation due to 
> clinically irreducible disease prevalence imbalance. We show that the bias is introduced 
> at the teacher level and propose two source-level fixes — fair teacher retraining (T1) 
> and calibrated soft label transfer (K1) — that reduce EO Gap from 0.22 to <0.05 without 
> post-hoc calibration, while maintaining competitive RMSE."

---

## Implementation Notes

### T1: Fair Teacher Retraining
- Teacher script: `scripts/time_llm/run_experiments.py` + `config_generator.py`
- Add `--fairness-feature gender --fairness-weight <w>` to teacher training
- Or: add WeightedRandomSampler to teacher DataLoader in `data_processing/data_sets.py`
- Teacher checkpoint dir: `pipeline_2025-10-28_14-20-17/phase_1_teacher/`
- Save new fair teacher to: `pipeline_2025-10-28_14-20-17/phase_1_teacher_fair/`

### K1: Calibrated Soft Labels
- Pre-distillation step: run `scripts/fairness/compute_teacher_calibration.py` (to create)
  - Loads teacher checkpoint, runs on training set, finds per-gender threshold offsets
  - Saves offsets to `phase_1_teacher/calibration_offsets.json`
- Distillation: modify `distillation/core/distillation_trainer.py`
  - In KD loss computation, apply `teacher_logits += offset[group]` before KL divergence
  - Offset = scalar shift to teacher's continuous prediction to equalize soft-TPR

### O1: Lagrangian Constrained KD
- Library: `pip install cooper` (PyTorch constrained optimization)
- Modify `distillation_trainer.py` to use `cooper.ConstrainedMinimizationProblem`
- Constraint: `EO_Gap(batch) <= 0.05`
- Dual variable updated each step: increases pressure when constraint violated

---

## Related Files
- `fairness/loss_functions/fairness_losses.py` — current fairness loss implementations
- `distillation/core/distillation_trainer.py` — KD training loop
- `distillation/core/distillation_wrapper.py` — wraps training, applies demographics
- `fairness/utils/analyzer_utils.py` — demographics + shared utilities
- `scripts/fairness/compute_fairness_comparison.py` — generates comparison table
- `scripts/pipelines/run_fairness_distillation_experiments.sh` — runs all 4 variants
