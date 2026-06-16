# Experiment Folder Registry

All completed experiments are under the canonical pipeline directory:

```
distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17/
```

---

## Phase 1 — Teacher Model

| Dir | Checkpoint | Notes |
|-----|-----------|-------|
| `phase_1_teacher/bert_all_patients_10epochs/logs/logs_2025-10-28_14-20-20/` | `checkpoints/checkpoint.pth` | Baseline BERT teacher, 10 epochs, seed=42 |
| `phase_1_teacher/bert_all_patients_10epochs_fair_gender/bert_all_patients_10epochs/logs/logs_2026-06-13_11-13-20/` | `checkpoints/checkpoint.pth` | Fair-sampling BERT teacher (T1), seed=42 |

Inference (per-patient, all 12 patients):
```
phase_1_teacher/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

Fair-teacher comparable inference (per-patient, all 12 patients):
```
phase_1_teacher/per_patient_inference_fair_teacher/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

- **Teacher baseline**: RMSE 24.162 | EO Gap raw 0.2302 | EO Gap calibrated (holdout) 0.0761
- **Teacher fair sampling (T1 teacher)**: RMSE 22.564 | EO Gap raw 0.1925 | EO Gap calibrated (holdout) 0.0478

---

## Phase 2 — Student Baseline (no KD)

| Dir | Checkpoint | Notes |
|-----|-----------|-------|
| `phase_2_student/` | (see logs/ subdir) | BERT-tiny trained standalone, seed=42 |

Inference:
```
phase_2_student/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

---

## Phase 3 — Distillation Runs

All use: BERT (teacher) → BERT-tiny (student), all 12 OhioT1DM patients, seed=42, lr=0.001, batch=32, alpha=0.5, beta=0.5, epochs=10.

### Run 1 — Baseline Distillation (no fairness)
```
phase_3_distillation/bert_to_bert-tiny_all_patients/
  logs/logs_2025-10-28_15-53-41/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference (placed at phase_3 root due to early pipeline version):
```
phase_3_distillation/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.361 | **EO Gap raw**: 0.2171 | **EO Gap calibrated (holdout)**: 0.0766

---

### Run 2 v1 — EqualizedOddsLoss (fairness regularisation, v1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender/
  logs/logs_2026-06-09_12-14-26/student_distilled.pth     ← CANONICAL CHECKPOINT (v1)
  logs/logs_2026-06-10_10-07-22/student_distilled.pth     ← v2 overwrite (see Run 2 v2)
```
Inference (uses v1 checkpoint `logs_2026-06-09_12-14-26`):
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.672 | **EO Gap raw**: 0.2194 | **EO Gap calibrated (holdout)**: 0.0629
- Loss: `EqualizedOddsLoss`, fairness_weight=0.3

---

### Run 2 v2 — HypoglycemiaTPREqualityLoss (focal+soft, fairness regularisation v2)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_v2/
  (no checkpoint kept — model weights removed post-inference)
```
Inference (uses checkpoint from fairness_gender `logs_2026-06-10_10-07-22`):
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_v2/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.426 | **EO Gap raw**: 0.2300 | **EO Gap calibrated (holdout)**: 0.0764
- Loss: `HypoglycemiaTPREqualityLoss` (focal γ=3, soft sigmoid slope=0.1), fairness_weight=0.3

---

### Run 3 — Oversampling Only (no fairness loss)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_oversample_gender/
  logs/logs_2026-06-12_12-49-05/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_oversample_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.918 | **EO Gap raw**: 0.2326 | **EO Gap calibrated (holdout)**: 0.0752
- WeightedRandomSampler, 2.4× female hypo windows, no fairness loss

---

### Run 4 — Oversampling + HypoglycemiaTPREqualityLoss (combined)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_oversample/
  logs/logs_2026-06-12_17-57-56/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_oversample/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 24.141 | **EO Gap raw**: 0.2225 | **EO Gap calibrated (holdout)**: 0.0650
- WeightedRandomSampler (2.4×) + HypoglycemiaTPREqualityLoss, fairness_weight=0.3

---

### Run 5 — Distilled from Fair Teacher (T1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.828 | **EO Gap raw**: 0.2089 | **EO Gap calibrated (holdout)**: 0.0579

### Run 6 — Distilled from Fair Teacher + O1 Constraint
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/
  logs/logs_2026-06-15_09-20-01/student_distilled.pth     ← CANONICAL CHECKPOINT
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 23.141 | **EO Gap raw**: 0.2135 | **EO Gap calibrated (holdout)**: 0.0656
- O1 projected dual-ascent fairness constraint on top of the T1 fair teacher; operationally successful, scientifically still negative on raw EO

### Run 7 — Distilled from Fair Teacher + O2 Calibration Head
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/
  logs/logs_2026-06-15_16-10-49/student_distilled.pth          ← CANONICAL CHECKPOINT
  logs/logs_2026-06-15_16-10-49/student_calibration_head.json  ← CANONICAL O2 SIDECAR
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 22.645 | **EO Gap raw**: 0.1189 | **EO Gap calibrated (holdout)**: 0.0286
- T1 fair teacher distilled with a learned per-gender affine calibration head; strongest completed student-side fairness result so far

### Run 8 — K1 Calibrated Soft Labels
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 25.677 | **EO Gap raw**: 0.2326 | **EO Gap calibrated (holdout)**: 0.0706

### Run 9 — Distilled from Fair Teacher + K1 Calibrated Soft Labels (T1+K1)
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender_fair_teacher/
```
Inference:
```
phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender_fair_teacher/per_patient_inference/
  time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```
- **RMSE**: 22.654 | **EO Gap raw**: 0.1981 | **EO Gap calibrated (holdout)**: 0.0719
- T1+K1 completeness check: recovers most of the T1 fairness gain but does not beat O2 and leaves the calibrated gap higher

---

## Comparison Results

```
pipeline_2025-10-28_14-20-17/fairness_comparison_results.csv   ← machine-readable
pipeline_2025-10-28_14-20-17/fairness_comparison_results.txt   ← human-readable
pipeline_2025-10-28_14-20-17/FAIRNESS_EXPERIMENTS_SUMMARY.md   ← analysis + findings
```

**Key finding:** Most student-side fairness interventions remain weak, but O2 is a clear exception. The fair teacher reaches RMSE 22.564 / EO_raw 0.1925, the distilled T1 student only partially inherits that gain at RMSE 23.828 / EO_raw 0.2089, and T1+O1 improves RMSE to 23.141 without escaping the critical raw-gap range at EO_raw 0.2135. By contrast, T1+O2 reaches RMSE 22.645 / EO_raw 0.1189 / EO_cal 0.0286, making it the strongest completed student result and the first method to move the student well out of the worst raw-gap regime. Leakage-free patient-holdout calibration still matters because it prevents the older, overly optimistic near-zero calibrated gaps.

---

## Remaining Method Experiments

See [FAIRNESS_SOLUTIONS_ROADMAP.md](FAIRNESS_SOLUTIONS_ROADMAP.md) for full list.

All committed roadmap phases (1–3) are complete. The methods below are
proposed future contributions only — none are run yet.

| ID | Description |
|----|-------------|
| T2 | Per-group teachers distilled into a shared student (multi-teacher KD) |
| T3 | Counterfactual augmentation of female hypo windows before teacher training |
| K2 | Group-conditional KD temperature |
| K3 | Fairness-aware feature alignment (MMD/CORAL on hidden states) |
| K4 | Selective KD replay on minority-group hypo windows |
| O3 | Adversarial group erasure (gradient-reversal discriminator) |

Highest-priority next step:
```bash
# Not another method run. Do a multi-seed robustness check on
# baseline KD, T1, and T1+O2 to confirm O2's gain is not seed noise.
# If one stronger follow-on method is wanted, prioritize T2 or K3.
```
