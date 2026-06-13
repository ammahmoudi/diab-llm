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

Inference (per-patient, all 12 patients):
```
phase_1_teacher/per_patient_inference/time_llm_per_patient_inference_ohiot1dm/experiment_results.csv
```

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
- **RMSE**: 23.361 | **EO Gap raw**: 0.2171 | **EO Gap calibrated**: 0.0001

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
- **RMSE**: 23.672 | **EO Gap raw**: 0.2194 | **EO Gap calibrated**: 0.0000
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
- **RMSE**: 24.426 | **EO Gap raw**: 0.2300 | **EO Gap calibrated**: 0.0001
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
- **RMSE**: 24.918 | **EO Gap raw**: 0.2326 | **EO Gap calibrated**: 0.0001
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
- **RMSE**: 24.141 | **EO Gap raw**: 0.2225 | **EO Gap calibrated**: 0.0001
- WeightedRandomSampler (2.4×) + HypoglycemiaTPREqualityLoss, fairness_weight=0.3

---

## Comparison Results

```
pipeline_2025-10-28_14-20-17/fairness_comparison_results.csv   ← machine-readable
pipeline_2025-10-28_14-20-17/fairness_comparison_results.txt   ← human-readable
pipeline_2025-10-28_14-20-17/FAIRNESS_EXPERIMENTS_SUMMARY.md   ← analysis + findings
```

**Key finding:** All training-time fairness interventions fail. EO Gap stays 0.19–0.23 regardless of loss or sampling. Post-hoc threshold calibration (Fix B) reduces EO Gap to ~0.0001 for all models.

---

## Planned Experiments (not yet run)

See [FAIRNESS_SOLUTIONS_ROADMAP.md](FAIRNESS_SOLUTIONS_ROADMAP.md) for full list.

| ID | Dir (when run) | Description |
|----|---------------|-------------|
| T1 | `bert_to_bert-tiny_all_patients_fair_teacher/` | Distill from fair teacher (hypo-oversampled teacher training) |
| K1 | `bert_to_bert-tiny_all_patients_calibrated_labels/` | Calibrated soft labels (shift teacher logits by gender offset before KD) |
| O1 | `bert_to_bert-tiny_all_patients_lagrangian/` | Lagrangian constrained KD (hard EO constraint with dual variable) |

Run T1 with:
```bash
bash scripts/pipelines/run_fair_teacher_experiment.sh
```
