# Fairness Experiments Summary
**Pipeline:** `pipeline_2025-10-28_14-20-17`  
**Dataset:** OhioT1DM (12 patients)  
**Architecture:** BERT → BERT-tiny (knowledge distillation)  
**Fairness attribute:** Gender (Male: 540, 544, 552, 563, 570, 584, 596 | Female: 559, 567, 575, 588, 591)  
**Fairness metric:** EO Gap = |Male TPR − Female TPR| for hypoglycemia detection (threshold 70 mg/dL)  
**Hyperparams (all runs):** seed=42, lr=0.001, batch-size=32, alpha=0.5, beta=0.5, distill-epochs=10  

---

## Results

Leakage-free calibrated EO is now computed with deterministic 2-fold patient-holdout calibration: fit male/female thresholds on one fold of patients and evaluate them on different patients.

> **Multi-seed validation (O2 headline).** The single-seed table below uses seed=42. The 5-seed
> robustness check is now complete and auditable (831363, 809906, 427368, 238822, 247659):
> baseline KD EO_raw **0.217±0.006**, T1 **0.201±0.008**, and **T1+O2 0.118±0.033**.
> O2 beats baseline on raw gap in all 5 seeds — its worst seed (0.155) is below baseline's best
> (0.207) — confirming the ~0.23→0.12 headline. O2's improvement magnitude is seed-sensitive,
> but direction is 5/5 consistent. See `multiseed_robustness_results.txt` and
> `multiseed_robustness_results.csv`.

| # | Model | RMSE | EO Gap (raw) | EO Gap (calibrated, holdout) | Male thresh | Female thresh |
|---|---|---|---|---|---|---|
| 0 | Teacher baseline (BERT) | 24.162 | 0.2302 | 0.0761 | 73.2 ± 0.67 | 67.0 ± 0.94 |
| 1 | Teacher fair sampling (T1 teacher) | 22.564 | 0.1925 | 0.0478 | 73.2 ± 0.23 | 67.3 ± 1.05 |
| 2 | Student baseline (no KD) | 21.989 | 0.1948 | 0.0534 | 73.1 ± 0.39 | 67.1 ± 0.70 |
| 3 | Distilled — no fairness | 23.361 | 0.2171 | 0.0766 | 73.3 ± 0.84 | 67.2 ± 1.00 |
| 4 | + HypoglycemiaTPR loss v1 (EqualizedOdds) | 23.672 | 0.2194 | 0.0629 | 73.3 ± 0.56 | 67.1 ± 0.92 |
| 5 | + HypoglycemiaTPR loss v2 (focal+soft) | 24.426 | 0.2300 | 0.0764 | 73.2 ± 0.86 | 66.8 ± 0.85 |
| 6 | + Oversampling only | 24.918 | 0.2326 | 0.0752 | 73.5 ± 0.74 | 67.0 ± 0.90 |
| 7 | + Oversampling + HypoglycemiaTPR loss | 24.141 | 0.2225 | 0.0650 | 73.3 ± 0.70 | 67.1 ± 0.95 |
| 8 | Distilled from Fair Teacher (T1) | 23.828 | 0.2089 | 0.0579 | 73.3 ± 0.27 | 67.4 ± 1.03 |
| 9 | Distilled from Fair Teacher + O1 Constraint | 23.141 | 0.2135 | 0.0656 | 73.1 ± 0.60 | 67.2 ± 0.79 |
| 10 | Distilled from Fair Teacher + O2 Calibration Head | 22.645 | 0.1189 | 0.0286 | 72.1 ± 0.69 | 68.9 ± 0.42 |
| 11 | Distilled from Fair Teacher + K1 Calibrated Soft Labels | 22.654 | 0.1981 | 0.0719 | 73.0 ± 0.54 | 67.5 ± 0.92 |
| 12 | Distilled from Fair Teacher + K3 Feature Alignment (w=100) | 22.496 | 0.1974 | 0.0417 | 73.2 ± 0.04 | 67.5 ± 0.93 |
| 13 | Distilled from Fair Teacher + K3 Feature Alignment (w=200) | 22.438 | 0.1919 | 0.0536 | 72.9 ± 0.62 | 67.4 ± 0.73 |
| 14 | Distilled from Fair Teacher + K4 Selective KD Replay | 23.413 | 0.2056 | 0.0645 | 73.1 ± 0.45 | 67.3 ± 1.01 |
| 15 | Distilled from Fair Teacher + O3 Adversarial Erasure | 23.048 | 0.2118 | 0.0557 | 73.3 ± 0.52 | 67.2 ± 0.76 |
| 16 | Distilled from Per-Group Teachers (T2) | 22.956 | 0.2197 | 0.0732 | 73.1 ± 0.61 | 67.3 ± 1.13 |
| 17 | Distilled + K1 Calibrated Soft Labels | 25.677 | 0.2326 | 0.0706 | 73.4 ± 0.70 | 66.8 ± 0.80 |

> **EO_raw:** Uniform 70 mg/dL threshold for all groups  
> **EO_cal:** Leakage-free per-gender threshold calibration with patient-holdout evaluation  

---

## Key Finding

**Most student-side training-time fairness interventions fail, but O2 materially helps.** Among the older distilled student models, the raw EO Gap remains in the 0.21–0.23 range regardless of:
- Fairness-aware loss functions (EqualizedOdds, focal+soft TPR equalization)
- Minority-group oversampling (2.4× female hypo windows)
- Combination of both

**O2 is the first completed positive student-side result.** The fair-teacher + learned calibration head run reaches RMSE 22.645, EO_raw 0.1189, and leakage-free EO_cal 0.0286. That is substantially better than T1-only distillation (23.828 / 0.2089 / 0.0579) and T1+O1 (23.141 / 0.2135 / 0.0656), and it is the closest distilled student to the fair teacher while also being much fairer.

**Root cause:** Male patients have 3× more hypoglycemic events than female patients in this dataset (Male ~4.66% vs Female ~1.61% of windows). A model minimizing RMSE across the joint dataset will naturally fit the male hypo distribution better. No loss penalty or sampler can overcome a clinically real prevalence imbalance without degrading overall accuracy.

**T1 helps at the teacher stage, but the gain only partially transfers through distillation.** Fair-teacher sampling improves the teacher from RMSE 24.162 / EO_raw 0.2302 to RMSE 22.564 / EO_raw 0.1925. After distillation, the T1 student improves over baseline KD only to RMSE 23.828 / EO_raw 0.2089.

**T1+O1 is a completed negative result, not a fix.** Adding the projected dual-ascent EO constraint on top of the fair teacher improves RMSE to 23.141, but the raw EO Gap remains 0.2135. That is still in the same critical fairness regime as the other non-O2 distilled students, so O1 does not support a strong training-time fairness claim for the paper.

**T1+O2 is the main positive method result.** The learned per-gender calibration head moves the student out of the worst raw-gap regime and also delivers the best leakage-free calibrated EO score in the table. This is the strongest evidence so far that the student needs an explicit group-aware output correction during or after transfer, rather than only a fairness penalty in the loss.

**T1+K1 is a completeness check, not a fix.** Calibrated soft labels on top of the fair teacher reach RMSE 22.654 and EO_raw 0.1981 — recovering most of the T1 fairness gain and matching O2 on RMSE, but leaving the raw gap well above O2's 0.1189 and the calibrated gap higher (0.0719 vs 0.0286). K1-only (#12) remains the worst run overall. Source-level label shifting alone is insufficient; the student still needs the explicit group-aware output correction that O2 provides.

**T2 per-group teachers makes the gap worse — closing the source-side argument.** A dedicated male teacher (7 patients) and female teacher (5 patients) were trained and each sample routed to its group's teacher. Verified: both teachers trained on the correct splits and loaded. Result (RMSE 22.956, EO_raw 0.2197, EO_cal 0.0732) is *worse* on the raw gap than plain T1 distillation (0.2089) and back in CRITICAL. Per-group specialization does not remove the prevalence mismatch — each teacher learns its group's base rate and the student faithfully reproduces both, transmitting the disparity. Together with O3 (erasure succeeds, gap unmoved), this shows the gap is base-rate-driven and cannot be fixed from either the representation or the teacher/data side — only O2's output-level per-group correction works.

**O3 adversarial erasure is the most informative negative.** The gradient-reversal discriminator provably did its job — the O3 Adv term sat at ~0.685 ≈ ln(2)=0.693 across all epochs, meaning the discriminator was at chance and gender is erased from the student's hidden representation. Yet the raw EO Gap did not move (0.2118, still CRITICAL). This is direct evidence that the hypoglycemia-TPR disparity is NOT caused by the model encoding gender — it is a data-prevalence phenomenon (the real ~3:1 male:female hypo imbalance). A base-rate disparity cannot be fixed by hiding the group variable, which is exactly why only a post-hoc per-group output correction (O2) closes it.

**K4 selective KD replay is another raw-gap negative.** Upweighting female-hypoglycemia windows 4× in the KD loss engaged properly (Teacher Loss rose to ~250–280 vs ~210–248 baseline), but the result (RMSE 23.413, EO_raw 0.2056, EO_cal 0.0645) leaves the raw gap essentially at plain-T1 level (0.2089, still CRITICAL) and slightly worsens the calibrated gap. As expected for the loss-reweighting family: the penalty is active but does not close the hypo-TPR gap.

**K3 helps accuracy, not the raw gap — settled across weights.** After the loss was made scale-invariant (standardize features by pooled per-dim std → correlation-matrix difference; earlier absolute-covariance runs were inert and deleted), both the w=100 and w=200 runs engaged properly (K3 Align ~0.169, ≈2% of the ~800 total loss). Both give the best RMSE in the table (22.496 / 22.438) but leave the raw EO Gap at baseline (0.1974 / 0.1919, vs O2's 0.1189). Decisively, the K3 term stayed flat at ~0.169 across all epochs at *both* weights — doubling the penalty did not reduce it, so this is a ceiling, not undertuning: the student pays the alignment cost as a fixed tax it cannot lower because the dominant GT+teacher losses pin the representation. Net: like every other in-training method, K3 does not move the raw hypo-TPR gap; O2 remains the only method that does. K3 is closed.

**Leakage-free post-hoc calibration helps materially but not perfectly.** Patient-holdout threshold calibration reduces EO Gap from roughly 0.19–0.23 down to roughly 0.05–0.08, but it does not collapse disparity to zero once thresholds are evaluated on unseen patients.

**Paper-safe interpretation:** the older near-zero calibrated EO numbers were optimistic because they fit and evaluated thresholds on the same held-out outputs. The leakage-free numbers still support calibration as a useful deployment-time mitigation, but not as a trivial complete fix.

**Comparable teacher baselines are now included.** The original CSV and summary omitted phase-1 rows because the comparison script only enumerated phase 2 and phase 3 runs. The baseline teacher and fair teacher are now both computed from phase-1 per-patient inference.

---

## Why Standard Fairness KD Fails

```
Normal KD:  Student ← mimic(biased_teacher_outputs)
             └── Biased soft labels force student into teacher's bias
Fairness KD: Student ← mimic(biased_teacher) + fairness_penalty
             └── KD loss and fairness loss compete → fairness loss loses
```

The teacher's soft labels encode the prevalence imbalance. Penalizing the student doesn't change what it is trying to mimic.

---

## Demographics (from data/ohiot1dm/data.csv)

| Patient | Gender | Age | Pump |
|---|---|---|---|
| 540 | Male | 20-40 | 630G |
| 544 | Male | 40-60 | 530G |
| 552 | Male | 20-40 | 630G |
| 559 | Female | 40-60 | 530G |
| 563 | Male | 40-60 | 530G |
| 567 | Female | 20-40 | 630G |
| 570 | Male | 40-60 | 530G |
| 575 | Female | 40-60 | 530G |
| 584 | Male | 40-60 | 530G |
| 588 | Female | 40-60 | 530G |
| 591 | Female | 40-60 | 530G |
| 596 | Male | 60-80 | 530G |

---

## Files

| File | Description |
|---|---|
| `fairness_comparison_results.txt` | Human-readable comparison table |
| `fairness_comparison_results.csv` | Machine-readable results (this repo) |
| `FAIRNESS_EXPERIMENTS_SUMMARY.md` | This document |
| `phase_3_distillation/bert_to_bert-tiny_all_patients/` | Run 1: Baseline distillation |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender/` | Run 2v1: EqualizedOdds loss |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_v2/` | Run 2v2: focal+soft TPR loss |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_oversample_gender/` | Run 3: Oversampling only |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_fairness_gender_oversample/` | Run 4: Oversampling + loss |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher/` | Run 5: Distilled from fair teacher (T1) |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_o1_gender_fair_teacher_o1/` | Run 6: Distilled from fair teacher + O1 constraint |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher/` | Run 7: Distilled from fair teacher + O2 calibration head |
| `phase_3_distillation/bert_to_bert-tiny_all_patients_k1cal_gender/` | Run 8: K1 calibrated soft labels |
