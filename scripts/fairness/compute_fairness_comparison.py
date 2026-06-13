#!/usr/bin/env python3
"""
Compute full fairness comparison table across all distillation runs.

Produces:
  - fairness_comparison_results.csv  — machine-readable table
  - fairness_comparison_results.txt  — human-readable table for paper

Usage:
    python scripts/fairness/compute_fairness_comparison.py \
        --pipeline-dir distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import brentq

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ── Patient demographics — single source of truth from fairness module ───────
from fairness.utils.analyzer_utils import get_ohiot1dm_default_data as _get_demo
GENDER = {pid: d['gender'] for pid, d in _get_demo().items()}
HYPO_THRESHOLD = 70.0  # mg/dL


# ── Data loading ─────────────────────────────────────────────────────────────

def load_window_data(inference_dir: Path):
    """Load all per-patient inference_results_reformatted.csv into flat arrays."""
    all_preds, all_targets, all_groups = [], [], []
    for csv_path in sorted(inference_dir.rglob('inference_results_reformatted.csv')):
        patient_id = next(
            (p.replace('patient_', '') for p in csv_path.parts if p.startswith('patient_')),
            None
        )
        if not patient_id:
            continue
        gender = GENDER.get(patient_id, 'Unknown')
        df = pd.read_csv(csv_path)
        true_cols = [c for c in df.columns if c.endswith('_true')]
        pred_cols = [c for c in df.columns if c.endswith('_pred')]
        for _, row in df.iterrows():
            for tc, pc in zip(true_cols, pred_cols):
                all_targets.append(float(row[tc]))
                all_preds.append(float(row[pc]))
                all_groups.append(gender)
    return np.array(all_preds), np.array(all_targets), np.array(all_groups)


def load_patient_rmse(inference_dir: Path):
    """Load per-patient RMSE from experiment_results.csv."""
    csv = next(inference_dir.rglob('experiment_results.csv'), None)
    if csv is None:
        return None
    df = pd.read_csv(csv)
    df['patient_id'] = df['patient_id'].astype(str)
    df['gender'] = df['patient_id'].map(GENDER)
    return df


# ── Metrics ──────────────────────────────────────────────────────────────────

def tpr_at_threshold(preds, targets, thresh):
    hypo = targets < HYPO_THRESHOLD
    if hypo.sum() == 0:
        return 0.0
    return float((preds[hypo] < thresh).mean())


def eo_gap(preds, targets, groups, thresh_male=HYPO_THRESHOLD, thresh_female=HYPO_THRESHOLD):
    t = {'Male': thresh_male, 'Female': thresh_female}
    tprs = {g: tpr_at_threshold(preds[groups == g], targets[groups == g], t[g])
            for g in ['Male', 'Female']}
    return abs(tprs['Male'] - tprs['Female']), tprs


def calibrate_thresholds(preds, targets, groups):
    """Find per-gender thresholds that equalize TPR (Fix B)."""
    tpr_m = tpr_at_threshold(preds[groups == 'Male'],   targets[groups == 'Male'],   HYPO_THRESHOLD)
    tpr_f = tpr_at_threshold(preds[groups == 'Female'], targets[groups == 'Female'], HYPO_THRESHOLD)
    target_tpr = (tpr_m + tpr_f) / 2.0

    def find_t(g, lo=40, hi=200):
        pm, tm = preds[groups == g], targets[groups == g]
        try:
            return brentq(lambda t: tpr_at_threshold(pm, tm, t) - target_tpr, lo, hi)
        except ValueError:
            return HYPO_THRESHOLD

    t_male   = find_t('Male')
    t_female = find_t('Female')
    gap_cal, tprs_cal = eo_gap(preds, targets, groups, t_male, t_female)
    return t_male, t_female, gap_cal, tprs_cal


def assess(eo):
    if eo < 0.05:  return "✅ EXCELLENT"
    if eo < 0.10:  return "✅ GOOD"
    if eo < 0.15:  return "⚠️ MODERATE"
    if eo < 0.20:  return "⚠️ CONCERNING"
    return "❌ CRITICAL"


# ── Run analysis for one model ────────────────────────────────────────────────

def analyze_run(label, inference_dir):
    print(f"\n  Analyzing: {label}")
    inference_dir = Path(inference_dir)
    if not inference_dir.exists():
        print(f"    ⚠️  Directory not found: {inference_dir}")
        return None

    preds, targets, groups = load_window_data(inference_dir)
    if len(preds) == 0:
        print(f"    ⚠️  No window data found in {inference_dir}")
        return None

    overall_rmse = float(np.sqrt(np.mean((preds - targets) ** 2)))
    overall_mae  = float(np.mean(np.abs(preds - targets)))

    # Per-gender RMSE
    rmse_male   = float(np.sqrt(np.mean((preds[groups=='Male']   - targets[groups=='Male'])   ** 2)))
    rmse_female = float(np.sqrt(np.mean((preds[groups=='Female'] - targets[groups=='Female']) ** 2)))

    # Raw EO Gap
    gap_raw, tprs_raw = eo_gap(preds, targets, groups)

    # Calibrated EO Gap (Fix B)
    t_m, t_f, gap_cal, tprs_cal = calibrate_thresholds(preds, targets, groups)

    result = {
        'label':             label,
        'rmse':              round(overall_rmse, 3),
        'mae':               round(overall_mae,  3),
        'rmse_male':         round(rmse_male,   3),
        'rmse_female':       round(rmse_female, 3),
        'tpr_male_raw':      round(tprs_raw['Male'],   4),
        'tpr_female_raw':    round(tprs_raw['Female'], 4),
        'eo_gap_raw':        round(gap_raw, 4),
        'assess_raw':        assess(gap_raw),
        'thresh_male_cal':   round(t_m, 1),
        'thresh_female_cal': round(t_f, 1),
        'tpr_male_cal':      round(tprs_cal['Male'],   4),
        'tpr_female_cal':    round(tprs_cal['Female'], 4),
        'eo_gap_calibrated': round(gap_cal, 4),
        'assess_cal':        assess(gap_cal),
    }
    print(f"    RMSE={overall_rmse:.3f}  EO_raw={gap_raw:.4f}  EO_cal={gap_cal:.4f}")
    return result


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute fairness comparison table")
    parser.add_argument("--pipeline-dir", required=True,
                        help="Path to pipeline directory (e.g. distillation_experiments/all_patients_pipeline/pipeline_2025-10-28_14-20-17)")
    args = parser.parse_args()

    pipeline = Path(args.pipeline_dir)
    phase2   = pipeline / "phase_2_student"
    phase3   = pipeline / "phase_3_distillation"
    p2_inf   = phase2 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"

    # All runs to compare
    runs = [
        ("Student baseline (no KD)",
         p2_inf),
        ("Distilled — no fairness",
         phase3 / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + HypoglycemiaTPR loss (v1, EqualizedOdds)",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + HypoglycemiaTPR loss (v2, focal+soft)",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender_v2" / "per_patient_inference"),
        ("Distilled + Oversampling only",
         phase3 / "bert_to_bert-tiny_all_patients_oversample_gender" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled + Oversampling + HypoglycemiaTPR loss",
         phase3 / "bert_to_bert-tiny_all_patients_fairness_gender_oversample" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
        ("Distilled from Fair Teacher (T1)",
         phase3 / "bert_to_bert-tiny_all_patients_fair_teacher" / "per_patient_inference" / "time_llm_per_patient_inference_ohiot1dm"),
    ]

    print("\n" + "="*70)
    print("FAIRNESS COMPARISON TABLE")
    print("="*70)

    results = []
    for label, inf_dir in runs:
        r = analyze_run(label, inf_dir)
        if r:
            results.append(r)

    if not results:
        print("❌ No results found!")
        return

    df = pd.DataFrame(results)

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = pipeline / "fairness_comparison_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✅ CSV saved: {csv_path}")

    # ── Print human-readable table ────────────────────────────────────────────
    txt_path = pipeline / "fairness_comparison_results.txt"
    lines = []
    lines.append("FAIRNESS COMPARISON — Gender EO Gap (Hypoglycemia TPR Disparity)")
    lines.append("All-Patients Distillation Pipeline  |  OhioT1DM  |  BERT→BERT-tiny")
    lines.append("="*90)
    header = f"{'Model':<48} {'RMSE':>6}  {'EO_raw':>8}  {'EO_cal':>8}  {'Assessment (raw)'}"
    lines.append(header)
    lines.append("-"*90)
    for r in results:
        line = (f"{r['label']:<48} {r['rmse']:6.3f}  {r['eo_gap_raw']:8.4f}  "
                f"{r['eo_gap_calibrated']:8.4f}  {r['assess_raw']}")
        lines.append(line)
    lines.append("-"*90)
    lines.append("\nNotes:")
    lines.append("  EO_raw = EO Gap with uniform threshold=70 mg/dL for all groups")
    lines.append("  EO_cal = EO Gap after per-gender threshold calibration (Fix B)")
    lines.append("  EO Gap = |Male_TPR - Female_TPR| for hypoglycemia detection")
    lines.append("  Lower EO Gap = fairer model")
    lines.append("")
    lines.append("Per-gender thresholds after calibration:")
    for r in results:
        lines.append(f"  {r['label'][:45]:<45}  Male thresh={r['thresh_male_cal']:5.1f}  Female thresh={r['thresh_female_cal']:5.1f}")

    txt = "\n".join(lines)
    with open(txt_path, "w") as f:
        f.write(txt)

    print("\n" + txt)
    print(f"\n✅ Text table saved: {txt_path}")


if __name__ == "__main__":
    main()
