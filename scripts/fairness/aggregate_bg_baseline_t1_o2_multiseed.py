#!/usr/bin/env python3
"""Aggregate the focused BG Teacher/Student/KD/T1/O2 five-seed suite.

Expected layout, produced by run_bg_baseline_t1_o2_multiseed.sh:

  <suite-dir>/seed_<seed>/
    phase_1_teacher/
    phase_1_teacher_fair/
    phase_2_student/
    phase_3_distillation/<method-run-name>/

Every stage contains per_patient_inference/
time_llm_per_patient_inference_ohiot1dm with window-level reformatted results.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts" / "utilities"))

from scripts.fairness.compute_fairness_comparison import analyze_run  # noqa: E402


INFERENCE_SUBPATH = Path("per_patient_inference") / "time_llm_per_patient_inference_ohiot1dm"
METHODS = [
    ("Teacher baseline (BERT)", Path("phase_1_teacher")),
    ("Teacher fair sampling (T1 source)", Path("phase_1_teacher_fair")),
    ("Student baseline (no KD)", Path("phase_2_student")),
    ("Baseline KD (no fairness)", Path("phase_3_distillation/bert_to_bert-tiny_all_patients_seed{seed}")),
    ("Distilled from Fair Teacher (T1)", Path("phase_3_distillation/bert_to_bert-tiny_all_patients_fair_teacher_seed{seed}")),
    ("O2-only Calibration Head", Path("phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_seed{seed}")),
    ("Distilled from Fair Teacher + O2 Calibration Head", Path("phase_3_distillation/bert_to_bert-tiny_all_patients_o2_gender_fair_teacher_seed{seed}")),
]


def load_fixed_seeds():
    from seeds import fixed_seeds

    return list(fixed_seeds)


def format_metric(mean, std):
    return f"{mean:.3f} ± {std:.3f}"


def aggregate_metric(values):
    return statistics.mean(values), statistics.pstdev(values) if len(values) > 1 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate the focused OhioT1DM Teacher/Student/KD/T1/O2 multi-seed suite."
    )
    parser.add_argument("--suite-dir", required=True, help="Focused suite root directory")
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds; defaults to scripts/utilities/seeds.py fixed_seeds",
    )
    parser.add_argument("--calibration-mode", choices=["patient-holdout", "in-sample"], default="patient-holdout")
    parser.add_argument("--calibration-folds", type=int, default=2)
    args = parser.parse_args()

    seeds = [int(seed) for seed in args.seeds.split(",")] if args.seeds else load_fixed_seeds()
    suite_dir = Path(args.suite_dir)

    print("=" * 88)
    print("FOCUSED BG MULTI-SEED — Teacher / Student / KD / T1 / O2")
    print(f"Suite: {suite_dir}")
    print(f"Seeds ({len(seeds)}): {seeds}")
    print(f"Calibration: {args.calibration_mode} (folds={args.calibration_folds})")
    print("=" * 88)

    summary_rows = []
    for label, relative_stage in METHODS:
        per_seed = {"rmse": [], "eo_raw": [], "eo_cal": []}
        used_seeds = []
        for seed in seeds:
            stage = Path(str(relative_stage).format(seed=seed))
            inference_dir = suite_dir / f"seed_{seed}" / stage / INFERENCE_SUBPATH
            result = analyze_run(
                f"{label} [seed {seed}]",
                inference_dir,
                args.calibration_mode,
                args.calibration_folds,
            )
            if result is None:
                continue
            per_seed["rmse"].append(result["rmse"])
            per_seed["eo_raw"].append(result["eo_gap_raw"])
            per_seed["eo_cal"].append(result["eo_gap_calibrated"])
            used_seeds.append(seed)

        if not used_seeds:
            print(f"\n  {label}: no complete inference found")
            continue

        rmse_mean, rmse_std = aggregate_metric(per_seed["rmse"])
        eo_raw_mean, eo_raw_std = aggregate_metric(per_seed["eo_raw"])
        eo_cal_mean, eo_cal_std = aggregate_metric(per_seed["eo_cal"])
        summary_rows.append(
            {
                "method": label,
                "n_seeds": len(used_seeds),
                "seeds": used_seeds,
                "rmse_mean": rmse_mean,
                "rmse_std": rmse_std,
                "eo_raw_mean": eo_raw_mean,
                "eo_raw_std": eo_raw_std,
                "eo_cal_mean": eo_cal_mean,
                "eo_cal_std": eo_cal_std,
                "eo_raw_per_seed": per_seed["eo_raw"],
            }
        )
        print(f"\n  {label} (n={len(used_seeds)}; seeds={used_seeds})")
        print(f"    RMSE   = {format_metric(rmse_mean, rmse_std)}")
        print(f"    EO_raw = {format_metric(eo_raw_mean, eo_raw_std)}")
        print(f"    EO_cal = {format_metric(eo_cal_mean, eo_cal_std)}")

    text_output = suite_dir / "baseline_t1_o2_multiseed_results.txt"
    csv_output = suite_dir / "baseline_t1_o2_multiseed_results.csv"
    suite_dir.mkdir(parents=True, exist_ok=True)

    header = f"{'Method':<54}{'Seeds':>8}{'RMSE':>16}{'EO_raw':>16}{'EO_cal':>16}"
    lines = [
        f"Focused BG multi-seed results — seeds={seeds}, calibration={args.calibration_mode}/{args.calibration_folds}",
        "",
        header,
        "-" * len(header),
    ]
    for row in summary_rows:
        lines.append(
            f"{row['method'][:53]:<54}{row['n_seeds']:>8}{format_metric(row['rmse_mean'], row['rmse_std']):>16}"
            f"{format_metric(row['eo_raw_mean'], row['eo_raw_std']):>16}"
            f"{format_metric(row['eo_cal_mean'], row['eo_cal_std']):>16}"
        )
        lines.append(
            f"    seeds={row['seeds']} | EO_raw per seed={[round(value, 4) for value in row['eo_raw_per_seed']]}"
        )
    text_output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with csv_output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "method",
                "n_seeds",
                "seeds",
                "rmse_mean",
                "rmse_std",
                "eo_raw_mean",
                "eo_raw_std",
                "eo_cal_mean",
                "eo_cal_std",
                "eo_raw_per_seed",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "method": row["method"],
                    "n_seeds": row["n_seeds"],
                    "seeds": ";".join(map(str, row["seeds"])),
                    "rmse_mean": f"{row['rmse_mean']:.6f}",
                    "rmse_std": f"{row['rmse_std']:.6f}",
                    "eo_raw_mean": f"{row['eo_raw_mean']:.6f}",
                    "eo_raw_std": f"{row['eo_raw_std']:.6f}",
                    "eo_cal_mean": f"{row['eo_cal_mean']:.6f}",
                    "eo_cal_std": f"{row['eo_cal_std']:.6f}",
                    "eo_raw_per_seed": ";".join(f"{value:.6f}" for value in row["eo_raw_per_seed"]),
                }
            )

    print("\n" + "=" * 88)
    print(header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['method'][:53]:<54}{row['n_seeds']:>8}{format_metric(row['rmse_mean'], row['rmse_std']):>16}"
            f"{format_metric(row['eo_raw_mean'], row['eo_raw_std']):>16}"
            f"{format_metric(row['eo_cal_mean'], row['eo_cal_std']):>16}"
        )
    print("=" * 88)
    print(f"Saved: {text_output}")
    print(f"Saved: {csv_output}")


if __name__ == "__main__":
    main()