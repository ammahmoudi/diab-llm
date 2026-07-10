#!/usr/bin/env python3
"""Run fairness analysis for MIT-BIH ECG classifier predictions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fairness.analyzers.ecg_classifier_fairness_analyzer import ECGClassifierFairnessAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MIT-BIH ECG classifier fairness analysis")
    parser.add_argument("--prediction-csv", required=True, type=Path)
    parser.add_argument("--group-column", default="sex")
    parser.add_argument("--output-json", default=None, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyzer = ECGClassifierFairnessAnalyzer(args.prediction_csv, group_column=args.group_column)
    output_json = args.output_json or args.prediction_csv.parent / f"fairness_{args.group_column}.json"
    path = analyzer.save_json(output_json)
    print(f"Saved fairness report to {path}")
