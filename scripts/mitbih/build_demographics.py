"""Build MIT-BIH demographics CSV.

Usage:
    python scripts/mitbih/build_demographics.py
    python scripts/mitbih/build_demographics.py --dataset-dir data/mit-bih-arrhythmia
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.ecg_metadata_parser import DEFAULT_MITBIH_DIR, build_demographics_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-BIH demographics CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_MITBIH_DIR,
        help="Path to MIT-BIH dataset directory",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset-dir>/demographics_records.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = build_demographics_csv(
        dataset_dir=args.dataset_dir,
        output_path=args.output_path or Path(args.dataset_dir) / "demographics_records.csv",
    )
    print(f"Saved demographics CSV to {output}")
