"""Build MIT-BIH beat-centered index CSV.

Usage:
    python scripts/mitbih/prepare_beat_dataset.py
    python scripts/mitbih/prepare_beat_dataset.py --window-size 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.ecg_metadata_parser import DEFAULT_MITBIH_DIR
from data_processing.ecg_mitbih_dataset import MitBihBeatDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MIT-BIH beat-centered index CSV")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_MITBIH_DIR,
        help="Path to MIT-BIH dataset directory",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=256,
        help="Beat-centered window size in samples",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <dataset-dir>/beat_index.csv",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset = MitBihBeatDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        window_size=args.window_size,
    )
    output = dataset.build_index(
        output_path=args.output_path or Path(args.dataset_dir) / "beat_index.csv"
    )
    print(f"Saved beat index CSV to {output}")
