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

from data_processing.ecg_metadata_parser import (
    DEFAULT_MITBIH_DIR,
    MitBihMetadataParser,
    resolve_excluded_record_ids,
)
from data_processing.ecg_mitbih_dataset import (
    MitBihBeatDataset,
    build_record_level_split_assignments,
)


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
    parser.add_argument(
        "--include-duplicate-202",
        action="store_true",
        help="Include duplicate record 202 instead of using the default curated 47-record set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for record-level split assignment")
    parser.add_argument("--train-ratio", type=float, default=0.6, help="Record-level training split ratio")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Record-level validation split ratio")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    metadata_parser = MitBihMetadataParser(dataset_dir=args.dataset_dir)
    record_ids = metadata_parser.load_record_ids(
        exclude_record_ids=resolve_excluded_record_ids(
            include_duplicate_202=args.include_duplicate_202
        )
    )
    split_assignments = build_record_level_split_assignments(
        record_ids,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )
    dataset = MitBihBeatDataset(
        dataset_dir=args.dataset_dir,
        split="all",
        window_size=args.window_size,
        include_duplicate_202=args.include_duplicate_202,
        split_assignments=split_assignments,
    )
    output = dataset.build_index(
        output_path=args.output_path or Path(args.dataset_dir) / "beat_index.csv"
    )
    print(f"Saved beat index CSV to {output}")
