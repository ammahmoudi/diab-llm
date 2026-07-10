"""Beat-centered MIT-BIH dataset scaffold for Time-LLM classification.

This is an ECG-specific dataset module added alongside the BG forecasting data
modules so the original system remains untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from data_processing.ecg_label_map import MitBihAamiMapper
from data_processing.ecg_metadata_parser import (
    DEFAULT_MITBIH_DIR,
    MitBihMetadataParser,
    resolve_excluded_record_ids,
)


try:
    import wfdb  # type: ignore
except ImportError:  # pragma: no cover - dependency may be installed later
    wfdb = None


@dataclass
class EcgBeatSample:
    record_id: str
    beat_sample_index: int
    raw_symbol: str
    aami_class: str
    class_id: int
    signal_window: np.ndarray
    sex: str
    age_group: str
    paced_group: str
    difficulty_group: str
    split: str


class MitBihBeatDataset(Dataset):
    """Beat-centered MIT-BIH dataset.

    Current scope:
    - single lead
    - beat-centered windows
    - AAMI 5-class labels
    - record-level split metadata

    This scaffold is intentionally separate from the existing Dataset_T1DM so the
    BG forecasting workflow stays unchanged.
    """

    def __init__(
        self,
        dataset_dir: Path | str = DEFAULT_MITBIH_DIR,
        split: str = "train",
        window_size: int = 256,
        primary_lead_only: bool = True,
        metadata_csv: Optional[Path | str] = None,
        beat_index_csv: Optional[Path | str] = None,
        normalize: bool = True,
        split_assignments: Optional[Dict[str, str]] = None,
        include_duplicate_202: bool = False,
    ):
        if split not in {"train", "val", "test", "all"}:
            raise ValueError("split must be one of train/val/test/all")
        self.dataset_dir = Path(dataset_dir)
        self.split = split
        self.window_size = window_size
        self.primary_lead_only = primary_lead_only
        self.normalize = normalize
        self.mapper = MitBihAamiMapper()
        self.metadata_parser = MitBihMetadataParser(dataset_dir=self.dataset_dir)
        self.metadata_csv = Path(metadata_csv) if metadata_csv is not None else self.dataset_dir / "metadata_records.csv"
        self.beat_index_csv = Path(beat_index_csv) if beat_index_csv is not None else self.dataset_dir / "beat_index.csv"
        self.split_assignments = split_assignments or {}
        self.include_duplicate_202 = include_duplicate_202

        self.metadata = self._load_metadata()
        self.samples: List[EcgBeatSample] = []

        if self.beat_index_csv.exists():
            self.samples = self._load_from_index(self.beat_index_csv)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        x = sample.signal_window.astype(np.float32).reshape(-1, 1)
        y = np.int64(sample.class_id)
        meta = {
            "record_id": sample.record_id,
            "beat_sample_index": sample.beat_sample_index,
            "raw_symbol": sample.raw_symbol,
            "aami_class": sample.aami_class,
            "sex": sample.sex,
            "age_group": sample.age_group,
            "paced_group": sample.paced_group,
            "difficulty_group": sample.difficulty_group,
            "split": sample.split,
        }
        return x, y, meta

    def _load_metadata(self) -> pd.DataFrame:
        if self.metadata_csv.exists():
            return pd.read_csv(self.metadata_csv)
        rows = [
            vars(row)
            for row in self.metadata_parser.parse_all(
                record_ids=self.metadata_parser.load_record_ids(
                    exclude_record_ids=resolve_excluded_record_ids(
                        include_duplicate_202=self.include_duplicate_202
                    )
                )
            )
        ]
        return pd.DataFrame(rows)

    def _load_from_index(self, index_path: Path) -> List[EcgBeatSample]:
        df = pd.read_csv(index_path)
        samples: List[EcgBeatSample] = []
        for _, row in df.iterrows():
            row_split = row.get("split", "all")
            if self.split != "all" and row_split != self.split:
                continue
            signal = np.fromstring(str(row["signal_window"]), sep=" ")
            samples.append(
                EcgBeatSample(
                    record_id=str(row["record_id"]),
                    beat_sample_index=int(row["beat_sample_index"]),
                    raw_symbol=str(row["raw_symbol"]),
                    aami_class=str(row["aami_class"]),
                    class_id=int(row["class_id"]),
                    signal_window=signal,
                    sex=str(row.get("sex", "unknown")),
                    age_group=str(row.get("age_group", "unknown")),
                    paced_group=str(row.get("paced_group", "non_paced")),
                    difficulty_group=str(row.get("difficulty_group", "unknown")),
                    split=str(row_split),
                )
            )
        return samples

    def build_index(self, output_path: Optional[Path | str] = None) -> Path:
        if wfdb is None:
            raise ImportError("wfdb is required to build MIT-BIH beat windows. Install with `pip install wfdb`.")

        output_path = Path(output_path) if output_path is not None else self.beat_index_csv
        records = self.metadata_parser.load_record_ids(
            exclude_record_ids=resolve_excluded_record_ids(
                include_duplicate_202=self.include_duplicate_202
            )
        )
        rows: List[Dict[str, object]] = []
        half_left = self.window_size // 2
        half_right = self.window_size - half_left

        metadata_by_record = {str(row["record_id"]): row for _, row in self.metadata.iterrows()}

        for record_id in records:
            signal, lead_name = self._load_primary_signal(record_id)
            signal = self._normalize_signal(signal) if self.normalize else signal
            ann = wfdb.rdann(str(self.dataset_dir / record_id), "atr")
            record_meta = metadata_by_record.get(record_id, {})
            assigned_split = self.split_assignments.get(record_id, "all")

            for beat_idx, symbol in zip(ann.sample, ann.symbol):
                mapped = self.mapper.map_symbol(symbol)
                if mapped is None:
                    continue
                window = self._extract_window(signal, beat_idx, half_left, half_right)
                rows.append(
                    {
                        "record_id": record_id,
                        "beat_sample_index": int(beat_idx),
                        "raw_symbol": symbol,
                        "aami_class": mapped.aami_class,
                        "class_id": mapped.class_id,
                        "primary_lead_used": lead_name,
                        "sex": record_meta.get("sex", "unknown"),
                        "age_group": record_meta.get("age_group", "unknown"),
                        "paced_group": record_meta.get("paced_group", "non_paced"),
                        "difficulty_group": record_meta.get("difficulty_group", "unknown"),
                        "split": assigned_split,
                        "signal_window": " ".join(map(str, window.astype(np.float32))),
                    }
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        self.samples = self._load_from_index(output_path)
        return output_path

    def _load_primary_signal(self, record_id: str) -> Tuple[np.ndarray, str]:
        if wfdb is None:
            raise ImportError("wfdb is required to load MIT-BIH signals.")
        record = wfdb.rdrecord(str(self.dataset_dir / record_id))
        sig_name = list(record.sig_name)
        lead_idx = sig_name.index("MLII") if self.primary_lead_only and "MLII" in sig_name else 0
        signal = np.asarray(record.p_signal[:, lead_idx], dtype=np.float32)
        return signal, sig_name[lead_idx]

    @staticmethod
    def _normalize_signal(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        median = np.median(signal)
        mad = np.median(np.abs(signal - median))
        if mad > eps:
            return (signal - median) / mad
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        return (signal - mean) / max(std, eps)

    @staticmethod
    def _extract_window(signal: np.ndarray, center: int, left: int, right: int) -> np.ndarray:
        start = center - left
        end = center + right
        if start >= 0 and end <= len(signal):
            return signal[start:end]
        pad_left = max(0, -start)
        pad_right = max(0, end - len(signal))
        clipped = signal[max(0, start):min(len(signal), end)]
        if clipped.size == 0:
            clipped = np.zeros((0,), dtype=np.float32)
        return np.pad(clipped, (pad_left, pad_right), mode="reflect" if clipped.size > 1 else "constant")[: left + right]
