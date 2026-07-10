"""Sampling helpers for MIT-BIH ECG classification fairness experiments."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler


def build_group_labels_from_samples(dataset, feature: str) -> np.ndarray:
    """Extract one group label per ECG beat sample from dataset metadata."""
    valid_features = {"sex", "age_group", "paced_group", "difficulty_group"}
    if feature not in valid_features:
        raise ValueError(f"Unsupported ECG fairness feature: {feature}. Expected one of {sorted(valid_features)}")
    return np.asarray([getattr(sample, feature) for sample in dataset.samples], dtype=object)


def build_group_class_sampler(
    dataset,
    group_labels: np.ndarray,
    max_oversample: float = 4.0,
) -> Tuple[WeightedRandomSampler, Dict[str, object]]:
    """Create a sampler that upweights rare group/class combinations.

    This is the ECG classification analogue of fair teacher sampling.
    """
    if len(group_labels) != len(dataset.samples):
        raise ValueError("group_labels length must match number of ECG samples")

    class_ids = np.asarray([sample.class_id for sample in dataset.samples], dtype=int)
    combo_counter = Counter((str(group), int(class_id)) for group, class_id in zip(group_labels, class_ids))
    mean_count = float(np.mean(list(combo_counter.values()))) if combo_counter else 1.0

    weights = []
    for group, class_id in zip(group_labels, class_ids):
        combo_count = combo_counter[(str(group), int(class_id))]
        raw_weight = mean_count / max(1.0, float(combo_count))
        weights.append(min(max_oversample, raw_weight))

    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / weights.mean()

    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(weights),
        replacement=True,
    )

    info = {
        "group_class_counts": {f"{group}|{class_id}": count for (group, class_id), count in combo_counter.items()},
        "max_oversample": float(max_oversample),
        "mean_weight": float(weights.mean()),
    }
    return sampler, info
