#!/usr/bin/env python3
"""Fairness analyzer for MIT-BIH ECG classification outputs."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score


class ECGClassifierFairnessAnalyzer:
    """Analyze subgroup fairness for AAMI 5-class ECG classification outputs."""

    def __init__(self, prediction_csv: str | Path, group_column: str = "sex"):
        self.prediction_csv = Path(prediction_csv)
        self.group_column = group_column
        self.df = pd.read_csv(self.prediction_csv)
        required = {"y_true", "y_pred", group_column}
        missing = required.difference(self.df.columns)
        if missing:
            raise ValueError(f"Prediction CSV missing required columns: {sorted(missing)}")
        self.class_ids = sorted(int(v) for v in pd.unique(self.df["y_true"]))

    def analyze(self) -> Dict[str, object]:
        grouped = self._group_metrics()
        classwise = self._classwise_one_vs_rest(grouped)
        return {
            "group_column": self.group_column,
            "overall": grouped,
            "classwise_one_vs_rest": classwise,
        }

    def save_json(self, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(self.analyze(), f, indent=2)
        return output_path

    def _group_metrics(self) -> Dict[str, Dict[str, float]]:
        metrics: Dict[str, Dict[str, float]] = {}
        for group_value, group_df in self.df.groupby(self.group_column):
            y_true = group_df["y_true"].to_numpy()
            y_pred = group_df["y_pred"].to_numpy()
            metrics[str(group_value)] = {
                "count": int(len(group_df)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
            }
            for class_id in self.class_ids:
                y_true_bin = (y_true == class_id).astype(int)
                y_pred_bin = (y_pred == class_id).astype(int)
                metrics[str(group_value)][f"recall_class_{class_id}"] = float(
                    recall_score(y_true_bin, y_pred_bin, zero_division=0)
                )
        return metrics

    def _classwise_one_vs_rest(self, grouped_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        classwise: Dict[str, Dict[str, float]] = {}
        group_names = list(grouped_metrics.keys())
        for class_id in self.class_ids:
            recalls = [grouped_metrics[group][f"recall_class_{class_id}"] for group in group_names]
            if not recalls:
                continue
            classwise[str(class_id)] = {
                "max_recall_gap": float(max(recalls) - min(recalls)),
                "worst_group_recall": float(min(recalls)),
                "best_group_recall": float(max(recalls)),
            }
        if len(group_names) >= 2:
            macro_values = [grouped_metrics[group]["macro_f1"] for group in group_names]
            classwise["summary"] = {
                "macro_f1_gap": float(max(macro_values) - min(macro_values)),
                "groups_compared": group_names,
            }
        return classwise
