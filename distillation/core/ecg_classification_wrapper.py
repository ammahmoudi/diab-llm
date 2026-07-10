"""Classification distillation wrapper for MIT-BIH Time-LLM ECG models."""

from __future__ import annotations

import json
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.optim.adam import Adam

from data_processing.ecg.label_map import AAMI_CLASS_TO_ID
from models.ecg.time_llm_classifier import TimeLLMEcgClassifier


class ECGClassificationDistillationWrapper:
    """Teacher-student classification distillation for MIT-BIH ECG."""

    def __init__(self, settings, data_settings, log_dir, teacher_checkpoint_path):
        self.settings = settings
        self.data_settings = data_settings
        self.log_dir = log_dir
        self.teacher_checkpoint_path = teacher_checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.alpha = float(settings.get("distillation_alpha", 0.5))
        self.beta = float(settings.get("distillation_beta", 0.5))
        self.temperature = float(settings.get("distillation_temperature", 2.0))
        self.logger = logging.getLogger(__name__)

        self.teacher = self._build_model(is_student=False)
        self.student = self._build_model(is_student=True)
        self._load_teacher()

    def _teacher_model_name(self) -> str:
        return self.settings.get("teacher_model", self.settings.get("llm_model", "BERT"))

    def _teacher_model_config(self) -> Dict:
        teacher_model = self._teacher_model_name()
        teacher_defaults = {
            "BERT": {"llm_model": "BERT", "llm_layers": 12, "llm_dim": 768},
            "DistilBERT": {"llm_model": "DistilBERT", "llm_layers": 6, "llm_dim": 768},
            "TinyBERT": {"llm_model": "TinyBERT", "llm_layers": 4, "llm_dim": 312},
            "BERT-tiny": {"llm_model": "BERT-tiny", "llm_layers": 2, "llm_dim": 128},
            "MiniLM": {"llm_model": "MiniLM", "llm_layers": 6, "llm_dim": 384},
            "GPT2": {"llm_model": "GPT2", "llm_layers": 12, "llm_dim": 768},
        }
        config = teacher_defaults.get(teacher_model, teacher_defaults["BERT"]).copy()
        config.update(
            {
                "task_name": "ecg_classification",
                "sequence_length": int(self.settings.get("sequence_length", 256)),
                "enc_in": 1,
                "d_model": int(self.settings.get("d_model", 64)),
                "d_ff": int(self.settings.get("d_ff", 128)),
                "dropout": float(self.settings.get("dropout", 0.1)),
                "patch_len": int(self.settings.get("patch_len", 16)),
                "stride": int(self.settings.get("stride", 8)),
                "num_classes": int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID))),
                "pooling": self.settings.get("pooling", "mean"),
                "freeze_llm": bool(self.settings.get("freeze_llm", False)),
            }
        )
        return config

    def _student_model_config(self) -> Dict:
        return {
            "task_name": "ecg_classification",
            "sequence_length": int(self.settings.get("sequence_length", 256)),
            "enc_in": 1,
            "d_model": int(self.settings.get("d_model", 64)),
            "d_ff": int(self.settings.get("d_ff", 128)),
            "dropout": float(self.settings.get("dropout", 0.1)),
            "patch_len": int(self.settings.get("patch_len", 16)),
            "stride": int(self.settings.get("stride", 8)),
            "num_classes": int(self.settings.get("num_classes", len(AAMI_CLASS_TO_ID))),
            "pooling": self.settings.get("pooling", "mean"),
            "llm_model": self.settings.get("llm_model", "TinyBERT"),
            "llm_layers": int(self.settings.get("llm_layers", 4)),
            "llm_dim": int(self.settings.get("llm_dim", 312)),
            "freeze_llm": bool(self.settings.get("freeze_llm", False)),
        }

    def _build_model(self, is_student: bool) -> TimeLLMEcgClassifier:
        config = self._student_model_config() if is_student else self._teacher_model_config()
        return TimeLLMEcgClassifier(config).float().to(self.device)

    def _load_teacher(self):
        state_dict = torch.load(self.teacher_checkpoint_path, map_location=self.device, weights_only=True)
        self.teacher.load_state_dict(state_dict)
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.logger.info(f"Loaded ECG teacher checkpoint from {self.teacher_checkpoint_path}")

    def distill_knowledge(self, train_loader, val_loader=None, epochs=None):
        os.makedirs(os.path.join(self.log_dir, "checkpoints"), exist_ok=True)
        optimizer = Adam(
            [param for param in self.student.parameters() if param.requires_grad],
            lr=float(self.settings.get("learning_rate", 1e-4)),
        )
        class_weights = self._compute_class_weights(train_loader.dataset)
        ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        kl_loss = nn.KLDivLoss(reduction="batchmean")
        epochs = int(epochs or self.settings.get("train_epochs", 5))

        best_val = float("inf")
        best_path = os.path.join(self.log_dir, "checkpoints", "checkpoint_best.pth")
        last_path = os.path.join(self.log_dir, "checkpoints", "checkpoint_last.pth")
        train_history: List[Dict[str, float]] = []

        for epoch in range(epochs):
            train_loss = self._run_epoch(train_loader, optimizer, ce_loss, kl_loss, train=True)
            val_loss = self._run_epoch(val_loader, optimizer, ce_loss, kl_loss, train=False) if val_loader is not None else 0.0
            train_history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
            self.logger.info(
                f"ECG KD Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}"
            )
            if val_loader is not None and val_loss < best_val:
                best_val = val_loss
                torch.save(self.student.state_dict(), best_path)

        torch.save(self.student.state_dict(), last_path)
        if val_loader is None:
            best_path = last_path
        with open(os.path.join(self.log_dir, "distillation_history.json"), "w") as f:
            json.dump(train_history, f, indent=2)
        return best_path, [row["train_loss"] for row in train_history], [row["val_loss"] for row in train_history]

    def predict(self, test_loader, output_dir=None):
        self.student.eval()
        rows: List[Dict[str, object]] = []
        all_true: List[int] = []
        all_pred: List[int] = []
        with torch.no_grad():
            for batch_x, batch_y, meta in test_loader:
                logits = self.student(batch_x.to(self.device))
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                true = batch_y.numpy()
                all_true.extend(true.tolist())
                all_pred.extend(preds.tolist())
                for i in range(len(preds)):
                    row = {
                        "record_id": meta["record_id"][i],
                        "beat_sample_index": int(meta["beat_sample_index"][i]),
                        "raw_symbol": meta["raw_symbol"][i],
                        "aami_class": meta["aami_class"][i],
                        "sex": meta["sex"][i],
                        "age_group": meta["age_group"][i],
                        "paced_group": meta["paced_group"][i],
                        "difficulty_group": meta["difficulty_group"][i],
                        "y_true": int(true[i]),
                        "y_pred": int(preds[i]),
                    }
                    for class_idx in range(probs.shape[1]):
                        row[f"prob_{class_idx}"] = float(probs[i, class_idx])
                    rows.append(row)
        predictions_df = pd.DataFrame(rows)
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            predictions_df.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
        return np.asarray(all_pred), np.asarray(all_true), predictions_df

    def evaluate(self, predictions, targets, metrics=None):
        if metrics is None:
            metrics = ["accuracy", "macro_f1", "weighted_f1"]
        y_pred = np.asarray(predictions)
        y_true = np.asarray(targets)
        results: Dict[str, float] = {}
        if "accuracy" in metrics:
            results["accuracy"] = float(accuracy_score(y_true, y_pred))
        if "macro_f1" in metrics:
            results["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        if "weighted_f1" in metrics:
            results["weighted_f1"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
        return results

    def classification_report_dict(self, y_true, y_pred):
        return classification_report(y_true, y_pred, zero_division=0, output_dict=True)

    def _compute_class_weights(self, dataset) -> torch.Tensor:
        labels = [sample.class_id for sample in dataset.samples]
        counts = np.bincount(labels, minlength=len(AAMI_CLASS_TO_ID))
        counts = np.maximum(counts, 1)
        weights = counts.sum() / (len(counts) * counts)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _run_epoch(self, loader, optimizer, ce_loss, kl_loss, train: bool) -> float:
        if loader is None:
            return 0.0
        self.student.train(train)
        losses: List[float] = []
        for batch_x, batch_y, _meta in loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            with torch.no_grad():
                teacher_logits = self.teacher(batch_x)
            student_logits = self.student(batch_x)

            supervised = ce_loss(student_logits, batch_y)
            kd = kl_loss(
                torch.log_softmax(student_logits / self.temperature, dim=1),
                torch.softmax(teacher_logits / self.temperature, dim=1),
            ) * (self.temperature ** 2)
            loss = self.alpha * supervised + self.beta * kd

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else 0.0
