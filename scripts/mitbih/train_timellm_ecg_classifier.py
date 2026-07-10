"""Train a Time-LLM-inspired ECG classifier on MIT-BIH.

This runner is separate from the BG forecasting system and uses the MIT-BIH
beat-centered dataset and AAMI 5-class labels.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.adam import Adam
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.ecg.label_map import AAMI_CLASS_TO_ID
from data_processing.ecg.metadata import DEFAULT_MITBIH_DIR, MitBihMetadataParser, resolve_excluded_record_ids
from data_processing.ecg.dataset import MitBihBeatDataset, build_record_level_split_assignments
from models.ecg.time_llm_classifier import TimeLLMEcgClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Time-LLM ECG classifier")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_MITBIH_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs" / "mitbih_timellm_classifier")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-model", type=str, default="TinyBERT")
    parser.add_argument("--pooling", type=str, default="mean", choices=["mean", "last", "center"])
    parser.add_argument("--freeze-llm", action="store_true")
    parser.add_argument("--include-duplicate-202", action="store_true")
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    return parser.parse_args()


def _json_safe_config(args: argparse.Namespace) -> Dict[str, object]:
    safe = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            safe[key] = str(value)
        else:
            safe[key] = value
    return safe


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_split_assignments(dataset_dir: Path, include_duplicate_202: bool, seed: int) -> Dict[str, str]:
    parser = MitBihMetadataParser(dataset_dir=dataset_dir)
    record_ids = parser.load_record_ids(exclude_record_ids=resolve_excluded_record_ids(include_duplicate_202=include_duplicate_202))
    return build_record_level_split_assignments(record_ids, seed=seed)


def ensure_index(dataset_dir: Path, window_size: int, include_duplicate_202: bool, rebuild_index: bool, split_assignments: Dict[str, str]) -> Path:
    index_path = dataset_dir / "beat_index.csv"
    should_rebuild = rebuild_index or True
    if should_rebuild or not index_path.exists():
        dataset = MitBihBeatDataset(
            dataset_dir=dataset_dir,
            split="all",
            window_size=window_size,
            split_assignments=split_assignments,
            include_duplicate_202=include_duplicate_202,
        )
        dataset.build_index(index_path)
    return index_path


def make_loader(dataset_dir: Path, split: str, window_size: int, batch_size: int, include_duplicate_202: bool) -> DataLoader:
    dataset = MitBihBeatDataset(
        dataset_dir=dataset_dir,
        split=split,
        window_size=window_size,
        include_duplicate_202=include_duplicate_202,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=0)


def compute_class_weights(dataset: MitBihBeatDataset, device: torch.device) -> torch.Tensor:
    labels = [sample.class_id for sample in dataset.samples]
    counts = np.bincount(labels, minlength=len(AAMI_CLASS_TO_ID))
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(model, loader, optimizer, criterion, device, train: bool, max_batches: int | None = None) -> float:
    model.train(train)
    losses: List[float] = []
    for batch_idx, (batch_x, batch_y, _meta) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def evaluate(model, loader, device, max_batches: int | None = None) -> Tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    all_true: List[int] = []
    all_pred: List[int] = []
    rows: List[Dict[str, object]] = []
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y, meta) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            logits = model(batch_x.to(device))
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            true = batch_y.numpy()
            all_true.extend(true.tolist())
            all_pred.extend(preds.tolist())
            for i in range(len(preds)):
                rows.append(
                    {
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
                )
    metrics = {
        "accuracy": accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, average="weighted", zero_division=0),
    }
    return metrics, pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    split_assignments = build_split_assignments(args.dataset_dir, args.include_duplicate_202, args.seed)
    ensure_index(args.dataset_dir, args.window_size, args.include_duplicate_202, args.rebuild_index, split_assignments)

    train_dataset = MitBihBeatDataset(
        dataset_dir=args.dataset_dir,
        split="train",
        window_size=args.window_size,
        include_duplicate_202=args.include_duplicate_202,
    )
    val_loader = make_loader(args.dataset_dir, "val", args.window_size, args.batch_size, args.include_duplicate_202)
    test_loader = make_loader(args.dataset_dir, "test", args.window_size, args.batch_size, args.include_duplicate_202)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeLLMEcgClassifier(
        {
            "task_name": "ecg_classification",
            "sequence_length": args.window_size,
            "enc_in": 1,
            "d_model": 64,
            "d_ff": 128,
            "dropout": 0.1,
            "patch_len": 16,
            "stride": 8,
            "num_classes": len(AAMI_CLASS_TO_ID),
            "pooling": args.pooling,
            "llm_model": args.llm_model,
            "freeze_llm": args.freeze_llm,
        }
    ).to(device)

    class_weights = compute_class_weights(train_dataset, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history: List[Dict[str, float]] = []
    best_val = float("inf")
    best_path = args.output_dir / "checkpoint_best.pth"

    for epoch in range(args.epochs):
        train_loss = run_epoch(
            model, train_loader, optimizer, criterion, device, train=True, max_batches=args.max_train_batches
        )
        val_loss = run_epoch(
            model, val_loader, optimizer, criterion, device, train=False, max_batches=args.max_eval_batches
        )
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)

    torch.save(model.state_dict(), args.output_dir / "checkpoint_last.pth")
    with (args.output_dir / "train_history.json").open("w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_metrics, val_df = evaluate(model, val_loader, device, max_batches=args.max_eval_batches)
    test_metrics, test_df = evaluate(model, test_loader, device, max_batches=args.max_eval_batches)
    report = {
        "config": _json_safe_config(args),
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "class_names": sorted(AAMI_CLASS_TO_ID.keys(), key=lambda key: AAMI_CLASS_TO_ID[key]),
        "classification_report": classification_report(
            test_df["y_true"],
            test_df["y_pred"],
            zero_division=0,
            output_dict=True,
        ),
    }
    with (args.output_dir / "metrics_summary.json").open("w") as f:
        json.dump(report, f, indent=2)
    val_df.to_csv(args.output_dir / "val_predictions.csv", index=False)
    test_df.to_csv(args.output_dir / "test_predictions.csv", index=False)
    print(json.dumps({"val": val_metrics, "test": test_metrics}, indent=2))


if __name__ == "__main__":
    main()
