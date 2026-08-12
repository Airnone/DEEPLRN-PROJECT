"""Fit finding-label thresholds using the validation split only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from deeplrn.checkpoints import load_checkpoint
from deeplrn.evaluation import (
    apply_multilabel_thresholds,
    multilabel_classification_metrics,
    tune_multilabel_thresholds,
)
from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
from deeplrn.schema import FINDING_LABELS
from deeplrn.training.dataset import DeepLRNDataset, collate_documents


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune per-label finding thresholds on validation data only."
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--default-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = DeepLRNModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = DeepLRNDataset.from_manifest(args.manifest, "validation", max_tokens=args.max_tokens)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_documents,
    )
    probabilities: list[list[float]] = []
    gold: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                chunk_mask=batch["chunk_mask"],
                page_ids=batch["page_ids"],
                section_ids=batch["section_ids"],
                bboxes=batch["bboxes"],
            )
            probabilities.extend(outputs["cls_logits"].sigmoid().tolist())
            gold.extend(batch["finding_labels"].tolist())

    default_thresholds = [float(args.default_threshold)] * len(FINDING_LABELS)
    tuned_thresholds = tune_multilabel_thresholds(
        probabilities,
        gold,
        len(FINDING_LABELS),
        default_threshold=args.default_threshold,
    )
    default_metrics = multilabel_classification_metrics(
        apply_multilabel_thresholds(probabilities, default_thresholds),
        gold,
        len(FINDING_LABELS),
    )
    tuned_metrics = multilabel_classification_metrics(
        apply_multilabel_thresholds(probabilities, tuned_thresholds),
        gold,
        len(FINDING_LABELS),
    )
    result = {
        "calibration_split": "validation",
        "checkpoint": str(Path(args.checkpoint)),
        "default_threshold": float(args.default_threshold),
        "document_count": len(dataset),
        "evaluation_note": (
            "Thresholds and reported tuned metrics use the same validation data; "
            "the tuned metrics are an optimistic calibration-fit score."
        ),
        "label_thresholds": {
            label: tuned_thresholds[index] for index, label in enumerate(FINDING_LABELS)
        },
        "default_metrics": default_metrics,
        "tuned_metrics": tuned_metrics,
        "test_split_evaluated": False,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
