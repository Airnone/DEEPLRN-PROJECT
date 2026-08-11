"""Select a global finding threshold using validation predictions only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from deeplrn.checkpoints import load_checkpoint
from deeplrn.evaluation import multilabel_classification_metrics
from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
from deeplrn.schema import FINDING_LABELS
from deeplrn.training.dataset import DeepLRNDataset
from deeplrn.training.trainer import Trainer, TrainingConfig


def _parse_thresholds(value: str) -> list[float]:
    thresholds = sorted({float(item.strip()) for item in value.split(",")})
    if not thresholds or any(not 0.0 < threshold < 1.0 for threshold in thresholds):
        raise argparse.ArgumentTypeError("thresholds must be between zero and one")
    return thresholds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--thresholds",
        type=_parse_thresholds,
        default=_parse_thresholds("0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90"),
    )
    args = parser.parse_args()

    # Deliberately load only the train and validation entries. The train split is
    # needed for the pipeline's supervision/coverage checks; test is never opened.
    train_dataset = DeepLRNDataset.from_manifest(
        args.manifest, "train", max_tokens=args.max_tokens
    )
    validation_dataset = DeepLRNDataset.from_manifest(
        args.manifest, "validation", max_tokens=args.max_tokens
    )
    checkpoint = load_checkpoint(args.checkpoint, map_location="cpu")
    model = DeepLRNModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer = Trainer(
        model,
        train_dataset,
        validation_dataset,
        TrainingConfig(
            batch_size=args.batch_size,
            num_epochs=1,
            ner_loss_weight=0.0,
            cls_loss_weight=1.0,
            rel_loss_weight=0.0,
            selection_metric="finding_macro_f1",
            output_dir=str(Path(args.output).parent / "calibration_runtime"),
        ),
    )

    probabilities: list[torch.Tensor] = []
    gold: list[list[float]] = []
    trainer.model.eval()
    with torch.no_grad():
        for batch in trainer.eval_loader:
            outputs = trainer._forward(batch)
            probabilities.append(outputs["cls_logits"].sigmoid().cpu())
            gold.extend(batch["finding_labels"].tolist())
    probability_tensor = torch.cat(probabilities, dim=0)

    results = []
    for threshold in args.thresholds:
        prediction_tensor = (probability_tensor > threshold).int()
        predictions = prediction_tensor.tolist()
        metrics = multilabel_classification_metrics(
            predictions, gold, len(FINDING_LABELS)
        )
        predicted_positive_count = int(prediction_tensor.sum().item())
        results.append(
            {
                "threshold": threshold,
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "subset_accuracy": metrics["subset_accuracy"],
                "hamming_accuracy": metrics["hamming_accuracy"],
                "predicted_positive_count": predicted_positive_count,
                "predicted_positive_rate": predicted_positive_count
                / prediction_tensor.numel(),
                "mean_predicted_labels_per_document": predicted_positive_count
                / len(validation_dataset),
            }
        )

    best = max(
        results,
        key=lambda item: (
            item["macro_f1"],
            item["micro_f1"],
            item["hamming_accuracy"],
            -abs(item["threshold"] - 0.5),
        ),
    )
    report = {
        "selection_data": "validation",
        "test_loaded": False,
        "manifest": args.manifest,
        "checkpoint": args.checkpoint,
        "validation_documents": len(validation_dataset),
        "gold_positive_count": int(torch.tensor(gold).sum().item()),
        "gold_positive_rate": float(torch.tensor(gold).float().mean().item()),
        "finding_labels": list(FINDING_LABELS),
        "selection_metric": "macro_f1",
        "best": best,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
