"""Build a human-review queue from RoBERTa/SVM disagreements.

Only manifest records assigned to train or validation are loaded. The held-out
test split is never opened, and this script never changes annotation files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import torch
from torch.utils.data import DataLoader

from deeplrn.checkpoints import load_checkpoint
from deeplrn.model.deeplrn_model import DeepLRNModel, ModelConfig
from deeplrn.schema import FINDING_LABELS
from deeplrn.training.dataset import DeepLRNDataset, collate_documents

ALLOWED_SPLITS = ("train", "validation")
BLANKET_SUPERVISION = "owner_blanket_approved_machine_labels"


def _labels(record: dict[str, Any]) -> set[str]:
    if "finding_labels" in record:
        return set(record["finding_labels"])
    vector = record.get("finding_label_vector", [])
    return {
        label
        for index, label in enumerate(FINDING_LABELS)
        if index < len(vector) and int(vector[index]) == 1
    }


def _load_allowed_records(manifest_path: Path) -> dict[str, dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records: dict[str, dict[str, Any]] = {}
    for item in manifest.get("documents", []):
        split = item.get("split")
        if split not in ALLOWED_SPLITS:
            continue
        source = Path(item["source_path"])
        if not source.is_absolute():
            source = manifest_path.parent / source
        record = json.loads(source.read_text(encoding="utf-8"))
        doc_id = str(record["doc_id"])
        record["_manifest_split"] = split
        record["_manifest_source_path"] = str(source)
        records[doc_id] = record
    if not records:
        raise ValueError("manifest contains no train or validation records")
    return records


def _predict_roberta(
    manifest_path: Path,
    checkpoint_path: Path,
    *,
    batch_size: int,
    max_tokens: int,
) -> dict[str, list[float]]:
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    model = DeepLRNModel(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    predictions: dict[str, list[float]] = {}
    with torch.no_grad():
        for split in ALLOWED_SPLITS:
            dataset = DeepLRNDataset.from_manifest(manifest_path, split, max_tokens=max_tokens)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_documents,
            )
            for batch in loader:
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    chunk_mask=batch["chunk_mask"],
                    page_ids=batch["page_ids"],
                    section_ids=batch["section_ids"],
                    bboxes=batch["bboxes"],
                )
                probabilities = outputs["cls_logits"].sigmoid().tolist()
                predictions.update(zip(batch["doc_ids"], probabilities))
    return predictions


def _priority_score(
    approved: set[str],
    roberta: set[str],
    svm: set[str],
    probabilities: list[float],
    supports: Counter[str],
) -> tuple[float, set[str], set[str], set[str]]:
    model_disagreement = roberta ^ svm
    roberta_conflicts = roberta ^ approved
    svm_conflicts = svm ^ approved
    implicated = model_disagreement | roberta_conflicts | svm_conflicts
    rarity = sum(10.0 / math.sqrt(max(1, supports[label])) for label in implicated)
    uncertainty = sum(
        max(0.0, 1.0 - 2.0 * abs(probabilities[index] - 0.5))
        for index, label in enumerate(FINDING_LABELS)
        if label in implicated
    )
    score = (
        10.0 * len(model_disagreement)
        + 4.0 * len(roberta_conflicts)
        + 4.0 * len(svm_conflicts)
        + rarity
        + uncertainty
    )
    return score, model_disagreement, roberta_conflicts, svm_conflicts


def select_review_queue(
    candidates: list[dict[str, Any]],
    *,
    limit: int,
    validation_cap: int,
    per_label_seed: int = 4,
) -> list[dict[str, Any]]:
    """Select a rare-label-aware queue while capping validation review."""

    if limit < 1 or validation_cap < 0:
        raise ValueError("limit must be positive and validation cap non-negative")
    ranked = sorted(
        candidates,
        key=lambda item: (-float(item["priority_score"]), str(item["doc_id"])),
    )
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    validation_count = 0

    def add(item: dict[str, Any]) -> bool:
        nonlocal validation_count
        if item["doc_id"] in selected_ids or len(selected) >= limit:
            return False
        if item["split"] == "validation" and validation_count >= validation_cap:
            return False
        selected.append(item)
        selected_ids.add(item["doc_id"])
        validation_count += item["split"] == "validation"
        return True

    for label in FINDING_LABELS:
        label_count = 0
        for item in ranked:
            if label not in item["implicated_labels"]:
                continue
            if add(item):
                label_count += 1
            if label_count >= per_label_seed or len(selected) >= limit:
                break
    for item in ranked:
        add(item)
        if len(selected) >= limit:
            break
    return selected


def build_queue(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = Path(args.manifest)
    records = _load_allowed_records(manifest_path)
    blanket_records = {
        doc_id: record
        for doc_id, record in records.items()
        if record.get("annotation_metadata", {}).get("supervision_quality") == BLANKET_SUPERVISION
    }
    supports: Counter[str] = Counter(
        label for record in records.values() for label in _labels(record)
    )
    roberta_probabilities = _predict_roberta(
        manifest_path,
        Path(args.roberta_checkpoint),
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
    )
    baseline = joblib.load(args.svm_model)
    svm_pipeline = baseline["model"]
    doc_ids = sorted(blanket_records)
    texts = [blanket_records[doc_id]["document_text"] for doc_id in doc_ids]
    svm_predictions = svm_pipeline.predict(texts).tolist()
    svm_scores = svm_pipeline.decision_function(texts).tolist()

    candidates: list[dict[str, Any]] = []
    for doc_id, svm_vector, score_vector in zip(doc_ids, svm_predictions, svm_scores):
        record = blanket_records[doc_id]
        probabilities = roberta_probabilities[doc_id]
        approved = _labels(record)
        roberta = {
            label
            for index, label in enumerate(FINDING_LABELS)
            if probabilities[index] > args.roberta_threshold
        }
        svm = {label for index, label in enumerate(FINDING_LABELS) if int(svm_vector[index]) == 1}
        score, disagreement, roberta_conflicts, svm_conflicts = _priority_score(
            approved, roberta, svm, probabilities, supports
        )
        if not (disagreement or roberta_conflicts or svm_conflicts):
            continue
        implicated = disagreement | roberta_conflicts | svm_conflicts
        candidates.append(
            {
                "doc_id": doc_id,
                "split": record["_manifest_split"],
                "lgu": record.get("lgu", ""),
                "year": record.get("year"),
                "source_pdf": record.get("source_pdf", ""),
                "source_record": record["_manifest_source_path"],
                "observation_text": record.get("document_text", ""),
                "approved_labels": sorted(approved),
                "roberta_predicted_labels": sorted(roberta),
                "svm_predicted_labels": sorted(svm),
                "model_disagreement_labels": sorted(disagreement),
                "roberta_vs_approved_labels": sorted(roberta_conflicts),
                "svm_vs_approved_labels": sorted(svm_conflicts),
                "implicated_labels": sorted(implicated),
                "roberta_probabilities": {
                    label: round(float(probabilities[index]), 6)
                    for index, label in enumerate(FINDING_LABELS)
                },
                "svm_decision_scores": {
                    label: round(float(score_vector[index]), 6)
                    for index, label in enumerate(FINDING_LABELS)
                },
                "priority_score": round(score, 6),
                "review_decision": None,
                "corrected_labels": [],
                "review_notes": "",
            }
        )

    selected = select_review_queue(
        candidates,
        limit=args.limit,
        validation_cap=args.validation_cap,
    )
    if len(selected) < args.limit:
        raise ValueError(
            f"only {len(selected)} eligible disagreements for requested limit {args.limit}"
        )
    for rank, item in enumerate(selected, start=1):
        item["queue_rank"] = rank

    split_counts = Counter(item["split"] for item in selected)
    implicated_counts = Counter(label for item in selected for label in item["implicated_labels"])
    summary = {
        "queue_size": len(selected),
        "eligible_blanket_approved_records": len(blanket_records),
        "eligible_disagreement_records": len(candidates),
        "excluded_individually_reviewed_records": len(records) - len(blanket_records),
        "splits_loaded": list(ALLOWED_SPLITS),
        "selected_split_counts": dict(sorted(split_counts.items())),
        "selected_implicated_label_counts": dict(sorted(implicated_counts.items())),
        "roberta_checkpoint": str(args.roberta_checkpoint),
        "roberta_threshold": args.roberta_threshold,
        "svm_model": str(args.svm_model),
        "labels_changed": False,
        "test_split_accessed": False,
    }
    return selected, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--roberta-checkpoint", required=True)
    parser.add_argument("--svm-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--validation-cap", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--roberta-threshold", type=float, default=0.5)
    args = parser.parse_args()

    queue, summary = build_queue(args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in queue),
        encoding="utf-8",
    )
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
