"""Named model presets and executable non-neural baselines."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier

from deeplrn.evaluation import multilabel_classification_metrics
from deeplrn.schema import FINDING_LABELS


@dataclass(frozen=True)
class ExperimentPreset:
    name: str
    encoder_name: str
    max_tokens: int
    overlap_tokens: int
    single_sentence_chunks: bool = False
    model_options: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""


PRESETS = {
    "deeplrn": ExperimentPreset(
        "deeplrn",
        "roberta-base",
        384,
        64,
        False,
        {},
        "Full cross-chunk, layout-fused, multi-task model.",
    ),
    "sentence_roberta": ExperimentPreset(
        "sentence_roberta",
        "roberta-base",
        384,
        0,
        True,
        {"use_document_context": False, "use_layout": False},
        "Local RoBERTa control without cross-chunk or layout information.",
    ),
    "longformer": ExperimentPreset(
        "longformer",
        "allenai/longformer-base-4096",
        4096,
        256,
        False,
        {"use_document_context": False, "use_layout": False, "max_chunks": 32},
        "Long-context text baseline without page-layout features.",
    ),
    "layoutlmv3": ExperimentPreset(
        "layoutlmv3",
        "microsoft/layoutlmv3-base",
        512,
        64,
        False,
        {
            "encoder_uses_2d_positions": True,
            "use_document_context": False,
            "use_layout": False,
        },
        "LayoutLMv3 text-and-box baseline; pixels are intentionally omitted for a fair OCR-text input.",
    ),
}


def get_preset(name: str) -> ExperimentPreset:
    try:
        return PRESETS[name]
    except KeyError as error:
        raise ValueError(f"unknown experiment preset: {name}") from error


def _records_for_split(manifest: dict, manifest_path: Path, split: str) -> list[dict]:
    records = []
    for item in manifest.get("documents", []):
        if item.get("split") != split:
            continue
        path = Path(item["source_path"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with path.open("r", encoding="utf-8") as stream:
            records.append(json.load(stream))
    return records


def train_tfidf_svm(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    max_features: int = 100_000,
    c: float = 1.0,
    seed: int = 42,
    allow_weak_supervision: bool = False,
) -> dict[str, float]:
    """Train and evaluate the proposal's TF-IDF plus linear-SVM baseline."""

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    training = _records_for_split(manifest, manifest_path, "train")
    validation = _records_for_split(manifest, manifest_path, "validation")
    if not training or not validation:
        raise ValueError("the manifest needs non-empty train and validation splits")
    weak_count = sum(
        record.get("annotation_metadata", {}).get("supervision_quality")
        == "machine_draft"
        for record in training + validation
    )
    if weak_count and not allow_weak_supervision:
        raise ValueError(
            f"{weak_count} train/validation observations use machine-draft labels; "
            "pass --allow-weak-supervision only for an explicitly weak pilot"
        )

    finding_to_id = {label: index for index, label in enumerate(FINDING_LABELS)}

    def targets(record: dict) -> list[int]:
        if "finding_label_vector" in record:
            vector = [int(value) for value in record["finding_label_vector"]]
        else:
            labels = record.get("finding_labels")
            if labels is None:
                labels = [record["finding_label"]]
            vector = [0] * len(FINDING_LABELS)
            for label in labels:
                vector[finding_to_id[label]] = 1
        if len(vector) != len(FINDING_LABELS):
            raise ValueError("finding label vector has the wrong length")
        return vector

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=max_features,
                    sublinear_tf=True,
                ),
            ),
            (
                "svm",
                OneVsRestClassifier(
                    LinearSVC(C=c, random_state=seed, class_weight="balanced")
                ),
            ),
        ]
    )
    pipeline.fit(
        [record["document_text"] for record in training],
        [targets(record) for record in training],
    )
    gold = [targets(record) for record in validation]
    predictions = pipeline.predict(
        [record["document_text"] for record in validation]
    ).tolist()
    metrics = multilabel_classification_metrics(
        predictions, gold, len(FINDING_LABELS)
    )
    metrics["accuracy"] = metrics["subset_accuracy"]

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "finding_labels": list(FINDING_LABELS),
            "seed": seed,
            "metrics": metrics,
        },
        target,
    )
    target.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    return metrics


def experiments_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    list_parser = subparsers.add_parser("list", help="Print neural experiment presets")
    list_parser.add_argument("--output", help="Optional JSON output path")
    svm_parser = subparsers.add_parser("tfidf-svm", help="Train the linear baseline")
    svm_parser.add_argument("--manifest", required=True)
    svm_parser.add_argument("--output", required=True)
    svm_parser.add_argument("--max-features", type=int, default=100_000)
    svm_parser.add_argument("--c", type=float, default=1.0)
    svm_parser.add_argument("--seed", type=int, default=42)
    svm_parser.add_argument("--allow-weak-supervision", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "list":
        payload = {name: asdict(preset) for name, preset in PRESETS.items()}
        rendered = json.dumps(payload, indent=2)
        if args.output:
            Path(args.output).write_text(rendered, encoding="utf-8")
        else:
            print(rendered)
        return
    metrics = train_tfidf_svm(
        args.manifest,
        args.output,
        max_features=args.max_features,
        c=args.c,
        seed=args.seed,
        allow_weak_supervision=args.allow_weak_supervision,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
