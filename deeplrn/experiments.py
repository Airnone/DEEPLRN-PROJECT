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

from deeplrn.evaluation import classification_metrics
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
) -> dict[str, float]:
    """Train and evaluate the proposal's TF-IDF plus linear-SVM baseline."""

    manifest_path = Path(manifest_path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    training = _records_for_split(manifest, manifest_path, "train")
    validation = _records_for_split(manifest, manifest_path, "validation")
    if not training or not validation:
        raise ValueError("the manifest needs non-empty train and validation splits")

    finding_to_id = {label: index for index, label in enumerate(FINDING_LABELS)}
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
            ("svm", LinearSVC(C=c, random_state=seed, class_weight="balanced")),
        ]
    )
    pipeline.fit(
        [record["document_text"] for record in training],
        [finding_to_id[record["finding_label"]] for record in training],
    )
    gold = [finding_to_id[record["finding_label"]] for record in validation]
    predictions = pipeline.predict(
        [record["document_text"] for record in validation]
    ).tolist()
    metrics = classification_metrics(predictions, gold, len(FINDING_LABELS))

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
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
