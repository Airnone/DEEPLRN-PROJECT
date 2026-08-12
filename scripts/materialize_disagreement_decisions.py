"""Materialize PDF-verified disagreement decisions without touching test records."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from deeplrn.config import ChunkConfig
from deeplrn.preprocessing.pdf_extractor import PageData
from deeplrn.schema import FINDING_LABELS, AnnotatedDocument
from deeplrn.training.builder import TrainingRecordBuilder, save_training_record

ALLOWED_SPLITS = frozenset({"train", "validation"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _indexed(rows: list[dict[str, Any]], field: str, *, source: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {field} {key!r} in {source}")
        result[key] = row
    return result


def _annotation(
    source: dict[str, Any],
    candidate: dict[str, Any],
    decision: dict[str, Any],
) -> AnnotatedDocument:
    labels = tuple(str(label) for label in decision.get("finding_labels", []))
    unsupported = sorted(set(labels) - set(FINDING_LABELS))
    if unsupported:
        raise ValueError(f"unsupported labels for {source['doc_id']}: {unsupported}")
    if not labels:
        raise ValueError(
            f"reviewed-no-finding record {source['doc_id']} must be excluded, not built"
        )

    metadata = dict(source.get("annotation_metadata", {}))
    metadata.update(
        {
            "supervision_quality": "human_adjudicated",
            "source_review_queue": str(candidate.get("review_queue", "pdf_verified")),
            "reviewed_no_finding": False,
            "finding_decision": str(decision["decision"]),
            "boundary_status": "pdf_verified",
            "adjudication_reason": str(decision.get("notes", "")),
            "reviewer": str(decision.get("reviewer", "project_owner")),
            "reviewed_at": str(decision.get("reviewed_at", "")),
            "pdf_text_supported": bool(candidate.get("pdf_text_supported", False)),
        }
    )
    return AnnotatedDocument(
        doc_id=str(source["doc_id"]),
        parent_doc_id=str(source.get("parent_doc_id", source["doc_id"])),
        source_pdf=str(source["source_pdf"]),
        lgu=str(source["lgu"]),
        year=int(source["year"]),
        finding_labels=labels,
        observation_text=str(candidate["observation_text"]).strip(),
        recommendation_text="",
        evidence_page_numbers=tuple(
            int(page) for page in candidate.get("pdf_page_numbers_verified", [])
        ),
        metadata=metadata,
    )


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.resolve()
    queue_path = args.queue.resolve()
    decisions_path = args.decisions.resolve()
    output_records = args.output_records.resolve()
    output_manifest = args.output_manifest.resolve()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    queue = _indexed(_read_jsonl(queue_path), "doc_id", source=queue_path)
    decisions = _indexed(_read_jsonl(decisions_path), "doc_id", source=decisions_path)

    if set(queue) != set(decisions):
        missing = sorted(set(queue) - set(decisions))
        extra = sorted(set(decisions) - set(queue))
        raise ValueError(f"queue/decision mismatch; missing={missing}, extra={extra}")
    if any(str(row.get("split")) not in ALLOWED_SPLITS for row in queue.values()):
        raise ValueError("review queue contains a split other than train/validation")
    if any(not bool(row.get("pdf_text_supported")) for row in queue.values()):
        raise ValueError("review queue contains text not verified against its PDF")
    if output_records.exists() and any(output_records.iterdir()):
        raise FileExistsError(f"output record directory is not empty: {output_records}")
    if output_manifest.exists():
        raise FileExistsError(f"output manifest already exists: {output_manifest}")

    output_records.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, use_fast=True, local_files_only=args.local_files_only
    )
    builder = TrainingRecordBuilder(
        tokenizer,
        chunk_config=ChunkConfig(
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap,
            tokenizer_name=args.tokenizer,
        ),
    )

    output_documents: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    excluded: list[str] = []
    manifest_ids: set[str] = set()

    for entry in manifest.get("documents", []):
        split = str(entry.get("split"))
        if split not in ALLOWED_SPLITS:
            # Strict boundary: never open, copy, tokenize, or otherwise inspect a test record.
            continue
        doc_id = str(entry["doc_id"])
        manifest_ids.add(doc_id)
        candidate = queue.get(doc_id)
        decision = decisions.get(doc_id)
        if (candidate is None) != (decision is None):
            raise ValueError(f"incomplete review material for {doc_id}")
        if candidate is not None and str(candidate["split"]) != split:
            raise ValueError(f"split mismatch for {doc_id}: {candidate['split']} != {split}")
        if decision is not None and bool(decision.get("reviewed_no_finding")):
            if decision.get("finding_labels"):
                raise ValueError(f"no-finding decision has labels for {doc_id}")
            excluded.append(doc_id)
            continue

        source_path = Path(str(entry["source_path"]))
        if not source_path.is_absolute():
            source_path = manifest_path.parent / source_path
        target_path = output_records / f"{doc_id}.json"

        if decision is None:
            shutil.copy2(source_path, target_path)
            source_counts["unchanged"] += 1
        else:
            source = json.loads(source_path.read_text(encoding="utf-8"))
            annotation = _annotation(source, candidate, decision)
            page_number = annotation.evidence_page_numbers[0] if annotation.evidence_page_numbers else 1
            record = builder.build(
                [
                    PageData(
                        page_number=page_number,
                        text=annotation.training_text,
                        source="pdf_verified_review",
                        metadata={
                            "evidence_page_numbers": list(annotation.evidence_page_numbers)
                        },
                    )
                ],
                annotation,
            )
            save_training_record(record, target_path)
            source_counts["adjudicated"] += 1

        updated = dict(entry)
        updated["source_path"] = str(target_path)
        output_documents.append(updated)
        split_counts[split] += 1

    queue_outside_manifest = sorted(set(queue) - manifest_ids)
    if queue_outside_manifest:
        raise ValueError(
            "review queue contains records outside train/validation manifest entries: "
            f"{queue_outside_manifest}"
        )

    result_manifest = {
        "source_manifest": str(manifest_path),
        "split_assignments_preserved": True,
        "test_split_included": False,
        "test_split_accessed": False,
        "review_queue": str(queue_path),
        "review_decisions": str(decisions_path),
        "chunk_config": {
            "max_tokens": args.max_tokens,
            "overlap_tokens": args.overlap,
            "tokenizer_name": args.tokenizer,
        },
        "counts": dict(sorted(split_counts.items())),
        "excluded_reviewed_no_finding": sorted(excluded),
        "documents": output_documents,
    }
    output_manifest.write_text(
        json.dumps(result_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "output_manifest": str(output_manifest),
        "output_records": str(output_records),
        "counts": dict(sorted(split_counts.items())),
        "adjudicated_records": source_counts["adjudicated"],
        "unchanged_records": source_counts["unchanged"],
        "excluded_reviewed_no_finding": sorted(excluded),
        "test_split_accessed": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--queue", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--output-records", required=True, type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--tokenizer", default="roberta-base")
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    print(json.dumps(materialize(parse_args()), indent=2, ensure_ascii=False))
