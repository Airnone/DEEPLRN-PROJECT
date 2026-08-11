"""Build an exactly sized mixed reviewed/unreviewed observation corpus."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

DEFAULT_REVIEWED_DIR = Path("output/annotations/reviewed_findings_v2")
DEFAULT_CANDIDATE_PATHS = (
    Path("output/observation_candidates/pilot_observations.jsonl"),
    Path("output/observation_candidates/expansion_observations.jsonl"),
)
DEFAULT_OUTPUT = Path("output/observation_candidates/deeplrn_300_observations.jsonl")
DEFAULT_SUMMARY = Path("output/observation_candidates/deeplrn_300_observations.summary.json")
DEFAULT_EXCLUDED_DOC_IDS = frozenset({"naga-city-2024"})


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _normalized_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _quality_flags(candidate: dict[str, Any]) -> list[str]:
    text = str(candidate.get("observation_text", "")).strip()
    recommendation = str(candidate.get("recommendation_text", "")).strip()
    page_start = int(candidate.get("page_start", 0))
    page_end = int(candidate.get("page_end", page_start))
    initial = text[:750].casefold()
    digit_tokens = len(re.findall(r"\b\d[\d,.]*\b", initial))
    word_tokens = max(1, len(re.findall(r"\b[a-z]{2,}\b", initial)))

    flags: list[str] = []
    if len(text) < 80:
        flags.append("short_observation_text")
    if len(text) > 20_000:
        flags.append("long_observation_text")
    if page_end - page_start + 1 > 10:
        flags.append("wide_page_span")
    if not recommendation:
        flags.append("missing_recommendation")
    if candidate.get("metadata", {}).get("source_text_quality") == "ocr_artifacts":
        flags.append("ocr_artifacts")
    if initial.startswith(
        ("capital/investment receipts", "capital /investment receipts", "non-tax revenue")
    ) or digit_tokens / word_tokens > 0.45:
        flags.append("probable_table_fragment")
    return flags


def _quality_score(candidate: dict[str, Any]) -> int:
    penalties = {
        "probable_table_fragment": 80,
        "short_observation_text": 50,
        "long_observation_text": 25,
        "wide_page_span": 20,
        "missing_recommendation": 5,
        "ocr_artifacts": 10,
    }
    return 100 - sum(penalties[flag] for flag in _quality_flags(candidate))


def _reviewed_record(annotation: dict[str, Any], source_path: Path) -> dict[str, Any]:
    pages = tuple(int(page) for page in annotation.get("evidence_page_numbers", []))
    metadata = dict(annotation.get("metadata", {}))
    metadata.update(
        {
            "source_annotation_file": source_path.as_posix(),
            "annotation_schema_version": annotation.get("schema_version"),
            "requires_boundary_review": False,
            "selection_quality_flags": [],
        }
    )
    return {
        "observation_id": annotation["doc_id"],
        "doc_id": annotation.get("parent_doc_id", annotation["doc_id"]),
        "source_pdf": annotation["source_pdf"],
        "lgu": annotation["lgu"],
        "year": int(annotation["year"]),
        "section": "human_reviewed_finding",
        "source_item": None,
        "page_start": min(pages) if pages else None,
        "page_end": max(pages) if pages else None,
        "page_numbers": list(pages),
        "observation_text": annotation["observation_text"],
        "recommendation_text": annotation.get("recommendation_text", ""),
        "review_status": "reviewed",
        "finding_labels": list(annotation.get("finding_labels", [])),
        "entities": list(annotation.get("entities", [])),
        "relations": list(annotation.get("relations", [])),
        "schema_version": 1,
        "metadata": metadata,
    }


def load_reviewed_records(reviewed_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(reviewed_dir.glob("*.json")):
        annotation = json.loads(path.read_text(encoding="utf-8"))
        if "observation_text" not in annotation:
            continue
        records.append(_reviewed_record(annotation, path))
    return records


def select_unreviewed_candidates(
    candidate_paths: Sequence[Path],
    *,
    count: int,
    excluded_doc_ids: frozenset[str] = DEFAULT_EXCLUDED_DOC_IDS,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    unique: list[tuple[int, dict[str, Any]]] = []
    seen: set[tuple[str, int, str]] = set()
    excluded_counts: Counter[str] = Counter()
    ordinal = 0

    for path in candidate_paths:
        for candidate in _read_jsonl(path):
            if candidate["doc_id"] in excluded_doc_ids:
                excluded_counts["known_reviewed_source_overlap"] += 1
                continue
            key = (
                str(candidate["lgu"]).casefold(),
                int(candidate["year"]),
                _normalized_text(str(candidate["observation_text"])),
            )
            if key in seen:
                excluded_counts["normalized_duplicate"] += 1
                continue
            seen.add(key)
            unique.append((ordinal, candidate))
            ordinal += 1

    if len(unique) < count:
        raise ValueError(f"need {count} unreviewed candidates but only {len(unique)} are available")

    ranked = sorted(unique, key=lambda item: (-_quality_score(item[1]), item[0]))
    selected_ordinals = {ordinal for ordinal, _candidate in ranked[:count]}
    excluded_counts["quality_cap"] = len(unique) - count

    selected: list[dict[str, Any]] = []
    for ordinal, candidate in unique:
        if ordinal not in selected_ordinals:
            continue
        record = dict(candidate)
        flags = _quality_flags(record)
        metadata = dict(record.get("metadata", {}))
        metadata.update(
            {
                "requires_boundary_review": bool(flags),
                "selection_quality_flags": flags,
                "selection_quality_score": _quality_score(record),
            }
        )
        record["metadata"] = metadata
        record["review_status"] = "unreviewed"
        selected.append(record)
    return selected, dict(excluded_counts)


def build_corpus(
    reviewed_dir: Path,
    candidate_paths: Sequence[Path],
    *,
    total: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    reviewed = load_reviewed_records(reviewed_dir)
    unreviewed_count = total - len(reviewed)
    if unreviewed_count < 0:
        raise ValueError(f"reviewed records ({len(reviewed)}) exceed requested total ({total})")
    unreviewed, excluded = select_unreviewed_candidates(
        candidate_paths,
        count=unreviewed_count,
    )
    records = reviewed + unreviewed
    if len(records) != total:
        raise AssertionError(f"expected {total} records, assembled {len(records)}")

    doc_counts = Counter(record["doc_id"] for record in records)
    flag_counts = Counter(
        flag
        for record in unreviewed
        for flag in record.get("metadata", {}).get("selection_quality_flags", [])
    )
    boundary_review_count = sum(
        bool(record.get("metadata", {}).get("requires_boundary_review"))
        for record in unreviewed
    )
    summary = {
        "schema_version": 1,
        "requested_observations": total,
        "observations": len(records),
        "reviewed_observations": len(reviewed),
        "unreviewed_observations": len(unreviewed),
        "training_ready": False,
        "training_blocker": (
            "The unreviewed observations require human boundary and finding-label review. "
            "Do not materialize them as training records yet."
        ),
        "test_split_accessed": False,
        "candidate_sources": [path.as_posix() for path in candidate_paths],
        "reviewed_source": reviewed_dir.as_posix(),
        "excluded_candidates": excluded,
        "unreviewed_boundary_review_required": boundary_review_count,
        "selected_quality_flags": dict(sorted(flag_counts.items())),
        "document_counts": dict(sorted(doc_counts.items())),
        "interpretation_notice": (
            "Unreviewed records are structural audit-observation candidates, not labels or "
            "determinations of misconduct, intent, liability, or guilt."
        ),
    }
    return records, summary


def _write_jsonl(records: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewed-dir", type=Path, default=DEFAULT_REVIEWED_DIR)
    parser.add_argument(
        "--candidates",
        type=Path,
        nargs="+",
        default=list(DEFAULT_CANDIDATE_PATHS),
    )
    parser.add_argument("--total", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()

    records, summary = build_corpus(args.reviewed_dir, args.candidates, total=args.total)
    _write_jsonl(records, args.output)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {len(records)} observations: {summary['reviewed_observations']} reviewed, "
        f"{summary['unreviewed_observations']} unreviewed -> {args.output}"
    )


if __name__ == "__main__":
    main()
