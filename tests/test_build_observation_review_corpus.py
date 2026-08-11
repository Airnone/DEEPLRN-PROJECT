from __future__ import annotations

import json

from scripts.build_observation_review_corpus import build_corpus


def _candidate(doc_id: str, text: str, *, recommendation: str = "Review it."):
    return {
        "observation_id": f"{doc_id}-{text[:5]}",
        "doc_id": doc_id,
        "source_pdf": f"{doc_id}.pdf",
        "lgu": doc_id,
        "year": 2024,
        "section": "significant_observations",
        "source_item": "1",
        "page_start": 1,
        "page_end": 1,
        "page_numbers": [1],
        "observation_text": text,
        "recommendation_text": recommendation,
        "review_status": "unreviewed",
        "metadata": {},
    }


def test_builds_exact_cap_and_excludes_overlap_duplicates_and_lowest_quality(tmp_path):
    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    (reviewed_dir / "reviewed.json").write_text(
        json.dumps(
            {
                "doc_id": "reviewed-obs-001",
                "parent_doc_id": "reviewed",
                "source_pdf": "reviewed.pdf",
                "lgu": "Reviewed City",
                "year": 2020,
                "observation_text": "A reviewed observation.",
                "recommendation_text": "A reviewed recommendation.",
                "evidence_page_numbers": [2],
                "finding_labels": ["other_audit_finding"],
                "entities": [],
                "relations": [],
                "schema_version": 2,
                "metadata": {"supervision_quality": "human_adjudicated"},
            }
        ),
        encoding="utf-8",
    )
    candidate_path = tmp_path / "candidates.jsonl"
    candidates = [
        _candidate("good-a", "A" * 120),
        _candidate("good-b", "B" * 120),
        _candidate("short", "Too short"),
        _candidate("good-a", "A" * 120),
        _candidate("naga-city-2024", "C" * 120),
    ]
    candidate_path.write_text(
        "".join(json.dumps(candidate) + "\n" for candidate in candidates),
        encoding="utf-8",
    )

    records, summary = build_corpus(reviewed_dir, [candidate_path], total=3)

    assert len(records) == 3
    assert summary["reviewed_observations"] == 1
    assert summary["unreviewed_observations"] == 2
    assert {record["doc_id"] for record in records} == {"reviewed", "good-a", "good-b"}
    assert summary["excluded_candidates"] == {
        "normalized_duplicate": 1,
        "known_reviewed_source_overlap": 1,
        "quality_cap": 1,
    }
    assert summary["test_split_accessed"] is False
