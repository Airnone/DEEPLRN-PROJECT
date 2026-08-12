from __future__ import annotations

import json

import pytest

from deeplrn.annotations import materialize_overlays
from deeplrn.schema import load_annotation


def test_materializer_marks_machine_drafts_and_joins_candidate_text(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "observation_id": "sample-obs-001",
                "doc_id": "sample",
                "source_pdf": "sample.pdf",
                "lgu": "Sample LGU",
                "year": 2023,
                "page_numbers": [3],
                "observation_text": "PPE records were not reconciled.",
                "recommendation_text": "Complete the reconciliation.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "draft.json"
    overlay.write_text(
        json.dumps(
            {
                "training_eligible": False,
                "annotations": [
                    {
                        "observation_id": "sample-obs-001",
                        "candidate_issue_family": "asset_record_reconciliation",
                        "entities": [],
                        "relations": [],
                        "human_review_required": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not training-eligible"):
        materialize_overlays(candidates, [overlay], tmp_path / "blocked")

    output = tmp_path / "weak"
    summary = materialize_overlays(candidates, [overlay], output, allow_machine_drafts=True)
    assert summary["written"] == 1
    annotation = load_annotation(output / "sample-obs-001.json")
    assert annotation.finding_labels == ("asset_record_reconciliation",)
    assert annotation.metadata["supervision_quality"] == "machine_draft"
    assert annotation.observation_text.startswith("PPE records")


def test_materializer_accepts_review_decisions_and_applies_audited_text_edits(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "observation_id": "sample-obs-001",
                "doc_id": "sample",
                "source_pdf": "sample.pdf",
                "lgu": "Sample LGU",
                "year": 2023,
                "page_numbers": [3],
                "observation_text": "Managementâ€™s records were not reconciled. iv",
                "recommendation_text": "Reconcile them. Exit-conference boilerplate.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    review_log = tmp_path / "review-log.json"
    review_log.write_text(
        json.dumps(
            {
                "training_eligible": True,
                "training_eligible_tasks": ["finding"],
                "repair_mojibake": True,
                "text_corrections": {
                    "sample-obs-001": {
                        "observation_text_edits": [
                            {"operation": "remove_literal", "text": " iv"}
                        ],
                        "recommendation_text_edits": [
                            {
                                "operation": "truncate_before",
                                "text": " Exit-conference boilerplate.",
                            }
                        ],
                    }
                },
                "decisions": [
                    {
                        "observation_id": "sample-obs-001",
                        "finding_labels": ["asset_record_reconciliation"],
                        "finding_decision": "approved",
                        "boundary_status": "approved_with_correction",
                        "ner_reviewed": False,
                        "relations_reviewed": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "reviewed"
    summary = materialize_overlays(candidates, [review_log], output)
    annotation = load_annotation(output / "sample-obs-001.json")

    assert summary["training_eligible_tasks"] == ["finding"]
    assert annotation.observation_text == "Management’s records were not reconciled."
    assert annotation.recommendation_text == "Reconcile them."
    assert annotation.metadata["supervision_quality"] == "human_adjudicated"
    assert annotation.metadata["ner_reviewed"] is False
    assert annotation.metadata["relations_reviewed"] is False


def test_materializer_preserves_explicit_supervision_provenance(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "observation_id": "sample-obs-001",
                "doc_id": "sample",
                "source_pdf": "sample.pdf",
                "lgu": "Sample LGU",
                "year": 2023,
                "page_numbers": [3],
                "observation_text": "The payment lacked support.",
                "recommendation_text": "Submit the supporting records.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    overlay = tmp_path / "blanket-approval.json"
    overlay.write_text(
        json.dumps(
            {
                "training_eligible": True,
                "training_eligible_tasks": ["finding"],
                "supervision_quality": "owner_blanket_approved_machine_labels",
                "decisions": [
                    {
                        "observation_id": "sample-obs-001",
                        "lgu": "Canonical Sample LGU",
                        "finding_labels": ["unsupported_disbursement"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "reviewed"
    summary = materialize_overlays(candidates, [overlay], output)
    annotation = load_annotation(output / "sample-obs-001.json")

    assert summary["supervision_counts"] == {
        "owner_blanket_approved_machine_labels": 1
    }
    assert (
        annotation.metadata["supervision_quality"]
        == "owner_blanket_approved_machine_labels"
    )
    assert annotation.lgu == "Canonical Sample LGU"


def test_materializer_rejects_missing_or_ambiguous_text_edit_markers(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        json.dumps(
            {
                "observation_id": "sample-obs-001",
                "doc_id": "sample",
                "source_pdf": "sample.pdf",
                "lgu": "Sample LGU",
                "year": 2023,
                "page_numbers": [3],
                "observation_text": "Repeated marker marker.",
                "recommendation_text": "Review it.",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    review_log = tmp_path / "review-log.json"
    review_log.write_text(
        json.dumps(
            {
                "training_eligible": True,
                "text_corrections": {
                    "sample-obs-001": {
                        "observation_text_edits": [
                            {"operation": "remove_literal", "text": "marker"}
                        ]
                    }
                },
                "decisions": [
                    {
                        "observation_id": "sample-obs-001",
                        "finding_labels": ["other_control_or_compliance_observation"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected one observation_text edit marker"):
        materialize_overlays(candidates, [review_log], tmp_path / "reviewed")
