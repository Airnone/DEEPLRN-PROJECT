from __future__ import annotations

import pytest

from deeplrn.schema import AnnotatedDocument, label_schema


def valid_document():
    return {
        "schema_version": 1,
        "doc_id": "sample-2023",
        "source_pdf": "sample.pdf",
        "lgu": "Sample LGU",
        "year": 2023,
        "finding_label": "procurement_irregularity",
        "entities": [
            {
                "entity_id": "e1",
                "label": "VIOLATION",
                "start_char": 0,
                "end_char": 11,
                "text": "unsupported",
                "page_number": 1,
            },
            {
                "entity_id": "e2",
                "label": "ORGANIZATION",
                "start_char": 20,
                "end_char": 31,
                "text": "Example Inc",
                "page_number": 1,
            },
        ],
        "relations": [
            {
                "relation_id": "r1",
                "head_entity_id": "e2",
                "tail_entity_id": "e1",
                "label": "RESPONSIBLE_FOR",
                "evidence_page_numbers": [1],
            }
        ],
    }


def test_legacy_labels_are_normalized_to_neutral_schema():
    document = AnnotatedDocument.from_dict(valid_document())
    assert document.entities[0].label == "FINDING"
    assert document.relations[0].label == "ASSOCIATED_WITH"


def test_unknown_relation_endpoint_is_rejected():
    raw = valid_document()
    raw["relations"][0]["tail_entity_id"] = "missing"
    with pytest.raises(ValueError, match="unknown entities"):
        AnnotatedDocument.from_dict(raw)


def test_overlapping_entities_are_rejected():
    raw = valid_document()
    raw["entities"][1]["start_char"] = 5
    with pytest.raises(ValueError, match="overlapping entities"):
        AnnotatedDocument.from_dict(raw)


def test_label_schema_is_self_describing():
    schema = label_schema()
    assert schema["schema_version"] == 2
    assert schema["finding_task"] == "multi_label"
    assert "FINDING" in schema["entity_types"]
    assert "ASSOCIATED_WITH" in schema["relation_types"]


def test_observation_schema_accepts_multiple_labels_and_migrates_v1():
    raw = valid_document()
    raw.pop("finding_label")
    raw["finding_labels"] = [
        "asset_record_reconciliation",
        "procurement_irregularity",
    ]
    raw["observation_text"] = "Unsupported payment to Example Inc."
    raw["recommendation_text"] = "Reconcile the records."
    document = AnnotatedDocument.from_dict(raw)
    assert document.schema_version == 2
    assert document.finding_labels == (
        "asset_record_reconciliation",
        "procurement_irregularity",
    )
    assert "Recommendation:" in document.training_text

    legacy = AnnotatedDocument.from_dict(valid_document())
    assert legacy.finding_labels == ("procurement_irregularity",)
    assert legacy.metadata["migrated_from_schema_version"] == 1


def test_empty_labels_require_explicit_reviewed_negative():
    raw = valid_document()
    raw.pop("finding_label")
    raw["schema_version"] = 2
    raw["finding_labels"] = []
    with pytest.raises(ValueError, match="reviewed_no_finding"):
        AnnotatedDocument.from_dict(raw)
    raw["metadata"] = {"reviewed_no_finding": True}
    assert AnnotatedDocument.from_dict(raw).finding_labels == ()
