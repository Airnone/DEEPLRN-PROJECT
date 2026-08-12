from __future__ import annotations

from scripts.bulk_approve_observations import SUPERVISION_QUALITY, build_overlay, labels_for


def _record(doc_id: str, number: int) -> dict[str, object]:
    return {
        "observation_id": f"{doc_id}-obs-{number:03d}",
        "doc_id": doc_id,
        "lgu": "Sample LGU",
        "review_status": "unreviewed",
    }


def test_rule_labels_cover_single_and_multilabel_findings():
    assert labels_for(_record("la-union-province-2024", 2)) == (
        "asset_record_reconciliation",
    )
    assert labels_for(_record("taytay-palawan-2020", 39)) == (
        "contractor_related_concern",
        "fund_utilization_or_liquidation",
    )
    assert labels_for(_record("asingan-2023", 10)) == ()


def test_overlay_records_blanket_approval_provenance():
    records = [_record("asingan-2022", 4) for _ in range(255)]

    overlay = build_overlay(records)

    assert overlay["reviewed_observations"] == 255
    assert overlay["supervision_quality"] == SUPERVISION_QUALITY
    assert overlay["decisions"][0]["finding_labels"] == []
    assert overlay["decisions"][0]["reviewed_no_finding"] is True
