from __future__ import annotations

import json

import pytest

from deeplrn.preprocessing.observation_extractor import (
    ObservationCandidate,
    ObservationExtractor,
    extract_manifest,
)
from deeplrn.preprocessing.pdf_extractor import PageData


def extract(*pages: str):
    return ObservationExtractor().extract(
        [PageData(page_number=index + 1, text=text) for index, text in enumerate(pages)],
        doc_id="sample-2024",
        source_pdf="pdfs/sample.pdf",
        lgu="Sample City",
        year=2024,
    )


def test_extracts_numbered_observations_recommendations_and_page_provenance():
    candidates = extract(
        """Auditor's Opinion on the Financial Statements
We rendered a qualified opinion due to the following:
1. The cash balance could not be ascertained due to unreconciled records.
We recommended that Management:
(a) Reconcile the accounting and treasury records;""",
        """(b) Prepare the necessary adjusting entries.
2. Various cash advances remained unliquidated at year-end.
We recommended that Management require immediate liquidation.
H. Status of Implementation of Prior Years' Audit Recommendations""",
    )

    assert len(candidates) == 2
    first, second = candidates
    assert first.source_item == "1"
    assert first.page_numbers == (1, 2)
    assert first.page_start == 1
    assert first.page_end == 2
    assert "unreconciled records" in first.observation_text
    assert "adjusting entries" in first.recommendation_text
    assert second.source_item == "2"
    assert second.page_numbers == (2,)
    assert "Status of Implementation" not in second.recommendation_text


def test_lowercase_numbered_recommendations_do_not_start_new_observations():
    candidates = extract(
        """Audit Opinion on the Financial Statements
We rendered a qualified opinion due to the following observations:
1. The PPE records did not reconcile with the physical count.
2. The construction balance lacked supporting details.
For the above-cited deficiencies, we recommended that the City Mayor direct:
1.a. the Accounting Office to update its ledgers;
2. the City Accountant to analyze the account;
Other observations that need immediate attention and action are as follows:
1. Cash was insufficient to cover current liabilities.
We recommended that the Treasurer monitor cash availability.
Status of Implementation of Prior Year's Audit Recommendations"""
    )

    assert len(candidates) == 3
    assert "1.a. the Accounting Office" in candidates[0].recommendation_text
    assert "2. the City Accountant" not in candidates[0].recommendation_text
    assert "shared_recommendation_block" in candidates[0].warnings
    assert "2. the City Accountant" in candidates[1].recommendation_text
    assert "1.a. the Accounting Office" not in candidates[1].recommendation_text
    assert candidates[2].section == "significant_observations"
    assert candidates[2].source_item == "1"


def test_extracts_unnumbered_summary_exception_before_numbered_observations():
    candidates = extract(
        """III. Independent Auditor's Report on the Financial Statements
The Auditor rendered a qualified opinion because accounting and property records
of PPE and Inventory have a discrepancy of P100 and P20, respectively.
IV. Significant Audit Observations and Recommendations
For the exceptions cited above, we recommended that the City Mayor:
(a) Direct the reconciliation of property and accounting records;
(b) Complete the physical inventory.
The Audit Team communicated the observations to management.""",
        """1. The disaster fund plan was incomplete and inconsistent with regulations.
We recommended that Management prepare a comprehensive plan.
V. Other Matters""",
    )

    assert len(candidates) == 2
    first, second = candidates
    assert first.source_item is None
    assert first.section == "audit_opinion"
    assert "unnumbered_observation" in first.warnings
    assert "discrepancy" in first.observation_text
    assert "reconciliation" in first.recommendation_text
    assert "Audit Team communicated" not in first.recommendation_text
    assert second.source_item == "1"
    assert second.section == "significant_observations"


def test_splits_embedded_observation_recommendation_and_stop_boundaries():
    candidates = extract(
        """Audit Opinion on the Financial Statements
We rendered a qualified opinion due to the following:
1. The first balance was overstated. We recommended that Management adjust it.
2. The second balance was understated. We recommended that Management reconcile it.
V. Summary of Total Audit Suspensions and Disallowances"""
    )

    assert len(candidates) == 2
    assert candidates[0].observation_text == "The first balance was overstated."
    assert "adjust it" in candidates[0].recommendation_text
    assert candidates[1].observation_text == "The second balance was understated."
    assert "Summary of Total" not in candidates[1].recommendation_text


def test_candidates_are_unlabeled_and_keep_document_identity():
    candidate = extract(
        """Audit Opinion on the Financial Statements
We rendered a qualified opinion due to the following:
1. The reported balance was overstated.
We recommended that Management prepare an adjustment.
Status of Implementation of Prior Years' Audit Recommendations"""
    )[0]
    data = candidate.to_dict()

    assert data["review_status"] == "unreviewed"
    assert "finding_label" not in data
    assert data["doc_id"] == "sample-2024"
    assert data["lgu"] == "Sample City"
    assert data["year"] == 2024
    assert data["source_pdf"] == "pdfs/sample.pdf"


def test_manifest_continues_when_one_document_has_no_candidates(tmp_path, monkeypatch):
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "one.pdf").write_bytes(b"placeholder")
    (pdf_dir / "empty.pdf").write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "one-2024",
                        "lgu": "One City",
                        "year": 2024,
                        "local_path": "pdfs/one.pdf",
                    },
                    {
                        "doc_id": "empty-2024",
                        "lgu": "Empty City",
                        "year": 2024,
                        "local_path": "pdfs/empty.pdf",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "deeplrn.preprocessing.observation_extractor.PDFExtractor.extract",
        lambda _self: [PageData(page_number=1, text="extracted text")],
    )

    def fake_extract(_self, _pages, *, doc_id, source_pdf, lgu, year, metadata=None):
        if doc_id == "empty-2024":
            raise ValueError("no audit-opinion or significant-observation section was found")
        return [
            ObservationCandidate(
                observation_id="one-2024-observation-001",
                doc_id=doc_id,
                source_pdf=source_pdf,
                lgu=lgu,
                year=year,
                section="audit_opinion",
                source_item="1",
                page_start=1,
                page_end=1,
                page_numbers=(1,),
                observation_text="The balance was misstated.",
                metadata=dict(metadata or {}),
            )
        ]

    monkeypatch.setattr(ObservationExtractor, "extract", fake_extract)

    candidates, summary = extract_manifest(manifest_path)

    assert [candidate.doc_id for candidate in candidates] == ["one-2024"]
    assert summary["documents"] == 2
    assert summary["observation_candidates"] == 1
    assert summary["document_summaries"][0]["extraction_status"] == "extracted"
    assert summary["document_summaries"][1]["extraction_status"] == "no_candidates"
    assert summary["document_summaries"][1]["observation_candidates"] == 0


def test_manifest_does_not_swallow_unrelated_value_errors(tmp_path, monkeypatch):
    pdf_path = tmp_path / "broken.pdf"
    pdf_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "broken-2024",
                        "lgu": "Broken City",
                        "year": 2024,
                        "local_path": "broken.pdf",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "deeplrn.preprocessing.observation_extractor.PDFExtractor.extract",
        lambda _self: [PageData(page_number=1, text="extracted text")],
    )

    def fail_extract(*_args, **_kwargs):
        raise ValueError("unexpected parser failure")

    monkeypatch.setattr(ObservationExtractor, "extract", fail_extract)

    with pytest.raises(ValueError, match="unexpected parser failure"):
        extract_manifest(manifest_path)


def test_manifest_skips_known_image_only_document_without_ocr(tmp_path, monkeypatch):
    pdf_path = tmp_path / "scan.pdf"
    pdf_path.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "documents": [
                    {
                        "doc_id": "scan-2024",
                        "lgu": "Scan City",
                        "year": 2024,
                        "local_path": "scan.pdf",
                        "pages": 12,
                        "extraction_profile": "image_only_ocr_required",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    def unexpected_extract(_self):
        pytest.fail("image-only PDF should not be opened when OCR is disabled")

    monkeypatch.setattr(
        "deeplrn.preprocessing.observation_extractor.PDFExtractor.extract",
        unexpected_extract,
    )

    candidates, summary = extract_manifest(manifest_path)

    assert candidates == []
    assert summary["document_summaries"][0]["pages"] == 12
    assert summary["document_summaries"][0]["extraction_status"] == "ocr_required"
