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
