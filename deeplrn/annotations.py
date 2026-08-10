"""Materialize observation overlays into schema-v2 annotation records."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from deeplrn.schema import FINDING_LABELS, AnnotatedDocument

ISSUE_FAMILY_LABELS: dict[str, tuple[str, ...]] = {
    "asset_record_reconciliation": ("asset_record_reconciliation",),
    "asset_and_inventory_record_reconciliation": (
        "asset_record_reconciliation",
        "inventory_count_or_record",
    ),
    "asset_accounting_error": ("asset_record_reconciliation",),
    "asset_depreciation_recording": ("asset_record_reconciliation",),
    "asset_disposal_delay": ("asset_record_reconciliation",),
    "construction_in_progress_accounting": ("asset_record_reconciliation",),
    "cash_or_bank_reconciliation": ("cash_or_bank_reconciliation",),
    "cash_or_bank_compliance": ("cash_or_bank_reconciliation",),
    "cashbook_reconciliation": ("cash_or_bank_reconciliation",),
    "cash_deficit": ("cash_or_bank_reconciliation",),
    "inventory_count_or_record": ("inventory_count_or_record",),
    "inventory_recording": ("inventory_count_or_record",),
    "inventory_recording_and_reporting": ("inventory_count_or_record",),
    "inventory_documentation_and_recording": ("inventory_count_or_record",),
    "program_fund_utilization": ("fund_utilization_or_liquidation",),
    "transferred_fund_recording_utilization_and_liquidation": ("fund_utilization_or_liquidation",),
    "donation_funded_program_documentation": ("fund_utilization_or_liquidation",),
    "disaster_fund_planning_and_reversion": ("fund_utilization_or_liquidation",),
    "unliquidated_cash_advance": ("unliquidated_cash_advance",),
    "procurement_irregularity": ("procurement_irregularity",),
    "unsupported_disbursement": ("unsupported_disbursement",),
    "unauthorized_expenditure": ("unauthorized_expenditure",),
    "contractor_related_concern": ("contractor_related_concern",),
    "unsupported_or_unverified_balance": ("other_control_or_compliance_observation",),
    "tax_share_accounting": ("other_control_or_compliance_observation",),
    "revenue_code_governance": ("other_control_or_compliance_observation",),
    "revenue_generation_policy": ("other_control_or_compliance_observation",),
    "revenue_reporting_classification": ("other_control_or_compliance_observation",),
    "tax_receipt_control": ("other_control_or_compliance_observation",),
    "tax_collection_without_legal_basis": ("other_control_or_compliance_observation",),
    "tax_rate_application": ("other_control_or_compliance_observation",),
}


def _load_candidates(path: str | Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            candidate = json.loads(line)
            observation_id = str(candidate["observation_id"])
            if observation_id in candidates:
                raise ValueError(f"duplicate observation ID {observation_id!r}")
            candidates[observation_id] = candidate
    if not candidates:
        raise ValueError("candidate JSONL is empty")
    return candidates


def _overlay_entries(
    overlay: Mapping[str, Any],
) -> Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]]:
    if overlay.get("annotations") is not None:
        context = {
            "parent_doc_id": overlay.get("parent_doc_id", ""),
            "source_pdf": overlay.get("source_pdf", ""),
            "lgu": overlay.get("lgu", ""),
            "year": overlay.get("year"),
        }
        for annotation in overlay["annotations"]:
            yield context, annotation
    for document in overlay.get("documents", []):
        for annotation in document.get("annotations", []):
            yield document, annotation


def _labels_for(annotation: Mapping[str, Any]) -> tuple[str, ...]:
    explicit = annotation.get("finding_labels")
    if explicit is not None:
        return tuple(str(label) for label in explicit)
    proposed = annotation.get("proposed_finding_label")
    if proposed:
        return (str(proposed),)
    family = str(annotation.get("candidate_issue_family", ""))
    if family == "positive_performance":
        return ()
    return ISSUE_FAMILY_LABELS.get(family, ("other_control_or_compliance_observation",))


def materialize_overlays(
    candidates_path: str | Path,
    overlay_paths: Iterable[str | Path],
    output_dir: str | Path,
    *,
    allow_machine_drafts: bool = False,
) -> dict[str, Any]:
    """Join candidate text with label overlays and write validated annotations."""

    candidates = _load_candidates(candidates_path)
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    skipped: list[dict[str, str]] = []
    supervision_counts: dict[str, int] = {}

    for overlay_path in overlay_paths:
        path = Path(overlay_path)
        overlay = json.loads(path.read_text(encoding="utf-8"))
        is_training_eligible = bool(overlay.get("training_eligible", False))
        if not is_training_eligible and not allow_machine_drafts:
            raise ValueError(
                f"{path} is not training-eligible; adjudicate it or pass "
                "--allow-machine-drafts for an explicitly weak-supervision run"
            )
        supervision = "human_adjudicated" if is_training_eligible else "machine_draft"
        for context, raw_annotation in _overlay_entries(overlay):
            observation_id = str(raw_annotation["observation_id"])
            if observation_id not in candidates:
                raise ValueError(f"overlay references unknown observation {observation_id!r}")
            labels = _labels_for(raw_annotation)
            unknown = sorted(set(labels) - set(FINDING_LABELS))
            if unknown:
                raise ValueError(f"unsupported labels for {observation_id}: {unknown}")
            candidate = candidates[observation_id]
            document = AnnotatedDocument.from_dict(
                {
                    "schema_version": 2,
                    "doc_id": observation_id,
                    "parent_doc_id": context.get("parent_doc_id") or candidate["doc_id"],
                    "source_pdf": candidate.get("source_pdf") or context.get("source_pdf"),
                    "lgu": candidate.get("lgu") or context.get("lgu"),
                    "year": candidate.get("year") or context.get("year"),
                    "observation_text": candidate["observation_text"],
                    "recommendation_text": candidate.get("recommendation_text", ""),
                    "evidence_page_numbers": raw_annotation.get(
                        "evidence_page_numbers", candidate.get("page_numbers", [])
                    ),
                    "finding_labels": labels,
                    "entities": raw_annotation.get("entities", []),
                    "relations": raw_annotation.get("relations", []),
                    "metadata": {
                        "supervision_quality": supervision,
                        "source_overlay": str(path),
                        "candidate_issue_family": raw_annotation.get("candidate_issue_family", ""),
                        "label_confidence": raw_annotation.get("confidence"),
                        "human_review_required": raw_annotation.get(
                            "human_review_required", not is_training_eligible
                        ),
                        "ner_reviewed": raw_annotation.get("ner_reviewed", False),
                        "relations_reviewed": raw_annotation.get("relations_reviewed", False),
                        "reviewed_no_finding": not labels,
                    },
                }
            )
            output_path = target_dir / f"{observation_id}.json"
            output_path.write_text(
                json.dumps(document.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            written.append(str(output_path))
            supervision_counts[supervision] = supervision_counts.get(supervision, 0) + 1

    summary = {
        "schema_version": 2,
        "candidates_path": str(candidates_path),
        "written": len(written),
        "skipped": skipped,
        "supervision_counts": supervision_counts,
        "finding_labels": list(FINDING_LABELS),
        "annotation_files": written,
    }
    (target_dir / "materialization_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def annotations_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--overlays", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-machine-drafts", action="store_true")
    args = parser.parse_args(argv)
    summary = materialize_overlays(
        args.candidates,
        args.overlays,
        args.output,
        allow_machine_drafts=args.allow_machine_drafts,
    )
    print(json.dumps(summary, indent=2))
