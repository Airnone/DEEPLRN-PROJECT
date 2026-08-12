"""Create a reproducible overlay for the owner's blanket corpus approval.

The project owner explicitly approved all remaining observation candidates and
their rule-based finding labels on 2026-08-11. This is weaker than item-by-item
adjudication, so the generated overlay records that provenance rather than
claiming conventional human adjudication.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from deeplrn.schema import FINDING_LABELS

DEFAULT_CORPUS = Path("output/observation_candidates/deeplrn_300_observations.jsonl")
DEFAULT_OUTPUT = Path(
    "output/annotations/bulk_owner_approved/bulk_approval_overlay.json"
)
SUPERVISION_QUALITY = "owner_blanket_approved_machine_labels"
CANONICAL_LGU_NAMES = {
    "City of Naga": "Naga City",
    "City of Pasig": "Pasig City",
}

Labels = tuple[str, ...]
RangeRule = tuple[int, int, Labels]


RULES: dict[str, tuple[RangeRule, ...]] = {
    "la-union-province-2024": (
        (1, 1, ("fund_utilization_or_liquidation",)),
        (2, 2, ("asset_record_reconciliation",)),
        (3, 3, ("other_control_or_compliance_observation",)),
        (4, 4, ("inventory_count_or_record",)),
        (5, 5, ("fund_utilization_or_liquidation",)),
        (6, 10, ("procurement_irregularity",)),
        (11, 14, ("contractor_related_concern",)),
        (15, 15, ("fund_utilization_or_liquidation",)),
        (16, 17, ("unauthorized_expenditure",)),
        (18, 25, ("other_control_or_compliance_observation",)),
    ),
    "molave-2021": (
        (1, 1, ("asset_record_reconciliation",)),
        (2, 2, ("inventory_count_or_record",)),
        (3, 3, ("other_control_or_compliance_observation",)),
        (4, 4, ("cash_or_bank_reconciliation",)),
        (5, 5, ("asset_record_reconciliation",)),
        (6, 6, ("cash_or_bank_reconciliation",)),
        (7, 8, ("asset_record_reconciliation",)),
    ),
    "molave-2022": (
        (1, 2, ("asset_record_reconciliation",)),
        (3, 3, ("inventory_count_or_record",)),
        (4, 4, ("other_control_or_compliance_observation",)),
        (5, 5, ("asset_record_reconciliation",)),
        (6, 6, ("other_control_or_compliance_observation",)),
        (7, 9, ("asset_record_reconciliation",)),
    ),
    "taytay-palawan-2020": (
        (8, 8, ("fund_utilization_or_liquidation",)),
        (9, 9, ("other_control_or_compliance_observation",)),
        (10, 20, ("asset_record_reconciliation",)),
        (21, 25, ("fund_utilization_or_liquidation",)),
        (26, 26, ("unsupported_disbursement",)),
        (27, 29, ("fund_utilization_or_liquidation",)),
        (30, 32, ("procurement_irregularity",)),
        (33, 34, ("fund_utilization_or_liquidation", "unsupported_disbursement")),
        (35, 37, ("contractor_related_concern",)),
        (38, 38, ("fund_utilization_or_liquidation",)),
        (39, 39, ("contractor_related_concern", "fund_utilization_or_liquidation")),
        (40, 40, ("unauthorized_expenditure",)),
        (41, 47, ("other_control_or_compliance_observation",)),
        (48, 48, ("unsupported_disbursement",)),
        (49, 51, ("procurement_irregularity", "unsupported_disbursement")),
        (52, 54, ("unsupported_disbursement",)),
        (55, 58, ("procurement_irregularity", "unsupported_disbursement")),
    ),
    "taytay-palawan-2021": (
        (1, 2, ("asset_record_reconciliation",)),
        (3, 3, ("fund_utilization_or_liquidation",)),
        (4, 6, ("fund_utilization_or_liquidation",)),
        (7, 7, ("procurement_irregularity",)),
        (8, 9, ("fund_utilization_or_liquidation",)),
        (10, 10, ("unauthorized_expenditure",)),
        (11, 12, ("contractor_related_concern", "procurement_irregularity")),
        (13, 13, ("procurement_irregularity", "unsupported_disbursement")),
        (14, 14, ("contractor_related_concern",)),
        (15, 15, ("other_control_or_compliance_observation",)),
        (16, 16, ("fund_utilization_or_liquidation",)),
    ),
    "asingan-2021": ((1, 3, ("asset_record_reconciliation",)),),
    "asingan-2022": (
        (4, 7, ()),
        (9, 11, ("other_control_or_compliance_observation",)),
        (12, 14, ("inventory_count_or_record",)),
        (15, 17, ("asset_record_reconciliation",)),
        (18, 18, ("asset_record_reconciliation", "contractor_related_concern")),
        (19, 26, ("other_control_or_compliance_observation",)),
        (27, 27, ("asset_record_reconciliation",)),
        (28, 35, ("unsupported_disbursement",)),
        (36, 38, ("procurement_irregularity",)),
        (39, 39, ("contractor_related_concern",)),
        (40, 53, ("fund_utilization_or_liquidation",)),
        (54, 54, ()),
        (55, 55, ("fund_utilization_or_liquidation",)),
        (56, 57, ("other_control_or_compliance_observation",)),
        (58, 59, ("unauthorized_expenditure",)),
        (60, 61, ("fund_utilization_or_liquidation",)),
        (62, 64, ()),
    ),
    "asingan-2023": (
        (4, 5, ()),
        (7, 7, ()),
        (9, 9, ("asset_record_reconciliation",)),
        (10, 12, ()),
        (13, 13, ("cash_or_bank_reconciliation",)),
        (14, 17, ("asset_record_reconciliation",)),
        (18, 18, ("inventory_count_or_record",)),
        (19, 27, ("cash_or_bank_reconciliation",)),
        (28, 28, ("unauthorized_expenditure",)),
        (29, 29, ("procurement_irregularity", "unsupported_disbursement")),
        (30, 52, ("procurement_irregularity",)),
        (53, 55, ("unsupported_disbursement",)),
        (56, 58, ("unauthorized_expenditure",)),
        (59, 59, ("other_control_or_compliance_observation",)),
        (60, 60, ("fund_utilization_or_liquidation",)),
        (61, 62, ()),
    ),
    "maramag-2023": (
        (1, 2, ("cash_or_bank_reconciliation",)),
        (3, 3, ("fund_utilization_or_liquidation", "unliquidated_cash_advance")),
        (4, 15, ("asset_record_reconciliation",)),
    ),
    "naga-city-2023": (
        (1, 3, ("fund_utilization_or_liquidation",)),
        (4, 4, ("unliquidated_cash_advance",)),
    ),
    "pasig-city-2024": (
        (1, 2, ("asset_record_reconciliation",)),
        (3, 3, ("inventory_count_or_record",)),
        (4, 4, ("cash_or_bank_reconciliation",)),
        (5, 5, ("other_control_or_compliance_observation",)),
        (6, 6, ("asset_record_reconciliation",)),
        (7, 7, ("inventory_count_or_record",)),
        (8, 8, ("other_control_or_compliance_observation",)),
    ),
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _observation_number(observation_id: str) -> int:
    match = re.search(r"-obs-(\d+)$", observation_id)
    if not match:
        raise ValueError(f"cannot parse observation number from {observation_id!r}")
    return int(match.group(1))


def labels_for(record: dict[str, Any]) -> Labels:
    doc_id = str(record["doc_id"])
    number = _observation_number(str(record["observation_id"]))
    matches = [
        labels
        for start, end, labels in RULES.get(doc_id, ())
        if start <= number <= end
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one label rule for {record['observation_id']}, "
            f"found {len(matches)}"
        )
    labels = matches[0]
    unknown = sorted(set(labels) - set(FINDING_LABELS))
    if unknown:
        raise ValueError(f"unsupported labels for {record['observation_id']}: {unknown}")
    return labels


def build_overlay(records: list[dict[str, Any]]) -> dict[str, Any]:
    unreviewed = [record for record in records if record.get("review_status") == "unreviewed"]
    decisions: list[dict[str, Any]] = []
    distribution: Counter[str] = Counter()

    for record in unreviewed:
        labels = labels_for(record)
        distribution.update(labels or ("reviewed_no_finding",))
        no_finding = not labels
        decision = {
            "observation_id": record["observation_id"],
            "lgu": CANONICAL_LGU_NAMES.get(str(record["lgu"]), str(record["lgu"])),
            "boundary_status": "owner_blanket_approved",
            "finding_labels": list(labels),
            "finding_decision": (
                "approved_reviewed_no_finding" if no_finding else "approved_blanket"
            ),
            "reviewed_no_finding": no_finding,
            "ner_reviewed": False,
            "relations_reviewed": False,
            "human_review_required": False,
            "adjudication_reason": (
                "Project owner blanket-approved the observation and its deterministic "
                "machine-proposed finding labels on 2026-08-11; this was not an "
                "item-by-item adjudication."
            ),
        }
        if record["observation_id"] == "la-union-province-2024-obs-001":
            decision["boundary_status"] = "owner_blanket_approved_with_correction"
            decision["boundary_note"] = (
                "Remove the next observation heading from recommendation_text."
            )
        decisions.append(decision)

    if len(decisions) != 255:
        raise ValueError(f"expected 255 unreviewed records, found {len(decisions)}")

    return {
        "status": "complete",
        "review_method": "project_owner_blanket_approval_of_rule_based_labels",
        "approval_date": "2026-08-11",
        "finding_review_complete": True,
        "ner_review_complete": False,
        "relations_review_complete": False,
        "training_eligible": True,
        "training_eligible_tasks": ["finding"],
        "supervision_quality": SUPERVISION_QUALITY,
        "repair_mojibake": True,
        "label_assignment_method": "deterministic_document_item_rules_v1",
        "reviewed_observations": len(decisions),
        "label_distribution": dict(sorted(distribution.items())),
        "text_corrections": {
            "la-union-province-2024-obs-001": {
                "recommendation_text_edits": [
                    {
                        "operation": "remove_suffix",
                        "text": (
                            " Unreliable Property, Plant and Equipment (PPE) "
                            "Account Balances"
                        ),
                    }
                ]
            }
        },
        "decisions": decisions,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    overlay = build_overlay(_read_jsonl(args.corpus))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(overlay, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {overlay['reviewed_observations']} blanket-approved decisions "
        f"to {args.output}"
    )
    print(json.dumps(overlay["label_distribution"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
