"""Canonical labels and validated annotation records for DEEPLRN.

The schema deliberately represents textual evidence, not legal conclusions.
Character offsets refer to the concatenated text produced by preprocessing and
use Python's half-open convention: ``start_char`` is inclusive and ``end_char``
is exclusive.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping
import json


SCHEMA_VERSION = 2

ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "LGU",
    "CONTRACTOR",
    "AMOUNT",
    "DATE",
    "PROJECT",
    "FINDING",
)

FINDING_LABELS = (
    "unauthorized_expenditure",
    "unliquidated_cash_advance",
    "procurement_irregularity",
    "unsupported_disbursement",
    "contractor_related_concern",
    "asset_record_reconciliation",
    "cash_or_bank_reconciliation",
    "inventory_count_or_record",
    "fund_utilization_or_liquidation",
    "other_control_or_compliance_observation",
)

RELATION_TYPES = (
    "INVOLVES",
    "AMOUNT_OF",
    "ASSOCIATED_WITH",
)

LEGACY_ENTITY_ALIASES = {"VIOLATION": "FINDING"}
LEGACY_FINDING_ALIASES = {
    "suspicious_contractor_activity": "contractor_related_concern",
    "asset_ppe_records": "asset_record_reconciliation",
    "cash_and_bank_reconciliation": "cash_or_bank_reconciliation",
    "inventory_records": "inventory_count_or_record",
    "fund_utilization_liquidation": "fund_utilization_or_liquidation",
    "other_control_compliance_observation": "other_control_or_compliance_observation",
}
LEGACY_RELATION_ALIASES = {"RESPONSIBLE_FOR": "ASSOCIATED_WITH"}


def normalize_entity_label(label: str) -> str:
    value = label.strip().upper()
    return LEGACY_ENTITY_ALIASES.get(value, value)


def normalize_finding_label(label: str) -> str:
    value = label.strip().lower()
    return LEGACY_FINDING_ALIASES.get(value, value)


def normalize_relation_label(label: str) -> str:
    value = label.strip().upper()
    return LEGACY_RELATION_ALIASES.get(value, value)


@dataclass(frozen=True)
class EntityAnnotation:
    entity_id: str
    label: str
    start_char: int
    end_char: int
    text: str = ""
    page_number: int | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EntityAnnotation":
        entity = cls(
            entity_id=str(raw["entity_id"]),
            label=normalize_entity_label(str(raw["label"])),
            start_char=int(raw["start_char"]),
            end_char=int(raw["end_char"]),
            text=str(raw.get("text", "")),
            page_number=(int(raw["page_number"]) if raw.get("page_number") is not None else None),
        )
        entity.validate()
        return entity

    def validate(self) -> None:
        if not self.entity_id:
            raise ValueError("entity_id must not be empty")
        if self.label not in ENTITY_TYPES:
            raise ValueError(f"unsupported entity label: {self.label}")
        if self.start_char < 0 or self.end_char <= self.start_char:
            raise ValueError(
                f"invalid character span for {self.entity_id}: "
                f"[{self.start_char}, {self.end_char})"
            )
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be one-indexed")


@dataclass(frozen=True)
class RelationAnnotation:
    relation_id: str
    head_entity_id: str
    tail_entity_id: str
    label: str
    evidence_page_numbers: tuple[int, ...] = ()

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RelationAnnotation":
        relation = cls(
            relation_id=str(raw["relation_id"]),
            head_entity_id=str(raw["head_entity_id"]),
            tail_entity_id=str(raw["tail_entity_id"]),
            label=normalize_relation_label(str(raw["label"])),
            evidence_page_numbers=tuple(int(x) for x in raw.get("evidence_page_numbers", [])),
        )
        relation.validate()
        return relation

    def validate(self) -> None:
        if not self.relation_id:
            raise ValueError("relation_id must not be empty")
        if not self.head_entity_id or not self.tail_entity_id:
            raise ValueError("relation endpoints must not be empty")
        if self.head_entity_id == self.tail_entity_id:
            raise ValueError("a relation cannot connect an entity to itself")
        if self.label not in RELATION_TYPES:
            raise ValueError(f"unsupported relation label: {self.label}")
        if any(page < 1 for page in self.evidence_page_numbers):
            raise ValueError("evidence pages must be one-indexed")


@dataclass(frozen=True)
class AnnotatedDocument:
    doc_id: str
    source_pdf: str
    lgu: str
    year: int
    finding_labels: tuple[str, ...]
    observation_text: str = ""
    recommendation_text: str = ""
    evidence_page_numbers: tuple[int, ...] = ()
    parent_doc_id: str = ""
    entities: tuple[EntityAnnotation, ...] = ()
    relations: tuple[RelationAnnotation, ...] = ()
    schema_version: int = SCHEMA_VERSION
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AnnotatedDocument":
        source_version = int(raw.get("schema_version", 1))
        if source_version not in (1, SCHEMA_VERSION):
            raise ValueError(
                f"unsupported annotation schema version {source_version}; "
                f"expected 1 or {SCHEMA_VERSION}"
            )
        if "finding_labels" in raw:
            finding_labels = tuple(
                normalize_finding_label(str(label)) for label in raw["finding_labels"]
            )
        elif "finding_label" in raw:
            finding_labels = (normalize_finding_label(str(raw["finding_label"])),)
        else:
            raise ValueError("finding_labels is required")
        metadata = dict(raw.get("metadata", {}))
        if source_version != SCHEMA_VERSION:
            metadata.setdefault("migrated_from_schema_version", source_version)
        document = cls(
            doc_id=str(raw["doc_id"]),
            source_pdf=str(raw["source_pdf"]),
            lgu=str(raw["lgu"]),
            year=int(raw["year"]),
            finding_labels=finding_labels,
            observation_text=str(raw.get("observation_text", "")),
            recommendation_text=str(raw.get("recommendation_text", "")),
            evidence_page_numbers=tuple(
                int(page) for page in raw.get("evidence_page_numbers", [])
            ),
            parent_doc_id=str(raw.get("parent_doc_id", "")),
            entities=tuple(EntityAnnotation.from_dict(x) for x in raw.get("entities", [])),
            relations=tuple(RelationAnnotation.from_dict(x) for x in raw.get("relations", [])),
            schema_version=SCHEMA_VERSION,
            metadata=metadata,
        )
        document.validate()
        return document

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported annotation schema version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.doc_id or not self.source_pdf or not self.lgu:
            raise ValueError("doc_id, source_pdf, and lgu are required")
        if self.year < 1900 or self.year > 2100:
            raise ValueError(f"implausible report year: {self.year}")
        if not self.finding_labels and not self.metadata.get("reviewed_no_finding", False):
            raise ValueError(
                "at least one finding label is required unless reviewed_no_finding is true"
            )
        if len(self.finding_labels) != len(set(self.finding_labels)):
            raise ValueError("finding labels must be unique within an observation")
        unsupported = sorted(set(self.finding_labels) - set(FINDING_LABELS))
        if unsupported:
            raise ValueError(f"unsupported finding labels: {unsupported}")
        if any(page < 1 for page in self.evidence_page_numbers):
            raise ValueError("evidence pages must be one-indexed")

        entity_ids = [entity.entity_id for entity in self.entities]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("entity IDs must be unique within a document")
        relation_ids = [relation.relation_id for relation in self.relations]
        if len(relation_ids) != len(set(relation_ids)):
            raise ValueError("relation IDs must be unique within a document")

        known_entities = set(entity_ids)
        for relation in self.relations:
            missing = {relation.head_entity_id, relation.tail_entity_id} - known_entities
            if missing:
                raise ValueError(
                    f"relation {relation.relation_id} references unknown entities: {sorted(missing)}"
                )

        ordered = sorted(self.entities, key=lambda item: (item.start_char, item.end_char))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start_char < previous.end_char:
                raise ValueError(
                    f"overlapping entities are not supported: "
                    f"{previous.entity_id} and {current.entity_id}"
                )

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["entities"] = [asdict(item) for item in self.entities]
        data["relations"] = [
            {**asdict(item), "evidence_page_numbers": list(item.evidence_page_numbers)}
            for item in self.relations
        ]
        return data

    @property
    def finding_label(self) -> str:
        """Legacy accessor for migrated records that have exactly one label."""
        if len(self.finding_labels) != 1:
            raise ValueError("this observation has multiple finding labels")
        return self.finding_labels[0]

    @property
    def training_text(self) -> str:
        """Observation-level model input, including recommendation when present."""
        if not self.observation_text:
            return ""
        if not self.recommendation_text:
            return self.observation_text
        return f"{self.observation_text}\n\nRecommendation:\n{self.recommendation_text}"


def load_annotation(path: str | Path) -> AnnotatedDocument:
    with Path(path).open("r", encoding="utf-8") as stream:
        return AnnotatedDocument.from_dict(json.load(stream))


def load_annotations(paths: Iterable[str | Path]) -> List[AnnotatedDocument]:
    return [load_annotation(path) for path in paths]


def label_schema() -> Dict[str, Any]:
    """Return the label snapshot stored inside every trained checkpoint."""
    return {
        "schema_version": SCHEMA_VERSION,
        "entity_types": list(ENTITY_TYPES),
        "finding_labels": list(FINDING_LABELS),
        "finding_task": "multi_label",
        "relation_types": list(RELATION_TYPES),
    }
