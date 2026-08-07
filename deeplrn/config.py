"""
Central configuration constants for the DEEPLRN pipeline.

All magic numbers are gathered here so they can be tuned in a single place
without hunting through the codebase.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ChunkConfig:
    """Controls the overlapping-window chunking strategy."""

    max_tokens: int = 384
    """Maximum number of *sub-word* tokens per chunk (RoBERTa BPE)."""

    overlap_tokens: int = 64
    """Number of overlapping tokens between consecutive chunks."""

    tokenizer_name: str = "roberta-base"
    """Hugging Face model id used for sub-word tokenisation."""

    single_sentence_chunks: bool = False
    """Emit one sentence per chunk for the sentence-level baseline."""


@dataclass(frozen=True)
class ExtractionConfig:
    """Controls PDF text extraction behaviour."""

    ocr_fallback: bool = True
    """If True, pages with very little extractable text are re-processed
    with Tesseract OCR."""

    ocr_min_chars: int = 30
    """A page whose extracted text has fewer characters than this threshold
    is considered "empty" and triggers OCR."""

    tesseract_lang: str = "eng"
    """Tesseract language code."""

    dpi: int = 300
    """Resolution used when rendering a PDF page to an image for OCR."""


@dataclass(frozen=True)
class NERConfig:
    """NER tag schema used throughout the pipeline."""

    tags: List[str] = field(default_factory=lambda: [
        "O",
        "B-PERSON", "I-PERSON",
        "B-ORGANIZATION", "I-ORGANIZATION",
        "B-LGU", "I-LGU",
        "B-CONTRACTOR", "I-CONTRACTOR",
        "B-AMOUNT", "I-AMOUNT",
        "B-DATE", "I-DATE",
        "B-PROJECT", "I-PROJECT",
        "B-FINDING", "I-FINDING",
    ])


@dataclass(frozen=True)
class FindingClasses:
    """The five neutral audit-finding categories predicted by the model."""

    labels: List[str] = field(default_factory=lambda: [
        "unauthorized_expenditure",
        "unliquidated_cash_advance",
        "procurement_irregularity",
        "unsupported_disbursement",
        "contractor_related_concern",
    ])


@dataclass(frozen=True)
class RelationConfig:
    """Relation extraction settings."""

    relation_types: List[str] = field(default_factory=lambda: [
        "INVOLVES",
        "AMOUNT_OF",
        "ASSOCIATED_WITH",
    ])

    sentence_window: int = 12
    """Max sentence distance between two entities for a candidate relation."""


@dataclass(frozen=True)
class LayoutConfig:
    """Configuration for optional token-level document-layout features."""

    enabled: bool = True
    use_page_embeddings: bool = True
    use_section_embeddings: bool = True
    use_bbox_embeddings: bool = True
    max_pages: int = 512
    max_sections: int = 256
    bbox_bins: int = 1024
    feature_dropout: float = 0.1


# ── Convenience singleton instances ──────────────────────────────────────────
CHUNK_CFG = ChunkConfig()
EXTRACT_CFG = ExtractionConfig()
NER_CFG = NERConfig()
FINDING_CFG = FindingClasses()
# Backwards-compatible alias for code and prototype files written before the
# neutral terminology migration. New code should use FINDING_CFG.
VIOLATION_CFG = FINDING_CFG
RELATION_CFG = RelationConfig()
LAYOUT_CFG = LayoutConfig()

# Backwards-compatible class alias. It can be removed after old serialized
# configuration files have been migrated.
ViolationClasses = FindingClasses
