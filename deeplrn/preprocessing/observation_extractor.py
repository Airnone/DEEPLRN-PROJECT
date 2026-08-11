"""Extract reviewable audit-observation candidates from COA report summaries.

This module performs structural segmentation only. It does not assign finding
labels or interpret an audit observation as evidence of misconduct. Candidate
records retain their source PDF, LGU, report year, and one-indexed PDF pages so
that a reviewer can verify every unit before annotation or training.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from deeplrn.config import ExtractionConfig
from deeplrn.preprocessing.pdf_extractor import PageData, PDFExtractor

logger = logging.getLogger(__name__)

OBSERVATION_CANDIDATE_SCHEMA_VERSION = 1
EXTRACTION_METHOD = "coa_summary_structure_v1"

_NUMBERED_ITEM = re.compile(r"^\s*(\d{1,2})\.\s+([A-Za-z].*)$")
_SUBITEM = re.compile(r"^\s*(?:\([a-z]\)|[a-z]\.|[•▪])\s*", re.IGNORECASE)

_RECOMMENDATION_PATTERNS = (
    re.compile(r"\bwe\s+(?:(?:also|still)\s+)?recommend(?:ed|bd)?\b", re.IGNORECASE),
    re.compile(r"\bwe\s+reiterate(?:d)?\b.*\brecommendation", re.IGNORECASE),
    re.compile(r"\bfor\s+the\s+(?:above|exceptions).*\brecommend(?:ed|ation)", re.IGNORECASE),
    re.compile(r"\bfor\s+the\s+above.*\bdeficien", re.IGNORECASE),
    re.compile(r"\bfor\s+the\s+exceptions\s+cited\s+above", re.IGNORECASE),
)

_SECTION_TRANSITION_PATTERNS = (
    re.compile(r"\bsignificant\s+(?:audit\s+)?observations?\s+and\s+recommendations?\b", re.IGNORECASE),
    re.compile(r"\bother\s+significant\s+observations?\s+and\s+recommendations?\b", re.IGNORECASE),
    re.compile(r"\bother\s+observations?\s+that\s+need\s+immediate\s+attention\b", re.IGNORECASE),
)

_STOP_PATTERNS = (
    re.compile(r"^summary\s+of\s+total\s+", re.IGNORECASE),
    re.compile(r"\bsummary\s+of\s+(?:total\s+)?audit\s+suspensions?", re.IGNORECASE),
    re.compile(r"\bstatus\s+of\s+implementation\s+of\s+prior\s+years?", re.IGNORECASE),
    re.compile(r"^status\s+of\s+implement\w*\s+of\s+prior\s+years?", re.IGNORECASE),
    re.compile(r"\bstatus\s+of\s+prior\s+years?", re.IGNORECASE),
)

_BOILERPLATE_PATTERNS = (
    re.compile(r"^the\s+audit\s+team\s+communicated\b", re.IGNORECASE),
)

_OBSERVATION_SIGNAL = re.compile(
    r"\b(?:discrepanc|could\s+not\s+be\s+ascertained|cannot\s+be\s+ascertained|"
    r"did\s+not\s+reconcile|not\s+reconciled|qualified\s+opinion\s+because|"
    r"deficienc|unliquidated|unsupported|overstated|understated)\b",
    re.IGNORECASE,
)

_AUDIT_ANCHOR = re.compile(
    r"\b(?:independent\s+auditor(?:'s|s)?\s+report|auditor(?:'s|s)?\s+opinion|"
    r"audit\s+opinion)\b.*\bfinancial\s+statements\b",
    re.IGNORECASE,
)

_EMBEDDED_BOUNDARY = re.compile(
    r"\s+(?=(?:"
    r"\d{1,2}\.\s+[A-Z]|"
    r"we\s+(?:(?:also|still)\s+)?recommend\w*|"
    r"[IVXLCDM]+\.\s+(?:Summary|Status)"
    r"))",
    re.IGNORECASE,
)

_ROMAN_MARKER = re.compile(r"^[IVXLCDM]+\.$", re.IGNORECASE)
_SHARED_RECOMMENDATION = re.compile(
    r"\bfor\s+the\s+(?:above|exceptions).*\b(?:deficien|exception|recommend)",
    re.IGNORECASE,
)
_SHARED_RECOMMENDATION_ITEM = re.compile(
    r"^(\d+)(?:\.[a-z])?\.\s+",
    re.IGNORECASE,
)


def _canonical(text: str) -> str:
    return (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("–", "-")
        .replace("—", "-")
    )


@dataclass(frozen=True)
class SourceLine:
    page_number: int
    text: str


@dataclass(frozen=True)
class ObservationCandidate:
    """One unlabeled observation/recommendation unit awaiting human review."""

    observation_id: str
    doc_id: str
    source_pdf: str
    lgu: str
    year: int
    section: str
    source_item: str | None
    page_start: int
    page_end: int
    page_numbers: tuple[int, ...]
    observation_text: str
    recommendation_text: str = ""
    review_status: str = "unreviewed"
    extraction_method: str = EXTRACTION_METHOD
    warnings: tuple[str, ...] = ()
    schema_version: int = OBSERVATION_CANDIDATE_SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["page_numbers"] = list(self.page_numbers)
        data["warnings"] = list(self.warnings)
        data["metadata"] = dict(self.metadata)
        return data


@dataclass
class _CandidateBuffer:
    section: str
    source_item: str | None
    observation_lines: list[SourceLine] = field(default_factory=list)
    recommendation_lines: list[SourceLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def all_lines(self) -> list[SourceLine]:
        return self.observation_lines + self.recommendation_lines


def _flatten_pages(pages: Sequence[PageData]) -> list[SourceLine]:
    split_lines: list[SourceLine] = []
    for page in pages:
        for raw_line in page.text.splitlines():
            for part in _EMBEDDED_BOUNDARY.split(raw_line.strip()):
                line = part.strip()
                if line and not _ROMAN_MARKER.fullmatch(line):
                    split_lines.append(SourceLine(page.page_number, line))

    lines: list[SourceLine] = []
    index = 0
    while index < len(split_lines):
        line = split_lines[index]
        if re.fullmatch(r"\d{1,2}\.", line.text) and index + 1 < len(split_lines):
            following = split_lines[index + 1]
            if following.text[:1].isupper():
                lines.append(SourceLine(line.page_number, f"{line.text} {following.text}"))
                index += 2
                continue
        lines.append(line)
        index += 1
    return lines


def _find_start(lines: Sequence[SourceLine]) -> int:
    """Find the most specific, latest summary-level audit-opinion anchor."""
    qualified = [
        index
        for index, line in enumerate(lines)
        if "qualified opinion" in _canonical(line.text).lower()
    ]
    if qualified:
        # Naga contains a full independent-auditor section and then an executive
        # summary. The final qualified-opinion occurrence is the compact summary
        # appropriate for candidate segmentation.
        target = qualified[-1]
        for index in range(target, max(-1, target - 12), -1):
            if _AUDIT_ANCHOR.search(_canonical(lines[index].text)):
                return index
        return target

    anchors = [
        index
        for index, line in enumerate(lines)
        if _AUDIT_ANCHOR.search(_canonical(line.text))
    ]
    if anchors:
        return anchors[-1]

    significant = [
        index
        for index, line in enumerate(lines)
        if any(pattern.search(_canonical(line.text)) for pattern in _SECTION_TRANSITION_PATTERNS)
    ]
    if significant:
        return significant[0]
    raise ValueError("no audit-opinion or significant-observation section was found")


def _is_section_transition(text: str) -> bool:
    return any(pattern.search(_canonical(text)) for pattern in _SECTION_TRANSITION_PATTERNS)


def _is_stop(text: str) -> bool:
    return any(pattern.search(_canonical(text)) for pattern in _STOP_PATTERNS)


def _is_boilerplate(text: str) -> bool:
    return any(pattern.search(_canonical(text)) for pattern in _BOILERPLATE_PATTERNS)


def _is_recommendation_start(text: str) -> bool:
    canonical = _canonical(text)
    return any(pattern.search(canonical) for pattern in _RECOMMENDATION_PATTERNS)


def _numbered_observation(text: str) -> tuple[str, str] | None:
    """Return an item number and text for a top-level observation line.

    Uppercase after the item marker distinguishes report observations such as
    ``2. The balance ...`` from numbered recommendations such as
    ``2. the City Accountant ...`` in the Bacoor summary.
    """
    match = _NUMBERED_ITEM.match(_canonical(text))
    if not match:
        return None
    item, body = match.groups()
    if not body[0].isupper():
        return None
    return item, body


def _join_lines(lines: Sequence[SourceLine]) -> str:
    """Undo common PDF wrapping while retaining list-item boundaries."""
    paragraphs: list[str] = []
    current = ""
    for line in lines:
        text = line.text.strip()
        if not text:
            continue
        begins_item = bool(_SUBITEM.match(_canonical(text)))
        if begins_item and current:
            paragraphs.append(current.strip())
            current = text
        elif not current:
            current = text
        elif current.endswith("-") and text[:1].islower():
            current += text
        else:
            current += " " + text
    if current:
        paragraphs.append(current.strip())
    return "\n".join(paragraphs).strip()


class ObservationExtractor:
    """Segment extracted COA pages into unlabeled observation candidates."""

    def extract(
        self,
        pages: Sequence[PageData],
        *,
        doc_id: str,
        source_pdf: str,
        lgu: str,
        year: int,
        metadata: Mapping[str, Any] | None = None,
    ) -> list[ObservationCandidate]:
        if not pages:
            return []

        lines = _flatten_pages(pages)
        start = _find_start(lines)
        section = "audit_opinion"
        current: _CandidateBuffer | None = None
        preamble: list[SourceLine] = []
        recommendation_mode = False
        buffers: list[_CandidateBuffer] = []

        def finalize() -> None:
            nonlocal current
            if current is None or not current.observation_lines:
                current = None
                return
            buffers.append(current)
            current = None

        for line in lines[start:]:
            text = line.text
            if _is_stop(text):
                finalize()
                break

            if _is_boilerplate(text):
                finalize()
                recommendation_mode = False
                preamble = []
                continue

            if _is_section_transition(text):
                section = "significant_observations"
                if current is None and preamble:
                    preamble_text = _join_lines(preamble)
                    if _OBSERVATION_SIGNAL.search(_canonical(preamble_text)):
                        current = _CandidateBuffer(
                            section="audit_opinion",
                            source_item=None,
                            observation_lines=list(preamble),
                            warnings=["unnumbered_observation"],
                        )
                preamble = []
                continue

            numbered = _numbered_observation(text)
            if numbered is not None:
                finalize()
                item, body = numbered
                current = _CandidateBuffer(
                    section=section,
                    source_item=item,
                    observation_lines=[SourceLine(line.page_number, body)],
                )
                recommendation_mode = False
                preamble = []
                continue

            if current is None:
                preamble.append(line)
                continue

            if _is_recommendation_start(text):
                recommendation_mode = True

            if recommendation_mode:
                current.recommendation_lines.append(line)
            else:
                current.observation_lines.append(line)

        else:
            finalize()

        self._distribute_shared_recommendations(buffers)
        candidates = [
            self._build_candidate(
                buffer,
                sequence=index,
                doc_id=doc_id,
                source_pdf=source_pdf,
                lgu=lgu,
                year=year,
                metadata=metadata or {},
            )
            for index, buffer in enumerate(buffers, start=1)
        ]
        if not candidates:
            raise ValueError(f"no observation candidates were extracted from {doc_id}")
        return candidates

    @staticmethod
    def _distribute_shared_recommendations(buffers: Sequence[_CandidateBuffer]) -> None:
        """Map an aggregate numbered recommendation block to its observations.

        Some summaries list several observations first and then provide one
        block such as ``For the above-cited deficiencies ...`` whose numbered
        recommendations correspond to observation items 1, 2, 3, and so on.
        """
        for donor_index, donor in enumerate(buffers):
            if not donor.recommendation_lines:
                continue
            heading = donor.recommendation_lines[0]
            if not _SHARED_RECOMMENDATION.search(_canonical(heading.text)):
                continue

            by_item: dict[str, list[SourceLine]] = {}
            active_item: str | None = None
            for line in donor.recommendation_lines[1:]:
                match = _SHARED_RECOMMENDATION_ITEM.match(_canonical(line.text))
                if match:
                    active_item = match.group(1)
                if active_item is not None:
                    by_item.setdefault(active_item, []).append(line)
            if not by_item:
                continue

            for target in buffers[:donor_index + 1]:
                if target.section != donor.section or target.source_item not in by_item:
                    continue
                target.recommendation_lines = [heading, *by_item[target.source_item]]
                target.warnings.append("shared_recommendation_block")

    @staticmethod
    def _build_candidate(
        buffer: _CandidateBuffer,
        *,
        sequence: int,
        doc_id: str,
        source_pdf: str,
        lgu: str,
        year: int,
        metadata: Mapping[str, Any],
    ) -> ObservationCandidate:
        observation_text = _join_lines(buffer.observation_lines)
        recommendation_text = _join_lines(buffer.recommendation_lines)
        all_lines = buffer.all_lines
        pages = tuple(sorted({line.page_number for line in all_lines}))
        warnings = list(buffer.warnings)
        if not recommendation_text:
            warnings.append("recommendation_not_detected")
        return ObservationCandidate(
            observation_id=f"{doc_id}-obs-{sequence:03d}",
            doc_id=doc_id,
            source_pdf=source_pdf,
            lgu=lgu,
            year=year,
            section=buffer.section,
            source_item=buffer.source_item,
            page_start=min(pages),
            page_end=max(pages),
            page_numbers=pages,
            observation_text=observation_text,
            recommendation_text=recommendation_text,
            warnings=tuple(warnings),
            metadata=dict(metadata),
        )


def load_manifest_documents(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("corpus manifest must contain a non-empty documents list")
    return [dict(document) for document in documents]


def _display_source_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def extract_manifest(
    manifest_path: str | Path,
    *,
    ocr_fallback: bool = False,
) -> tuple[list[ObservationCandidate], dict[str, Any]]:
    """Extract candidates from every document in a collected-corpus manifest."""
    manifest_path = Path(manifest_path)
    extractor = ObservationExtractor()
    candidates: list[ObservationCandidate] = []
    document_summaries: list[dict[str, Any]] = []

    for document in load_manifest_documents(manifest_path):
        required = ("doc_id", "lgu", "year", "local_path")
        missing = [key for key in required if key not in document]
        if missing:
            raise ValueError(f"manifest document is missing required fields: {missing}")
        pdf_path = manifest_path.parent / str(document["local_path"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"manifest PDF not found: {pdf_path}")

        if (
            document.get("extraction_profile") == "image_only_ocr_required"
            and not ocr_fallback
        ):
            logger.warning(
                "Skipping %s because its manifest marks it as image-only and OCR is disabled",
                document["doc_id"],
            )
            document_summaries.append(
                {
                    "doc_id": document["doc_id"],
                    "lgu": document["lgu"],
                    "year": int(document["year"]),
                    "pages": int(document.get("pages", 0)),
                    "extraction_status": "ocr_required",
                    "observation_candidates": 0,
                    "candidates_without_recommendations": 0,
                    "candidate_pages": [],
                }
            )
            continue

        pages = PDFExtractor(
            pdf_path,
            ExtractionConfig(ocr_fallback=ocr_fallback),
        ).extract()
        candidate_pages = pages
        candidate_page_range = document.get("candidate_page_range")
        if candidate_page_range is not None:
            if (
                not isinstance(candidate_page_range, list)
                or len(candidate_page_range) != 2
                or not all(isinstance(value, int) for value in candidate_page_range)
            ):
                raise ValueError(
                    f"candidate_page_range for {document['doc_id']} must be [start, end]"
                )
            page_start, page_end = candidate_page_range
            if page_start < 1 or page_end < page_start:
                raise ValueError(
                    f"candidate_page_range for {document['doc_id']} is invalid: "
                    f"{candidate_page_range}"
                )
            candidate_pages = [
                page for page in pages if page_start <= page.page_number <= page_end
            ]
            if not candidate_pages:
                raise ValueError(
                    f"candidate_page_range for {document['doc_id']} selected no PDF pages"
                )

        section_anchor = document.get("section_anchor")
        if section_anchor:
            anchor_page = candidate_pages[0].page_number
            candidate_pages = [
                PageData(page_number=anchor_page, text=str(section_anchor)),
                *candidate_pages,
            ]

        doc_id = str(document["doc_id"])
        try:
            document_candidates = extractor.extract(
                candidate_pages,
                doc_id=doc_id,
                source_pdf=_display_source_path(pdf_path),
                lgu=str(document["lgu"]),
                year=int(document["year"]),
                metadata={
                    "document_scope": document.get("document_scope", ""),
                    "audit_opinion": document.get("audit_opinion", ""),
                    "manifest_path": _display_source_path(manifest_path),
                    "candidate_page_range": candidate_page_range,
                    "source_text_quality": document.get("source_text_quality", "embedded_text"),
                },
            )
            extraction_status = "extracted"
        except ValueError as exc:
            expected_messages = {
                f"no observation candidates were extracted from {doc_id}",
                "no audit-opinion or significant-observation section was found",
            }
            if str(exc) not in expected_messages:
                raise
            logger.warning("%s; continuing with the remaining manifest documents", exc)
            document_candidates = []
            extraction_status = "no_candidates"
        candidates.extend(document_candidates)
        document_summaries.append(
            {
                "doc_id": document["doc_id"],
                "lgu": document["lgu"],
                "year": int(document["year"]),
                "pages": len(pages),
                "candidate_page_range": candidate_page_range,
                "extraction_status": extraction_status,
                "observation_candidates": len(document_candidates),
                "candidates_without_recommendations": sum(
                    not candidate.recommendation_text for candidate in document_candidates
                ),
                "candidate_pages": sorted(
                    {page for candidate in document_candidates for page in candidate.page_numbers}
                ),
            }
        )

    summary = {
        "schema_version": OBSERVATION_CANDIDATE_SCHEMA_VERSION,
        "extraction_method": EXTRACTION_METHOD,
        "manifest_path": _display_source_path(manifest_path),
        "documents": len(document_summaries),
        "observation_candidates": len(candidates),
        "review_status": "unreviewed",
        "interpretation_notice": (
            "Candidates are structural extractions of audit observations, not labels or "
            "determinations of misconduct, intent, liability, or guilt."
        ),
        "document_summaries": document_summaries,
    }
    return candidates, summary


def write_jsonl(candidates: Iterable[ObservationCandidate], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        for candidate in candidates:
            stream.write(json.dumps(candidate.to_dict(), ensure_ascii=False) + "\n")


def write_summary(summary: Mapping[str, Any], output_path: str | Path) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(summary), stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract unlabeled, page-linked observation candidates from a COA corpus."
    )
    parser.add_argument("--manifest", required=True, help="Collected-corpus manifest JSON")
    parser.add_argument("--output", required=True, help="Output candidate JSONL path")
    parser.add_argument(
        "--summary",
        help="Output QA summary JSON (default: <output stem>.summary.json)",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable Tesseract fallback for sparse pages (disabled by default)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    candidates, summary = extract_manifest(args.manifest, ocr_fallback=args.ocr)
    output_path = Path(args.output)
    summary_path = (
        Path(args.summary)
        if args.summary
        else output_path.with_name(f"{output_path.stem}.summary.json")
    )
    write_jsonl(candidates, output_path)
    write_summary(summary, summary_path)
    logger.info(
        "Extracted %d candidates from %d documents -> %s",
        len(candidates),
        summary["documents"],
        output_path,
    )


if __name__ == "__main__":
    main()
