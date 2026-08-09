"""Format-dispatching entry point for document extraction.

Picks :class:`~deeplrn.preprocessing.pdf_extractor.PDFExtractor` or
:class:`~deeplrn.preprocessing.docx_extractor.DocxExtractor` by file
extension so callers (the preprocessing CLI, ``deeplrn-prepare``) don't need
to know which source format they're looking at.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from deeplrn.config import ExtractionConfig
from deeplrn.preprocessing.docx_extractor import DocxExtractor
from deeplrn.preprocessing.pdf_extractor import PageData, PDFExtractor

SUPPORTED_SUFFIXES = (".pdf", ".docx")


def extract_document(
    path: str | Path, extract_cfg: ExtractionConfig | None = None
) -> List[PageData]:
    """Extract *path* into :class:`PageData` objects, dispatching on suffix.

    Parameters
    ----------
    path : str | Path
        A ``.pdf`` or ``.docx`` source document.
    extract_cfg : ExtractionConfig, optional
        Forwarded to :class:`PDFExtractor` (OCR thresholds etc.). Ignored
        for ``.docx`` sources, which have no OCR fallback concept.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PDFExtractor(path, config=extract_cfg).extract()
    if suffix == ".docx":
        return DocxExtractor(path).extract()
    raise ValueError(
        f"unsupported source document type {suffix!r} for {path}; "
        f"expected one of {SUPPORTED_SUFFIXES}"
    )
