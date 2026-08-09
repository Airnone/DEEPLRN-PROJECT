"""
deeplrn.preprocessing.docx_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts text from ``.docx`` COA report sections (e.g. "Part 2 - Observations
and Recommendations") using **python-docx**, producing the same
:class:`~deeplrn.preprocessing.pdf_extractor.PageData` objects that
:class:`~deeplrn.preprocessing.pdf_extractor.PDFExtractor` produces, so the
existing :class:`~deeplrn.preprocessing.chunker.DocumentChunker` and every
downstream consumer work unchanged regardless of source format.

Important limitation: DOCX has no fixed page geometry
------------------------------------------------------
A ``.docx`` is a *reflowable* document — where page breaks fall depends on
fonts, margins, and the rendering application, none of which are recorded in
the file. There is therefore no way to recover the page numbers a printed or
PDF-exported copy would have purely from the ``.docx`` XML.

What this extractor *can* do is honor **manual page breaks** the author
explicitly inserted (Word's Ctrl+Enter, stored as ``<w:br w:type="page"/>``)
and number the resulting sections sequentially starting at 1. If a document
has no manual page breaks, the whole document is returned as a single
"page". Treat ``PageData.page_number`` from this extractor as an
**author-inserted section index**, not a literal printed-page number —
this matters when annotating evidence pages, since a DOCX-derived page
number will generally not line up with the page count of a PDF export of
the same document.

There are also no per-word bounding boxes (DOCX is not a fixed-layout
format), so ``PageData.word_boxes`` is always empty and any layout features
that depend on it (bbox embeddings) are inert for DOCX-derived text — page
and section embeddings still work normally.

Tables are included
--------------------
COA report tables (e.g. observation/recommendation pairs, financial
breakdowns) are walked in document order alongside paragraphs via
``Document.iter_inner_content()`` so they aren't silently dropped. Each row
is flattened to ``cell | cell | cell`` text.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from deeplrn.preprocessing.pdf_extractor import PageData, _detect_headers, _normalise_whitespace

logger = logging.getLogger(__name__)


def _is_manual_page_break(paragraph: Paragraph) -> bool:
    """True if *paragraph* contains an explicit (author-inserted) page break."""
    for run in paragraph.runs:
        for br in run._element.findall(qn("w:br")):
            if br.get(qn("w:type")) == "page":
                return True
    return False


def _table_text(table: Table) -> str:
    lines = []
    for row in table.rows:
        cells = [cell.text.strip() for cell in row.cells]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


class DocxExtractor:
    """Parse a single ``.docx`` file into a list of :class:`PageData` objects.

    Parameters
    ----------
    docx_path : str | Path
        Path to the ``.docx`` file.
    """

    def __init__(self, docx_path: str | Path) -> None:
        self.docx_path = Path(docx_path)
        if not self.docx_path.exists():
            raise FileNotFoundError(f"DOCX not found: {self.docx_path}")

    def extract(self) -> List[PageData]:
        """Extract text (paragraphs + tables), split at manual page breaks.

        Returns
        -------
        list[PageData]
            One entry per author-inserted page section (see module
            docstring for what "page" means for a DOCX source).
        """
        document = Document(self.docx_path)

        pages: List[PageData] = []
        current_parts: List[str] = []

        def flush() -> None:
            if not current_parts:
                return
            text = _normalise_whitespace("\n".join(current_parts))
            if not text:
                return
            page_number = len(pages) + 1
            pages.append(
                PageData(
                    page_number=page_number,
                    text=text,
                    word_boxes=[],
                    headers=_detect_headers(text),
                    source="docx",
                    metadata={},
                )
            )

        for item in document.iter_inner_content():
            if isinstance(item, Paragraph):
                if _is_manual_page_break(item):
                    flush()
                    current_parts = []
                stripped = item.text.strip()
                if stripped:
                    current_parts.append(stripped)
            elif isinstance(item, Table):
                table_text = _table_text(item)
                if table_text:
                    current_parts.append(table_text)

        flush()

        if not pages:
            logger.warning("No extractable text found in %s", self.docx_path.name)

        logger.info(
            "Extracted %s — %d page section(s) (split on manual page breaks)",
            self.docx_path.name,
            len(pages),
        )
        return pages
