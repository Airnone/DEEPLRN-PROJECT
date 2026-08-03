"""
deeplrn.preprocessing.pdf_extractor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Extracts raw text from COA PDF reports.

Strategy
--------
1.  Use **pdfplumber** to pull text, along with page-level metadata
    (page number, bounding boxes of each word, detected section headers).
2.  If a page yields fewer than `ocr_min_chars` characters it is likely a
    scanned image — fall back to **Tesseract OCR** via pytesseract.
3.  Return a list of ``PageData`` objects that downstream modules can
    consume without re-opening the PDF.

Usage
-----
>>> from deeplrn.preprocessing.pdf_extractor import PDFExtractor
>>> extractor = PDFExtractor("path/to/report.pdf")
>>> pages = extractor.extract()
>>> pages[0].text[:80]
'Republic of the Philippines COMMISSION ON AUDIT ...'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber
from PIL import Image

from deeplrn.config import EXTRACT_CFG, ExtractionConfig

logger = logging.getLogger(__name__)

# ── Regex heuristics for section headers (typical COA report structure) ──────
_HEADER_PATTERNS: List[re.Pattern] = [
    re.compile(r"^(PART|SECTION|CHAPTER)\s+[IVXLCDM\d]+", re.IGNORECASE),
    re.compile(r"^\d+(?:\.\d+)*\.\s+[A-Z]"),          # "1. Intro" or "1.1 Background"
    re.compile(r"^[A-Z][A-Z0-9\s:,\-&]{4,}$"),        # ALL-CAPS (with digits/punct)
    re.compile(
        r"^(Observations?|Recommendations?|Status of Implementation)",
        re.IGNORECASE,
    ),
]


# ── Data containers ──────────────────────────────────────────────────────────

@dataclass
class WordBox:
    """A single word with its bounding box on the PDF page."""

    text: str
    x0: float
    top: float
    x1: float
    bottom: float

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return (self.x0, self.top, self.x1, self.bottom)


@dataclass
class PageData:
    """All information extracted from a single PDF page."""

    page_number: int
    """1-indexed page number."""

    text: str
    """Full extracted text (whitespace-normalised)."""

    word_boxes: List[WordBox] = field(default_factory=list)
    """Per-word bounding boxes (empty when text came from OCR)."""

    headers: List[str] = field(default_factory=list)
    """Detected section header strings on this page."""

    source: str = "pdfplumber"
    """'pdfplumber' or 'tesseract' — records how text was obtained."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Catch-all for extra info (e.g. page dimensions)."""


# ── Core extractor ───────────────────────────────────────────────────────────

class PDFExtractor:
    """Parse a single PDF file into a list of :class:`PageData` objects.

    Parameters
    ----------
    pdf_path : str | Path
        Path to the PDF file.
    config : ExtractionConfig, optional
        Overrides for OCR thresholds etc.  Defaults to the global
        ``EXTRACT_CFG`` singleton.
    """

    def __init__(
        self,
        pdf_path: str | Path,
        config: ExtractionConfig | None = None,
    ) -> None:
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.cfg = config or EXTRACT_CFG

    # ── public API ───────────────────────────────────────────────────────

    def extract(self) -> List[PageData]:
        """Extract text (with OCR fallback) from every page.

        Returns
        -------
        list[PageData]
            One entry per page, in order.
        """
        pages: List[PageData] = []

        with pdfplumber.open(self.pdf_path) as pdf:
            total = len(pdf.pages)
            logger.info("Opened %s — %d pages", self.pdf_path.name, total)

            for plumber_page in pdf.pages:
                page_num = plumber_page.page_number  # 1-indexed
                page_data = self._extract_page(plumber_page, page_num)
                pages.append(page_data)

                if page_num % 20 == 0 or page_num == total:
                    logger.info(
                        "  … processed page %d / %d (%s)",
                        page_num,
                        total,
                        page_data.source,
                    )

        logger.info(
            "Extraction complete: %d pages (%d via OCR)",
            len(pages),
            sum(1 for p in pages if p.source == "tesseract"),
        )
        return pages

    # ── internals ────────────────────────────────────────────────────────

    def _extract_page(
        self,
        plumber_page: pdfplumber.page.Page,
        page_num: int,
    ) -> PageData:
        """Extract a single page, falling back to OCR when needed."""

        # --- Try pdfplumber first ---
        raw_text = plumber_page.extract_text() or ""
        word_boxes = self._get_word_boxes(plumber_page)

        if len(raw_text.strip()) >= self.cfg.ocr_min_chars:
            text = _normalise_whitespace(raw_text)
            headers = _detect_headers(text)
            return PageData(
                page_number=page_num,
                text=text,
                word_boxes=word_boxes,
                headers=headers,
                source="pdfplumber",
                metadata={"width": plumber_page.width, "height": plumber_page.height},
            )

        # --- OCR fallback ---
        if not self.cfg.ocr_fallback:
            text = _normalise_whitespace(raw_text)
            logger.debug("Page %d: sparse text, OCR disabled — returning as-is", page_num)
            return PageData(
                page_number=page_num,
                text=text,
                word_boxes=word_boxes,
                headers=_detect_headers(text),
                source="pdfplumber",
                metadata={"width": plumber_page.width, "height": plumber_page.height},
            )

        logger.debug("Page %d: only %d chars — falling back to OCR", page_num, len(raw_text.strip()))
        return self._ocr_page(plumber_page, page_num)

    def _ocr_page(
        self,
        plumber_page: pdfplumber.page.Page,
        page_num: int,
    ) -> PageData:
        """Render the page to an image and run Tesseract."""
        try:
            import pytesseract  # lazy import — not needed if OCR never triggers
        except ImportError as exc:
            raise ImportError(
                "pytesseract is required for OCR fallback.  "
                "Install it with:  pip install pytesseract\n"
                "You also need the Tesseract binary — see "
                "https://github.com/tesseract-ocr/tesseract"
            ) from exc

        # pdfplumber can render a page to a PIL Image
        page_image = plumber_page.to_image(resolution=self.cfg.dpi)
        img: Image.Image = page_image.original

        try:
            ocr_text: str = pytesseract.image_to_string(img, lang=self.cfg.tesseract_lang)
        finally:
            img.close()

        text = _normalise_whitespace(ocr_text)
        headers = _detect_headers(text)

        return PageData(
            page_number=page_num,
            text=text,
            word_boxes=[],           # no bounding boxes from OCR text-only mode
            headers=headers,
            source="tesseract",
            metadata={"width": plumber_page.width, "height": plumber_page.height},
        )

    @staticmethod
    def _get_word_boxes(plumber_page: pdfplumber.page.Page) -> List[WordBox]:
        """Pull per-word bounding boxes from pdfplumber."""
        words = plumber_page.extract_words() or []
        return [
            WordBox(
                text=w["text"],
                x0=w["x0"],
                top=w["top"],
                x1=w["x1"],
                bottom=w["bottom"],
            )
            for w in words
        ]


# ── Utility helpers ──────────────────────────────────────────────────────────

def _normalise_whitespace(text: str) -> str:
    """Collapse runs of horizontal whitespace, normalise line endings, and strip."""
    # Normalise Windows line endings and non-breaking spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ")
    # Collapse horizontal whitespace (spaces and tabs) but preserve newlines.
    return re.sub(r"[ \t]+", " ", text).strip()


def _detect_headers(text: str) -> List[str]:
    """Return lines that look like section headers."""
    headers: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        for pat in _HEADER_PATTERNS:
            if pat.search(stripped):
                headers.append(stripped)
                break
    return headers
