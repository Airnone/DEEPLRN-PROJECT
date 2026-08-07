"""
deeplrn.cli
~~~~~~~~~~~

Command-line interface for the DEEPLRN preprocessing pipeline.

Usage::

    # Process a single PDF
    python -m deeplrn.cli --input report.pdf --output output/

    # Process an entire directory of PDFs
    python -m deeplrn.cli --input pdfs/ --output output/

    # Disable OCR fallback
    python -m deeplrn.cli --input report.pdf --output output/ --no-ocr
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from deeplrn.preprocessing.pdf_extractor import PDFExtractor, PageData
from deeplrn.preprocessing.chunker import DocumentChunker, TextChunk
from deeplrn.config import ExtractionConfig, ChunkConfig

logger = logging.getLogger("deeplrn")


def _serialize_pages(pages: List[PageData]) -> List[Dict[str, Any]]:
    """Convert PageData objects to JSON-friendly dicts."""
    return [
        {
            "page_number": p.page_number,
            "text": p.text,
            "text_length": len(p.text),
            "text_preview": p.text[:200] + ("…" if len(p.text) > 200 else ""),
            "source": p.source,
            "headers": p.headers,
            "word_count": len(p.word_boxes),
            "word_boxes": [
                {"text": word.text, "bbox": list(word.bbox)}
                for word in p.word_boxes
            ],
            "metadata": p.metadata,
        }
        for p in pages
    ]


def _serialize_chunks(chunks: List[TextChunk]) -> List[Dict[str, Any]]:
    """Convert TextChunk objects to JSON-friendly dicts."""
    return [
        {
            "chunk_id": c.chunk_id,
            "token_count": c.token_count,
            "page_numbers": c.page_numbers,
            "char_offset_start": c.char_offset_start,
            "char_offset_end": c.char_offset_end,
            "overlap_tokens_prev": c.overlap_tokens_prev,
            "text_preview": c.text[:200] + ("…" if len(c.text) > 200 else ""),
            "text": c.text,
            "token_ids": c.token_ids,
        }
        for c in chunks
    ]


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    extract_cfg: ExtractionConfig,
    chunk_cfg: ChunkConfig,
) -> Dict[str, Any]:
    """Run the full preprocessing pipeline on a single PDF.

    Returns the result dict (also written to disk as JSON).
    """
    logger.info("=" * 60)
    logger.info("Processing: %s", pdf_path.name)
    logger.info("=" * 60)

    t0 = time.perf_counter()

    # ── 1. Extract ───────────────────────────────────────────────────
    extractor = PDFExtractor(pdf_path, config=extract_cfg)
    pages = extractor.extract()

    t_extract = time.perf_counter() - t0

    # ── 2. Chunk ─────────────────────────────────────────────────────
    chunker = DocumentChunker(config=chunk_cfg)
    chunks = chunker.chunk_pages(pages)

    t_total = time.perf_counter() - t0

    # ── 3. Build result ──────────────────────────────────────────────
    result: Dict[str, Any] = {
        "document": {
            "filename": pdf_path.name,
            "total_pages": len(pages),
            "total_chunks": len(chunks),
            "ocr_pages": sum(1 for p in pages if p.source == "tesseract"),
            "extraction_time_s": round(t_extract, 2),
            "total_time_s": round(t_total, 2),
        },
        "pages": _serialize_pages(pages),
        "chunks": _serialize_chunks(chunks),
    }

    # ── 4. Write JSON ────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{pdf_path.stem}_preprocessed.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("  ✓ Wrote %s  (%d pages → %d chunks in %.1fs)",
                out_path.name, len(pages), len(chunks), t_total)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deeplrn-preprocess",
        description="Extract and chunk text from COA PDF reports.",
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to a single PDF or a directory of PDFs.",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Output directory for JSON results (default: ./output).",
    )
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="Disable Tesseract OCR fallback.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=384,
        help="Max sub-word tokens per chunk (default: 384).",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=64,
        help="Token overlap between consecutive chunks (default: 64).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging setup
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    input_path = Path(args.input)
    output_dir = Path(args.output)

    extract_cfg = ExtractionConfig(ocr_fallback=not args.no_ocr)
    chunk_cfg = ChunkConfig(max_tokens=args.max_tokens, overlap_tokens=args.overlap)

    # Collect PDF paths
    if input_path.is_file():
        pdfs = [input_path]
    elif input_path.is_dir():
        pdfs = sorted(input_path.glob("*.pdf"))
        if not pdfs:
            logger.error("No PDF files found in %s", input_path)
            sys.exit(1)
    else:
        logger.error("Input path does not exist: %s", input_path)
        sys.exit(1)

    logger.info("Found %d PDF(s) to process", len(pdfs))

    for pdf in pdfs:
        try:
            process_pdf(pdf, output_dir, extract_cfg, chunk_cfg)
        except Exception:
            logger.exception("Failed to process %s", pdf.name)

    logger.info("Done — all results in %s/", output_dir)


if __name__ == "__main__":
    main()
