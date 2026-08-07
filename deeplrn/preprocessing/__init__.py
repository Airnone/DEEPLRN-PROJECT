"""
deeplrn.preprocessing
~~~~~~~~~~~~~~~~~~~~~

PDF text extraction and overlapping-window chunking.
"""

from deeplrn.preprocessing.pdf_extractor import PDFExtractor, PageData, WordBox
from deeplrn.preprocessing.chunker import DocumentChunker, TextChunk, concatenate_pages

__all__ = [
    "PDFExtractor",
    "PageData",
    "WordBox",
    "DocumentChunker",
    "TextChunk",
    "concatenate_pages",
]
