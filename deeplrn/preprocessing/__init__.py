"""
deeplrn.preprocessing
~~~~~~~~~~~~~~~~~~~~~

PDF text extraction and overlapping-window chunking.
"""

from deeplrn.preprocessing.pdf_extractor import PDFExtractor, PageData, WordBox
from deeplrn.preprocessing.chunker import DocumentChunker, TextChunk

__all__ = [
    "PDFExtractor",
    "PageData",
    "WordBox",
    "DocumentChunker",
    "TextChunk",
]
